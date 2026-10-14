"""Verify immutable range receipts and replay whitelisted metadata offline."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit

import yaml

REG = Path('docs/validation-results/post2008-patchseq-NWB-metadata-pilot-registration-1266.yaml')
RESULT = Path('results/patchseq-NWB-metadata-pilot-1267/manifest.yaml')
OUTPUT = Path('results/patchseq-NWB-metadata-pilot-independent-1269/manifest.yaml')


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def equal(expected, observed):
    if json.dumps(expected, sort_keys=True) != json.dumps(observed, sort_keys=True):
        raise ValueError('type-preserving offline replay mismatch')


class SavedRanges(io.RawIOBase):
    def __init__(self, size, ranges):
        super().__init__()
        self.size, self.ranges, self.offset = size, sorted(ranges), 0
        self.read_spans = []
        previous = -1
        for start, data in self.ranges:
            if start <= previous or not data or start+len(data) > size:
                raise ValueError('overlapping or invalid saved range')
            previous = start+len(data)-1

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.offset

    def seek(self, offset, whence=0):
        if whence not in (0, 1, 2):
            raise ValueError('invalid seek mode')
        target = offset + {0: 0, 1: self.offset, 2: self.size}[whence]
        if target < 0:
            raise ValueError('negative seek')
        self.offset = target
        return target

    def readinto(self, buffer):
        self._checkClosed()
        required = min(len(buffer), max(0, self.size-self.offset))
        begin, copied = self.offset, 0
        while copied < required:
            candidates = [(start, data) for start, data in self.ranges if start <= self.offset < start+len(data)]
            if len(candidates) != 1:
                raise ValueError('offline replay attempted an uncovered byte span')
            start, data = candidates[0]
            count = min(required-copied, start+len(data)-self.offset)
            buffer[copied:copied+count] = data[self.offset-start:self.offset-start+count]
            copied += count
            self.offset += count
        self.read_spans.append([begin, self.offset])
        return copied

    def write(self, data):
        raise io.UnsupportedOperation('offline replay is read-only')

    def truncate(self, size=None):
        raise io.UnsupportedOperation('offline replay is read-only')


def reconstruct_scalar(raw, maximum):
    if hasattr(raw, 'shape'):
        if raw.shape not in ((), (1,)):
            raise ValueError('unapproved scalar shape')
        raw = raw.item()
    if type(raw) is bytes:
        if len(raw) > maximum:
            raise ValueError('oversized bytes')
        return {'raw_type': 'bytes', 'raw_hex': raw.hex(), 'decoded_text': raw.decode('utf-8')}
    if type(raw) is str:
        if len(raw.encode()) > maximum:
            raise ValueError('oversized text')
        return {'raw_type': 'str', 'value': raw}
    if type(raw) in (bool, int, float):
        if type(raw) is float and not math.isfinite(raw):
            raise ValueError('nonfinite scalar')
        return {'raw_type': type(raw).__name__, 'value': raw}
    raise TypeError('unsupported scalar')


def replay(stream, policy, h5py):
    result = {'root_attributes': {}, 'datasets': {}, 'logical_dataset_reads': []}
    maximum = policy['maximum_decoded_value_bytes']
    with h5py.File(stream, 'r') as file:
        for key in policy['root_attributes']:
            result['root_attributes'][key] = {'present': key in file.attrs}
            if key in file.attrs:
                identity = file.attrs.get_id(key)
                if identity.shape not in ((), (1,)) or identity.get_storage_size() > maximum:
                    raise ValueError('attribute bound exceeded')
                result['root_attributes'][key]['scalar'] = reconstruct_scalar(file.attrs[key], maximum)
        for path in policy['datasets']:
            parent, present = file, True
            for key in path[1:].split('/'):
                if not isinstance(parent, h5py.Group):
                    raise TypeError('invalid metadata parent')
                link = parent.get(key, getlink=True)
                if link is None:
                    present = False
                    break
                if type(link) is not h5py.HardLink:
                    raise TypeError('non-hard link rejected before dereferencing')
                parent = parent[key]
            record = {'present': present}
            if present:
                if not isinstance(parent, h5py.Dataset):
                    raise TypeError('metadata dataset required')
                if (parent.shape not in ((), (1,)) or parent.compression is not None or parent.is_virtual
                    or parent.external or parent.id.get_storage_size() > maximum):
                    raise ValueError('unsupported metadata storage')
                result['logical_dataset_reads'].append(path)
                record.update(shape=list(parent.shape), dtype=str(parent.dtype), scalar=reconstruct_scalar(parent[()], maximum))
            result['datasets'][path] = record
    return result


def approved_locator(url, policy):
    if type(url) is not str:
        return False
    parsed = urlsplit(url)
    try:
        port = parsed.port
    except ValueError:
        return False
    return (parsed.scheme == policy['allowed_scheme'] and parsed.hostname in policy['allowed_hosts']
            and parsed.username is None and parsed.password is None and port in (None, 443)
            and not parsed.fragment and parsed.path.startswith(policy['required_path_prefix'])
            and not any(p in ('.', '..') for p in unquote(parsed.path).split('/')))


def verify_ranges(report, reg, locator):
    ranges, head_count, etag, pins = [], 0, None, {}
    limit = reg['HTTP_range_limits']
    for receipt in report['HTTP_receipts']:
        target = urlsplit(locator)
        if receipt['URL_host'] != target.hostname or receipt['URL_path'] != target.path:
            raise ValueError('HTTP target differs from exact selected locator')
        for kind in ('headers', 'body'):
            path, expected = Path(receipt[kind+'_path']), receipt[kind+'_sha256']
            if sha(path) != expected or path.stat().st_size != receipt[kind+'_bytes']:
                raise ValueError('HTTP raw receipt changed')
            pins[path] = expected
        lines = Path(receipt['headers_path']).read_text().replace('\r\n', '\n').strip().split('\n\n')[-1].splitlines()
        code = int(lines[0].split()[1])
        fields = {}
        for line in lines[1:]:
            key, value = line.split(':', 1)
            fields.setdefault(key.lower(), []).append(value.strip())
        equal(fields, receipt['header_fields'])
        if code != receipt['http_status'] or receipt['client_exit_code'] != 0 or receipt['status'] != 'response-retained':
            raise ValueError('nonterminal or failed HTTP receipt')
        if receipt['URL_host'] not in reg['URL_resolution']['allowed_hosts'] or not receipt['URL_path'].startswith('/blobs/'):
            raise ValueError('unapproved HTTP target')
        if fields.get('content-encoding', ['identity']) != ['identity']:
            raise ValueError('unexpected content encoding')
        if receipt['kind'] == 'HEAD':
            head_count += 1
            if head_count > reg['URL_resolution']['maximum_HEAD_requests'] or etag is not None:
                raise ValueError('extra or repeated HEAD request')
            if code in (301, 302, 307, 308):
                locations = fields.get('location', [])
                if head_count > reg['URL_resolution']['maximum_HEAD_redirects'] or len(locations) != 1:
                    raise ValueError('HEAD redirect budget exceeded')
                new_locator = urljoin(locator, locations[0])
                if not approved_locator(new_locator, reg['URL_resolution']) or urlsplit(new_locator).path != target.path:
                    raise ValueError('unapproved HEAD redirect target')
                locator = new_locator
                continue
            if code != 200 or fields.get('content-length') != [str(report['asset']['size'])] or len(fields.get('etag', [])) != 1:
                raise ValueError('immutable successful HEAD required')
            etag = fields['etag'][0]
            if len(etag) < 3 or not etag.startswith('"') or not etag.endswith('"') or '"' in etag[1:-1]:
                raise ValueError('strong ETag required')
        elif receipt['kind'] == 'RANGE':
            start, end = receipt['start'], receipt['end']
            if (etag is None or code != 206 or fields.get('etag') != [etag]
                or fields.get('content-range') != [f'bytes {start}-{end}/{report["asset"]["size"]}']
                or type(start) is not int or type(end) is not int or start < 0 or end < start
                or start % limit['cache_block_bytes'] != 0
                or end != min(start+limit['cache_block_bytes'], report['asset']['size'])-1):
                raise ValueError('immutable aligned range mismatch')
            body = Path(receipt['body_path']).read_bytes()
            if len(body) != end-start+1:
                raise ValueError('range payload size mismatch')
            ranges.append((start, body))
        else:
            raise ValueError('unexpected HTTP method')
    downloaded = sum(len(data) for _, data in ranges)
    if (head_count < 1 or etag is None or len(ranges) > limit['maximum_range_requests'] or downloaded > limit['maximum_unique_downloaded_bytes']
        or len(ranges) != report['range_requests'] or downloaded != report['unique_downloaded_range_bytes']
        or downloaded != report['unique_cached_range_bytes'] or downloaded != report['budget_charged_range_bytes']):
        raise ValueError('pilot budgets or accounting differ')
    duration = datetime.fromisoformat(report['finished_at_utc'])-datetime.fromisoformat(report['started_at_utc'])
    if duration.total_seconds() < 0 or duration.total_seconds() > limit['maximum_run_seconds']:
        raise ValueError('pilot duration out of bounds')
    return ranges, pins


def main():
    pins = {REG: '47b1b01cc3b3bb8cb6ec011e7b374a2fa5650e47221157b6cb1ec5850f1f9733',
            RESULT: 'c1953a806b1ce34aa10f0cbd3067c80486ca7bb8f594ba2b08ca8847d0e3a72a',
            Path('scripts/run_patchseq_NWB_metadata_pilot.py'): 'c50fc863b901db0c179314c90f9730e1fba0aee9f759788da002d85817d5fa66',
            Path('scripts/patchseq_bounded_NWB.py'): '722128c416bf640247e5eb9aac1284c16d570e3785452ed4037c1a7392019206'}
    reg = yaml.load(REG.read_text(), Loader=yaml.CSafeLoader)
    report = yaml.load(RESULT.read_text(), Loader=yaml.CSafeLoader)
    pins.update({Path(p): h for p, h in report['source_hashes'].items()})
    if any(sha(p) != h for p, h in pins.items()):
        raise ValueError('pinned pilot evidence changed')
    if (report['status'] != 'bounded-NWB-metadata-inspected-independent-replay-pending'
        or report['registration_sha256'] != pins[REG] or report['all_pinned_sources_unchanged'] is not True
        or report['physiological_arrays_read'] is not False or report['specimen_eligibility_established'] is not False
        or report['complete_file_digest_independently_verified'] is not False):
        raise ValueError('terminal pilot scope differs')
    catalog = yaml.load(Path(reg['catalog']).read_text(), Loader=yaml.CSafeLoader)['assets']
    equal(min(catalog, key=lambda a: (a['path'], a['asset_id'])), report['asset'])
    source = Path(reg['acquisition_manifest']).parent/report['asset']['asset_id']/'response.json'
    if source not in pins or sha(source) != pins[source]:
        raise ValueError('original locator metadata is not pinned')
    raw = json.loads(source.read_bytes())
    candidates = []
    for label, container in [('top_level', raw), ('literal_metadata_object', raw.get('metadata'))]:
        if type(container) is not dict or 'contentUrl' not in container:
            continue
        values = container['contentUrl']
        if type(values) is list:
            candidates.extend({'container': label, 'value': v, 'permitted': approved_locator(v, reg['URL_resolution'])} for v in values)
        else:
            candidates.append({'container': label, 'value': values, 'permitted': False})
    equal(candidates, report['URL_candidates'])
    locators = {c['value'] for c in candidates if c['permitted']}
    if len(locators) != 1:
        raise ValueError('unique exact locator required')
    ranges, raw_pins = verify_ranges(report, reg, locators.pop())
    pins.update(raw_pins)
    if version('h5py') != reg['parser_dependency'].split('==')[1]:
        raise ValueError('registered HDF5 parser version differs')
    os.environ['HDF5_PLUGIN_PRELOAD'] = '::'
    import h5py
    with SavedRanges(report['asset']['size'], ranges) as stream:
        inspection = replay(stream, reg['logical_metadata_reads'], h5py)
        equal(inspection, report['inspection'])
        spans = stream.read_spans
    expected = {'/general/subject/subject_id': report['asset']['path'].split('/')[0][4:],
                '/session_id': report['asset']['path'].split('_ses-')[1].removesuffix('_icephys.nwb')}
    flags = {}
    for path, token in expected.items():
        item = inspection['datasets'][path].get('scalar', {})
        value = item.get('decoded_text') if item.get('raw_type') == 'bytes' else item.get('value') if item.get('raw_type') == 'str' else None
        flags[path] = value == token if value is not None else None
    equal(flags, report['literal_path_identity_diagnostics'])
    if any(sha(p) != h for p, h in pins.items()):
        raise ValueError('evidence changed during offline replay')
    OUTPUT.parent.mkdir()
    with OUTPUT.open('x') as handle:
        yaml.safe_dump({'status': 'bounded-NWB-ranges-and-metadata-independently-replayed-offline',
            'verifier_sha256': sha(__file__), 'source_hashes': {str(p): h for p, h in pins.items()},
            'inspection': inspection, 'literal_path_identity_diagnostics': flags, 'offline_read_spans': spans,
            'range_bytes': sum(len(data) for _, data in ranges), 'range_count': len(ranges),
            'verified_at_utc': datetime.now(UTC).isoformat(), 'pilot_collector_imported': False,
            'network_requests': 0, 'physiological_arrays_read': False,
            'complete_file_digest_independently_verified': False, 'specimen_eligibility_established': False}, handle, sort_keys=False)
    print('All range receipts and whitelisted metadata independently replayed offline')


if __name__ == '__main__':
    main()
