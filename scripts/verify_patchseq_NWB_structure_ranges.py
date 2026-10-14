"""Independently check HTTP receipts and reconstruct partial structure offline."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import yaml
from verify_patchseq_NWB_metadata_pilot import SavedRanges, approved_locator, equal, sha

REG = Path('docs/validation-results/post2008-patchseq-NWB-structure-range-registration-1274.yaml')
RESULT = Path('results/patchseq-NWB-structure-range-1275/manifest.yaml')
OUTPUT = Path('results/patchseq-NWB-structure-range-independent-1277/manifest.yaml')


def reconstruct(file, policy, h5py, output):
    """Independent traversal; no structural collector import or value access."""
    visited = {}

    def names(keys):
        result = sorted(keys)
        if any(len(key.encode()) > policy['maximum_name_utf8_bytes'] for key in result):
            raise ValueError('name bound exceeded')
        return result

    def emit(record):
        if len(output) >= policy['maximum_unique_object_paths']:
            raise ValueError('record bound exceeded')
        output.append(record)

    def walk(node, path, remaining):
        previous = visited.get(path)
        if previous is not None and previous >= remaining:
            return
        visited[path] = remaining
        attributes = names(node.attrs.keys())
        if len(attributes) > policy['maximum_attribute_names_per_object']:
            raise ValueError('attribute bound exceeded')
        descriptor = {'path': path, 'link_type': 'HardLink', 'attribute_names': attributes}
        if isinstance(node, h5py.Group):
            descriptor['object_type'] = 'group'
        elif isinstance(node, h5py.Dataset):
            descriptor.update(object_type='dataset', dataset_shape=None if node.shape is None else list(node.shape),
                              dataset_dtype=str(node.dtype))
        else:
            raise TypeError('unsupported object')
        if previous is None:
            emit(descriptor)
        if not isinstance(node, h5py.Group) or remaining == 0:
            return
        for name in names(node.keys()):
            child_path = path.rstrip('/')+'/'+name
            link = node.get(name, getlink=True)
            if type(link) is not h5py.HardLink or name in policy['excluded_descent']:
                emit({'path': child_path, 'link_type': type(link).__name__, 'dereferenced': False})
            else:
                walk(node[name], child_path, remaining-1)

    for name in names(file.keys()):
        emit({'path': '/'+name, 'link_type': type(file.get(name, getlink=True)).__name__, 'root_name_only': True})
    for root_path in sorted(policy['inspect_metadata_roots']):
        cursor = file
        for name in root_path.strip('/').split('/'):
            if not isinstance(cursor, h5py.Group):
                raise TypeError('metadata parent not group')
            link = cursor.get(name, getlink=True)
            if link is None:
                emit({'path': root_path, 'present': False})
                break
            if type(link) is not h5py.HardLink:
                emit({'path': root_path, 'link_type': type(link).__name__, 'dereferenced': False})
                break
            cursor = cursor[name]
        else:
            walk(cursor, root_path, policy['maximum_descendant_depth_per_root'])


def check_receipts(report, pilot, reg, locator):
    ranges, pins = [], {}
    for receipt in pilot['HTTP_receipts']:
        if receipt['kind'] == 'RANGE':
            path = Path(receipt['body_path'])
            if sha(path) != receipt['body_sha256']:
                raise ValueError('seed block changed')
            ranges.append((receipt['start'], path.read_bytes()))
    etags = {r['header_fields']['etag'][0] for r in pilot['HTTP_receipts'] if r['kind'] == 'RANGE'}
    if len(etags) != 1:
        raise ValueError('ambiguous seed ETag')
    etag = etags.pop()
    head, new_bytes, count, seen, head_ok = 0, 0, 0, {s for s, _ in ranges}, False
    limits = reg['HTTP_range_limits']
    for receipt in report['HTTP_receipts']:
        target = urlsplit(locator)
        if receipt['URL_host'] != target.hostname or receipt['URL_path'] != target.path:
            raise ValueError('unapproved receipt target')
        for kind in ('headers', 'body'):
            path = Path(receipt[kind+'_path'])
            if sha(path) != receipt[kind+'_sha256'] or path.stat().st_size != receipt[kind+'_bytes']:
                raise ValueError('receipt bytes changed')
            pins[path] = receipt[kind+'_sha256']
        headers_path = Path(receipt['headers_path'])
        if headers_path.stat().st_size > 65536:
            raise ValueError('header bound exceeded')
        lines = headers_path.read_text().replace('\r\n', '\n').strip().split('\n\n')[-1].splitlines()
        status, fields = int(lines[0].split()[1]), {}
        for line in lines[1:]:
            key, value = line.split(':', 1)
            fields.setdefault(key.lower(), []).append(value.strip())
        equal(fields, receipt['header_fields'])
        if (status != receipt['http_status'] or receipt['client_exit_code'] != 0
            or receipt['status'] != 'response-retained' or fields.get('content-encoding', ['identity']) != ['identity']):
            raise ValueError('receipt was not retained successfully')
        if receipt['kind'] == 'HEAD':
            head += 1
            if head_ok or head > reg['URL_resolution']['maximum_HEAD_requests']:
                raise ValueError('extra HEAD')
            if status in (301, 302, 307, 308):
                if head > reg['URL_resolution']['maximum_HEAD_redirects'] or len(fields.get('location', [])) != 1:
                    raise ValueError('redirect bound exceeded')
                redirected = urljoin(locator, fields['location'][0])
                if not approved_locator(redirected, reg['URL_resolution']) or urlsplit(redirected).path != target.path:
                    raise ValueError('unapproved redirect')
                locator = redirected
                continue
            if status != 200 or fields.get('etag') != [etag] or fields.get('content-length') != [str(pilot['asset']['size'])]:
                raise ValueError('immutable HEAD differs')
            head_ok = True
        elif receipt['kind'] == 'RANGE':
            start, end = receipt['start'], receipt['end']
            if (not head_ok or type(start) is not int or type(end) is not int or start < 0
                or start in seen or start % limits['cache_block_bytes']
                or end != min(start+limits['cache_block_bytes'], pilot['asset']['size'])-1
                or status != 206 or fields.get('etag') != [etag]
                or fields.get('content-range') != [f'bytes {start}-{end}/{pilot["asset"]["size"]}']):
                raise ValueError('new range differs or duplicates seed')
            body = Path(receipt['body_path']).read_bytes()
            if len(body) != end-start+1:
                raise ValueError('range payload size differs')
            seen.add(start)
            ranges.append((start, body))
            count += 1
            new_bytes += len(body)
        else:
            raise ValueError('unsupported receipt kind')
    if not head_ok or count > limits['maximum_range_requests'] or new_bytes > limits['maximum_unique_downloaded_bytes']:
        raise ValueError('transport bounds exceeded')
    if (count != report['new_range_requests'] or new_bytes != report['charged_new_range_bytes']
        or sum(len(data) for _, data in ranges) != report['combined_cache_bytes']
        or report['combined_cache_bytes'] > reg['seed_cache']['maximum_combined_cache_bytes']):
        raise ValueError('cache accounting differs')
    elapsed = (datetime.fromisoformat(report['finished_at_utc'])-datetime.fromisoformat(report['started_at_utc'])).total_seconds()
    if not 0 <= elapsed <= limits['maximum_run_seconds']:
        raise ValueError('deadline exceeded')
    return ranges, pins


def main():
    fixed = {REG: 'e5ad7da418565cc19da5a80e08d15fbc910922455b693d8f7a040a301939beee',
             RESULT: '15a0b50a71e15772ed2803a0c5e7a5934185ede7928d7104c4fc6efb13d24700'}
    if any(sha(p) != digest for p, digest in fixed.items()):
        raise ValueError('sealed evidence changed')
    reg = yaml.safe_load(REG.read_text())
    report = yaml.safe_load(RESULT.read_text())
    fixed.update({Path(p): digest for p, digest in report['source_hashes'].items()})
    if any(sha(p) != digest for p, digest in fixed.items()):
        raise ValueError('input source changed')
    pilot = yaml.safe_load(Path(reg['pilot']).read_text())
    equal(report['asset'], pilot['asset'])
    if (report['status'] != 'bounded-structural-acquisition-incomplete'
        or report['error'] != {'type': 'ValueError', 'message': 'registered range budget exhausted'}
        or report['pinned_sources_unchanged'] is not True
        or any(report[k] is not False for k in ('dataset_values_read', 'attribute_values_read', 'physiological_arrays_read', 'specimen_eligibility_established'))):
        raise ValueError('terminal scope or outcome differs')
    raw = json.loads((Path('results/patchseq-archive-identity-metadata-1239')/pilot['asset']['asset_id']/'response.json').read_bytes())
    candidates = {url for url in raw['contentUrl'] if approved_locator(url, reg['URL_resolution'])}
    if len(candidates) != 1:
        raise ValueError('unique original locator required')
    ranges, receipts = check_receipts(report, pilot, reg, candidates.pop())
    fixed.update(receipts)
    limits = reg['HTTP_range_limits']
    if report['new_range_requests'] != limits['maximum_range_requests'] and report['charged_new_range_bytes'] != limits['maximum_unique_downloaded_bytes']:
        raise ValueError('reported exhausted budget not actually exhausted')
    if version('h5py') != reg['parser_dependency'].split('==')[1]:
        raise ValueError('registered parser differs')
    os.environ['HDF5_PLUGIN_PRELOAD'] = '::'
    import h5py
    policy = yaml.safe_load(Path(reg['structural_registration']).read_text())['structure_policy']
    records, error = [], None
    with SavedRanges(pilot['asset']['size'], ranges) as stream:
        try:
            with h5py.File(stream, 'r') as file:
                reconstruct(file, policy, h5py, records)
        except ValueError as exc:
            error = str(exc)
        spans = stream.read_spans
    if error != 'offline replay attempted an uncovered byte span':
        raise ValueError('budget exhaustion not independently reproduced as an uncovered read')
    equal(records, report['records'])
    if any(sha(p) != digest for p, digest in fixed.items()):
        raise ValueError('sources changed during independent replay')
    OUTPUT.parent.mkdir()
    with OUTPUT.open('x') as handle:
        yaml.safe_dump({'status': 'partial-structure-and-range-receipts-independently-verified',
            'verifier_sha256': sha(__file__), 'source_hashes': {str(p): digest for p, digest in fixed.items()},
            'records': records, 'offline_error': error, 'offline_read_spans': spans,
            'new_ranges_verified': report['new_range_requests'], 'new_bytes_verified': report['charged_new_range_bytes'],
            'verified_at_utc': datetime.now(UTC).isoformat(), 'network_requests': 0,
            'structural_scope_complete': False, 'collector_imported': False,
            'dataset_values_read': False, 'attribute_values_read': False, 'specimen_eligibility_established': False}, handle, sort_keys=False)
    print('Partial structural records and all new receipts independently verified offline')


if __name__ == '__main__':
    main()
