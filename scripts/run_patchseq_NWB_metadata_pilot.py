"""Sealed first-file range-backed NWB metadata pilot; no physiological arrays."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import yaml
from patchseq_bounded_NWB import (
    BoundedReader,
    inspect_metadata,
    permitted_url,
    select_url,
    url_candidates,
)

REG = Path('docs/validation-results/post2008-patchseq-NWB-metadata-pilot-registration-1266.yaml')
REG_HASH = '47b1b01cc3b3bb8cb6ec011e7b374a2fa5650e47221157b6cb1ec5850f1f9733'


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def parse_headers(path):
    if path.stat().st_size > 65536:
        raise ValueError('HTTP header bound exceeded')
    lines = path.read_text().replace('\r\n', '\n').strip().split('\n\n')[-1].splitlines()
    status = int(lines[0].split()[1])
    headers = {}
    for line in lines[1:]:
        key, value = line.split(':', 1)
        headers.setdefault(key.lower(), []).append(value.strip())
    return status, headers


class Transport:
    def __init__(self, root, url, asset, reg):
        self.root, self.url, self.asset, self.reg = root, url, asset, reg
        self.started = time.monotonic()
        self.records, self.etag = [], None

    def request(self, kind, start=None, end=None):
        remaining = self.reg['HTTP_range_limits']['maximum_run_seconds']-(time.monotonic()-self.started)
        if remaining <= 0:
            raise TimeoutError('registered run deadline exceeded')
        ordinal = len(self.records)
        header, body = self.root/f'{ordinal:03d}-headers.txt', self.root/f'{ordinal:03d}-body.bin'
        record = {'kind': kind, 'URL_host': urlsplit(self.url).hostname, 'URL_path': urlsplit(self.url).path,
                  'start': start, 'end': end, 'started_at_utc': datetime.now(UTC).isoformat(), 'status': 'failed'}
        self.records.append(record)
        timeout = min(remaining, self.reg['HTTP_range_limits']['timeout_seconds_per_request'])
        args = ['curl', '--disable', '--silent', '--show-error', '--proto', '=https', '--retry', '0', '--max-redirs', '0',
                '--max-time', str(timeout), '--header', 'Accept-Encoding: identity', '--dump-header', str(header),
                '--output', str(body), '--write-out', '%{http_code}']
        if kind == 'HEAD':
            args.append('--head')
        else:
            args.extend(['--max-filesize', str(end-start+1), '--header', f'Range: bytes={start}-{end}',
                         '--header', 'If-Match: '+self.etag])
        try:
            result = subprocess.run([*args, self.url], capture_output=True, text=True, timeout=timeout+1, check=False)
            record.update(client_exit_code=result.returncode, client_error=result.stderr[:4096])
            if result.returncode != 0:
                raise ValueError('single bounded HTTP request failed')
            status, fields = parse_headers(header)
            record.update(http_status=status, header_fields=fields)
            if fields.get('content-encoding', ['identity']) != ['identity']:
                raise ValueError('HTTP content encoding must be identity')
            if kind == 'RANGE' and (status != 206 or fields.get('content-range') != [f'bytes {start}-{end}/{self.asset["size"]}']
                or fields.get('etag') != [self.etag] or body.stat().st_size != end-start+1):
                raise ValueError('immutable exact range response required')
            record['status'] = 'response-retained'
            return status, fields, body
        finally:
            for path, key in ((header, 'headers'), (body, 'body')):
                if path.exists():
                    record[key+'_path'] = str(path)
                    record[key+'_sha256'] = sha(path)
                    record[key+'_bytes'] = path.stat().st_size
            record['finished_at_utc'] = datetime.now(UTC).isoformat()

    def head(self):
        for attempt in range(self.reg['URL_resolution']['maximum_HEAD_requests']):
            status, fields, _ = self.request('HEAD')
            if status in (301, 302, 307, 308):
                location = fields.get('location', [])
                if attempt >= self.reg['URL_resolution']['maximum_HEAD_redirects'] or len(location) != 1:
                    raise ValueError('HEAD redirect bound or ambiguity')
                target = urljoin(self.url, location[0])
                if not permitted_url(target, self.reg['URL_resolution']) or urlsplit(target).path != urlsplit(self.url).path:
                    raise ValueError('unapproved HEAD redirect')
                self.url = target
                continue
            etags = fields.get('etag', [])
            if (status != 200 or fields.get('content-length') != [str(self.asset['size'])]
                or len(etags) != 1 or not re.fullmatch(r'"[^"\x00-\x1f]+"', etags[0])):
                raise ValueError('HEAD size and strong ETag required')
            self.etag = etags[0]
            return
        raise ValueError('no approved immutable object HEAD')

    def fetch(self, start, end):
        _, _, body = self.request('RANGE', start, end)
        return body.read_bytes()

    def check_deadline(self):
        if time.monotonic()-self.started > self.reg['HTTP_range_limits']['maximum_run_seconds']:
            raise TimeoutError('registered run deadline exceeded')


def main():
    if sha(REG) != REG_HASH:
        raise ValueError('sealed pilot registration changed')
    reg = yaml.load(REG.read_text(), Loader=yaml.CSafeLoader)
    pins = {REG: REG_HASH, Path(__file__): sha(__file__), Path('scripts/patchseq_bounded_NWB.py'): sha('scripts/patchseq_bounded_NWB.py')}
    for key in ('catalog', 'archive_evidence', 'independent_evidence', 'acquisition_manifest'):
        pins[Path(reg[key])] = reg[key+'_sha256']
    if any(sha(p) != h for p, h in pins.items()):
        raise ValueError('pinned pilot prerequisites changed')
    if version('h5py') != reg['parser_dependency'].split('==')[1]:
        raise ValueError('registered parser dependency differs')
    def load(key):
        return yaml.load(Path(reg[key]).read_text(), Loader=yaml.CSafeLoader)
    asset = min(load('catalog')['assets'], key=lambda a: (a['path'], a['asset_id']))
    manifest = load('acquisition_manifest')
    matches = [r for r in manifest['records'] if r['asset_id'] == asset['asset_id']]
    if len(matches) != 1 or sha(matches[0]['receipt']) != matches[0]['receipt_sha256']:
        raise ValueError('one unchanged pilot receipt required')
    receipt_path = Path(matches[0]['receipt'])
    receipt = yaml.load(receipt_path.read_text(), Loader=yaml.CSafeLoader)
    response_path = receipt_path.parent/'response.json'
    pins[receipt_path], pins[response_path] = matches[0]['receipt_sha256'], receipt['response_sha256']
    if sha(response_path) != receipt['response_sha256']:
        raise ValueError('verified raw metadata changed')
    root = Path(reg['output_directory'])
    root.mkdir()
    report = {'status': 'NWB-metadata-pilot-failure-retained', 'registration_sha256': REG_HASH,
              'source_hashes': {str(p): h for p, h in pins.items()}, 'asset': asset,
              'started_at_utc': datetime.now(UTC).isoformat(), 'physiological_arrays_read': False,
              'complete_file_digest_independently_verified': False, 'specimen_eligibility_established': False}
    transport, stream = None, None
    try:
        raw = json.loads(response_path.read_bytes())
        report['URL_candidates'] = url_candidates(raw, reg['URL_resolution'])
        url, candidates = select_url(raw, reg['URL_resolution'])
        report['URL_candidates'] = candidates
        transport = Transport(root, url, asset, reg)
        transport.head()
        stream = BoundedReader(asset['size'], reg['HTTP_range_limits'], transport.fetch, transport.check_deadline)
        if stream.read(8) != b'\x89HDF\r\n\x1a\n':
            raise ValueError('HDF5 signature required')
        stream.seek(0)
        os.environ['HDF5_PLUGIN_PRELOAD'] = reg['logical_metadata_reads']['HDF5_plugin_preload']
        import h5py
        report['inspection'] = {'root_attributes': {}, 'datasets': {}, 'logical_dataset_reads': []}
        with h5py.File(stream, 'r') as root_file:
            inspect_metadata(root_file, reg['logical_metadata_reads'], h5py, report['inspection'])
        transport.check_deadline()
        directory, filename = asset['path'].split('/')
        expected = {'/general/subject/subject_id': directory.removeprefix('sub-'),
                    '/session_id': filename.removesuffix('_icephys.nwb').split('_ses-')[1]}
        report['literal_path_identity_diagnostics'] = {}
        for path, token in expected.items():
            entry = report['inspection']['datasets'][path]
            scalar = entry.get('scalar', {})
            observed = scalar.get('decoded_text') if scalar.get('raw_type') == 'bytes' else scalar.get('value') if scalar.get('raw_type') == 'str' else None
            report['literal_path_identity_diagnostics'][path] = observed == token if observed is not None else None
        report['status'] = 'bounded-NWB-metadata-inspected-independent-replay-pending'
    except (OSError, ValueError, TypeError, KeyError, TimeoutError, subprocess.TimeoutExpired) as exc:
        report.update(error_type=type(exc).__name__, error=str(exc))
    finally:
        if transport is not None:
            report['HTTP_receipts'] = transport.records
        if stream is not None:
            received = sum(r.get('body_bytes', 0) for r in transport.records if r['kind'] == 'RANGE')
            report.update(unique_downloaded_range_bytes=received, budget_charged_range_bytes=stream.downloaded,
                          unique_cached_range_bytes=sum(len(v) for v in stream.cache.values()), range_requests=stream.requests)
            stream.close()
    report['all_pinned_sources_unchanged'] = all(sha(p) == h for p, h in pins.items())
    if not report['all_pinned_sources_unchanged']:
        report.update(status='NWB-metadata-pilot-failure-retained', error='pinned source changed during pilot')
    report['finished_at_utc'] = datetime.now(UTC).isoformat()
    with (root/'manifest.yaml').open('x') as handle:
        yaml.safe_dump(report, handle, sort_keys=False)
    print(report['status'])
    print({k: report[k] for k in ('unique_downloaded_range_bytes', 'range_requests', 'error') if k in report})
    raise SystemExit(0 if report['status'] == 'bounded-NWB-metadata-inspected-independent-replay-pending' else 1)


if __name__ == '__main__':
    main()
