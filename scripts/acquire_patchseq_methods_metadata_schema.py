"""Acquire pinned methods metadata, inspecting only its literal CSV header."""

from __future__ import annotations

import csv
import hashlib
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import yaml

REG = Path('docs/validation-results/post2008-patchseq-methods-mouse-metadata-schema-registration-1243.yaml')
REG_HASH = '38364e56bb514363ed7b6c9569ec2db4e65df0a43c86b871044cdff79e3db9a1'


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def inspect(body, headers, reg):
    size = body.stat().st_size
    if size != reg['expected_bytes'] or headers.stat().st_size > 65536:
        raise ValueError('source byte or header bound differs')
    final = headers.read_text().replace('\r\n', '\n').strip().split('\n\n')[-1].splitlines()
    if not final or final[0].split()[1:2] != ['200']:
        raise ValueError('HTTP 200 required')
    h = hashlib.sha1(f'blob {size}\0'.encode())
    with body.open('rb') as handle:
        for block in iter(lambda: handle.read(65536), b''):
            h.update(block)
    if h.hexdigest() != reg['expected_git_blob_sha1']:
        raise ValueError('pinned Git blob differs')
    with body.open('rb') as handle:
        raw_header = handle.readline(reg['maximum_header_bytes']+1)
    if len(raw_header) > reg['maximum_header_bytes'] or not raw_header.endswith(b'\n'):
        raise ValueError('bounded terminated header required')
    columns = next(csv.reader([raw_header.decode('utf-8-sig')], strict=True))
    if not 0 < len(columns) <= reg['maximum_columns'] or any(not c for c in columns) or len(set(columns)) != len(columns):
        raise ValueError('unique nonempty literal header required')
    return {'bytes': size, 'sha256': digest(body), 'git_blob_sha1': h.hexdigest(),
            'header_bytes': len(raw_header), 'header_sha256': hashlib.sha256(raw_header).hexdigest(),
            'columns': columns, 'data_rows_interpreted': False}


def main():
    if digest(REG) != REG_HASH:
        raise ValueError('sealed header acquisition registration changed')
    reg = yaml.load(REG.read_text(), Loader=yaml.CSafeLoader)
    root = Path(reg['output_directory'])
    root.mkdir()
    body, header = root/'metadata.csv', root/'headers.txt'
    report = {'status': 'methods-metadata-schema-acquisition-failure-retained',
              'registration_sha256': REG_HASH, 'collector_sha256': digest(__file__),
              'source_url': reg['source_url'], 'started_at_utc': datetime.now(UTC).isoformat(),
              'source_rows_interpreted': False, 'physiological_arrays_read': False,
              'cell_qualification_performed': False}
    try:
        result = subprocess.run(['curl', '--silent', '--show-error', '--proto', '=https',
            '--max-time', str(reg['timeout_seconds']), '--max-filesize', str(reg['expected_bytes']),
            '--header', 'Accept-Encoding: identity', '--dump-header', str(header), '--output', str(body),
            '--write-out', '%{http_code}', reg['source_url']],
            capture_output=True, text=True, timeout=reg['timeout_seconds']+5, check=False)
        report.update(client_exit_code=result.returncode, http_status=result.stdout, client_error=result.stderr[:4096])
        if result.returncode != 0 or result.stdout != '200':
            raise ValueError('single bounded request failed; no retries')
        report['schema'] = inspect(body, header, reg)
        if digest(REG) != REG_HASH:
            raise ValueError('registration changed during acquisition')
        report['status'] = 'methods-metadata-header-read-independent-verification-pending'
    except (OSError, ValueError, csv.Error, UnicodeError, subprocess.TimeoutExpired) as exc:
        report['error'] = str(exc)
    report['finished_at_utc'] = datetime.now(UTC).isoformat()
    for path, key in [(body, 'raw_sha256'), (header, 'headers_sha256')]:
        if path.exists():
            report[key] = digest(path)
    with (root/'manifest.yaml').open('x') as handle:
        yaml.safe_dump(report, handle, sort_keys=False)
    print(yaml.safe_dump(report, sort_keys=False))
    raise SystemExit(0 if report['status'] == 'methods-metadata-header-read-independent-verification-pending' else 1)


if __name__ == '__main__':
    main()
