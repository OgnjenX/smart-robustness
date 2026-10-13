"""Conditional bounded acquisition of official July metadata; header only."""

from __future__ import annotations

import csv
import hashlib
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import yaml

REG = Path('docs/validation-results/post2008-patchseq-July-metadata-acquisition-registration-1194.yaml')


def validate(path, headers, reg):
    if path.stat().st_size != reg['expected_bytes']:
        raise ValueError('conditional object size differs')
    if headers.stat().st_size > 65536:
        raise ValueError('HTTP header bound exceeded')
    final = headers.read_text().replace('\r\n', '\n').strip().split('\n\n')[-1].splitlines()
    if not final or final[0].split()[1:2] != ['200']:
        raise ValueError('final HTTP status differs')
    fields = {}
    for line in final[1:]:
        key, value = line.split(':', 1)
        fields.setdefault(key.lower(), []).append(value.strip())
    if (fields.get('etag') != [reg['expected_etag']]
        or fields.get('x-amz-version-id') != [reg['expected_version_id']]
        or fields.get('content-length') != [str(reg['expected_bytes'])]
        or fields.get('content-encoding', ['identity']) != ['identity']):
        raise ValueError('conditional HTTP identity differs')
    with path.open('rb') as handle:
        header = handle.readline(65537)
    if len(header) > 65536 or not header.endswith(b'\n'):
        raise ValueError('bounded terminated CSV header required')
    columns = next(csv.reader([header.decode('utf-8-sig')], strict=True))
    if not 0 < len(columns) <= 256 or any(not c for c in columns) or len(set(columns)) != len(columns):
        raise ValueError('nonempty unique header columns required')
    return {'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'header_bytes': len(header), 'header_sha256': hashlib.sha256(header).hexdigest(), 'columns': columns}


def main():
    if hashlib.sha256(REG.read_bytes()).hexdigest() != '9fcf9bdf71505d80584da94d23fd581507c806ab39762462258d1f3bdc1e14b9':
        raise ValueError('sealed July acquisition registration changed')
    reg = yaml.safe_load(REG.read_text())
    root = Path(reg['output_directory'])
    root.mkdir()
    body, headers = root / 'metadata.csv', root / 'response-headers.txt'
    report = {'status': 'July-metadata-acquisition-failure-retained',
              'registration_sha256': hashlib.sha256(REG.read_bytes()).hexdigest(),
              'collector_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'started_at_utc': datetime.now(UTC).isoformat(), 'source_url': reg['source_url'],
              'row_values_interpreted': False, 'physiological_arrays_read': False}
    try:
        result = subprocess.run(['curl', '--fail', '--silent', '--show-error', '--proto', '=https',
                                 '--max-time', '120', '--max-filesize', '1090285',
                                 '--header', 'If-Match: ' + reg['expected_etag'],
                                 '--dump-header', str(headers), '--output', str(body),
                                 '--write-out', '%{http_code}', reg['source_url']],
                                capture_output=True, text=True, timeout=130, check=False)
        report.update(client_exit_code=result.returncode, http_status=result.stdout, client_error=result.stderr[:4096])
        if result.returncode != 0 or result.stdout != '200':
            raise ValueError('single conditional request failed')
        report['schema'] = validate(body, headers, reg)
        report['status'] = 'July-metadata-acquired-header-only-independent-verification-pending'
    except (OSError, ValueError, csv.Error, subprocess.TimeoutExpired) as exc:
        report['error'] = str(exc)
    report['finished_at_utc'] = datetime.now(UTC).isoformat()
    with (root / 'manifest.yaml').open('x') as handle:
        yaml.safe_dump(report, handle, sort_keys=False)
    print(report['status'])
    raise SystemExit(0 if report['status'] == 'July-metadata-acquired-header-only-independent-verification-pending' else 1)


if __name__ == '__main__':
    main()
