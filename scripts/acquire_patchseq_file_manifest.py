"""One pinned manifest request; interpret only its bounded CSV header."""

from __future__ import annotations

import csv
import hashlib
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import yaml

REG = Path('docs/validation-results/post2008-patchseq-file-manifest-acquisition-registration-1182.yaml')
URL = 'https://raw.githubusercontent.com/AllenInstitute/ipfx/c607a36e25618a6bb8ba5438c1cd945fb429e659/ipfx/data_release/2020-06-26_mouse_file_manifest.csv'
ROOT = Path('results/patchseq-file-manifest-schema-1183')


def inspect_object(path, expected_size, expected_blob):
    if path.stat().st_size != expected_size:
        raise ValueError('official manifest size differs')
    blob = hashlib.sha1(b'blob ' + str(expected_size).encode() + b'\x00')
    sha = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(65536), b''):
            blob.update(block)
            sha.update(block)
    if blob.hexdigest() != expected_blob:
        raise ValueError('official Git object differs')
    with path.open('rb') as handle:
        header = handle.readline(65537)
    if not header.endswith(b'\n') or len(header) > 65536:
        raise ValueError('bounded terminated CSV header required')
    columns = next(csv.reader([header.decode('utf-8-sig')], strict=True))
    if not 0 < len(columns) <= 256 or any(not c for c in columns) or len(set(columns)) != len(columns):
        raise ValueError('unique nonempty header columns required')
    return {'bytes': expected_size, 'git_blob_sha1': blob.hexdigest(), 'sha256': sha.hexdigest(),
            'columns': columns, 'header_bytes': len(header), 'header_sha256': hashlib.sha256(header).hexdigest()}


def main():
    if hashlib.sha256(REG.read_bytes()).hexdigest() != '5f7225a217ea88d61223512a08fe599989c5f9d8c541375af4b559c7a25433ed':
        raise ValueError('sealed manifest acquisition registration changed')
    reg = yaml.safe_load(REG.read_text())
    if (reg['source_url'] != URL or reg['expected_bytes'] != 5783047
        or reg['expected_git_blob_sha1'] != '91c19126fce62de427707fb70829d9eb84cfd743'
        or reg['output_directory'] != str(ROOT) or reg['maximum_requests'] != 1
        or reg['maximum_seconds'] != 120 or reg['maximum_header_bytes'] != 65536
        or any(reg[k] is not False for k in ('redirects_authorized', 'retries_authorized',
            'annotation_row_values_interpreted', 'recording_download_authorized',
            'physiological_array_reads_authorized', 'cell_class_or_layer_qualification_authorized',
            'parameter_fitting_authorized', 'cell_or_network_execution_authorized'))):
        raise ValueError('acquisition scope changed')
    ROOT.mkdir()
    path = ROOT / 'file-manifest.csv'
    report = {'status': 'file-manifest-acquisition-failure-retained',
              'registration_sha256': hashlib.sha256(REG.read_bytes()).hexdigest(),
              'collector_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'started_at_utc': datetime.now(UTC).isoformat(), 'source_url': URL,
              'annotation_row_values_interpreted': False, 'physiological_arrays_read': False}
    try:
        completed = subprocess.run(['curl', '--fail', '--silent', '--show-error', '--proto', '=https',
                                    '--max-time', '120', '--max-filesize', '5783047',
                                    '--dump-header', str(ROOT / 'response-headers.txt'),
                                    '--output', str(path), '--write-out', '%{http_code}', URL],
                                   capture_output=True, text=True, timeout=130, check=False)
        report.update(client_exit_code=completed.returncode, http_status=completed.stdout,
                      client_error=completed.stderr[:4096])
        if completed.returncode != 0 or completed.stdout != '200':
            raise ValueError('single request did not yield HTTP 200')
        report['schema'] = inspect_object(path, reg['expected_bytes'], reg['expected_git_blob_sha1'])
        report['status'] = 'file-manifest-acquired-header-only-independent-verification-pending'
    except (OSError, ValueError, csv.Error, subprocess.TimeoutExpired) as exc:
        report['error'] = str(exc)
    report['finished_at_utc'] = datetime.now(UTC).isoformat()
    with (ROOT / 'manifest.yaml').open('x') as handle:
        yaml.safe_dump(report, handle, sort_keys=False)
    print(report['status'])
    raise SystemExit(0 if report['status'] == 'file-manifest-acquired-header-only-independent-verification-pending' else 1)


if __name__ == '__main__':
    main()
