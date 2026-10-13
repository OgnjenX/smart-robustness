"""Immutable historical taxonomy archive acquisition; inspect directory only."""

from __future__ import annotations

import hashlib
import subprocess
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import yaml

REG = Path('docs/validation-results/post2008-patchseq-CCN-archive-acquisition-registration-1205.yaml')


def inspect(path, size, blob_id):
    if path.stat().st_size != size:
        raise ValueError('immutable reference size differs')
    git = hashlib.sha1(b'blob ' + str(size).encode() + b'\x00')
    sha = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(65536), b''):
            git.update(block)
            sha.update(block)
    if git.hexdigest() != blob_id:
        raise ValueError('official Git blob differs')
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        names = [m.filename for m in members]
        if (len(members) > 1000 or len(set(names)) != len(names)
            or sum(m.file_size for m in members) > 67108864
            or any(n.startswith('/') or '\\' in n or '..' in n.split('/') for n in names)
            or any(m.flag_bits & 1 or (m.external_attr >> 16) & 0o170000 == 0o120000 for m in members)):
            raise ValueError('bounded safe archive directory required')
        return {'bytes': size, 'git_blob_sha1': git.hexdigest(), 'sha256': sha.hexdigest(),
                'members': [{'name': m.filename, 'bytes': m.file_size, 'compressed_bytes': m.compress_size,
                             'compression': m.compress_type, 'directory': m.is_dir()} for m in members],
                'member_contents_interpreted': False, 'archive_extracted': False}


def main():
    if hashlib.sha256(REG.read_bytes()).hexdigest() != 'dd69708e4bf20b5fce45fc422ff718278cb3a7f313f6058482f8899e5fa19562':
        raise ValueError('sealed CCN acquisition registration changed')
    reg = yaml.safe_load(REG.read_text())
    root = Path(reg['output_directory'])
    root.mkdir()
    body = root / 'reference.zip'
    report = {'status': 'CCN-directory-acquisition-failure-retained',
              'registration_sha256': hashlib.sha256(REG.read_bytes()).hexdigest(),
              'collector_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'started_at_utc': datetime.now(UTC).isoformat(), 'source_url': reg['source_url'],
              'member_contents_interpreted': False, 'physiological_arrays_read': False}
    try:
        result = subprocess.run(['curl', '--fail', '--silent', '--show-error', '--proto', '=https',
                                 '--max-time', '120', '--max-filesize', '844341', '--dump-header', str(root / 'response-headers.txt'),
                                 '--output', str(body), '--write-out', '%{http_code}', reg['source_url']],
                                capture_output=True, text=True, timeout=130, check=False)
        report.update(client_exit_code=result.returncode, http_status=result.stdout, client_error=result.stderr[:4096])
        if result.returncode != 0 or result.stdout != '200':
            raise ValueError('single immutable request failed')
        report['directory'] = inspect(body, reg['expected_bytes'], reg['expected_git_blob_sha1'])
        report['status'] = 'CCN-archive-directory-acquired-independent-verification-pending'
    except (OSError, ValueError, zipfile.BadZipFile, subprocess.TimeoutExpired) as exc:
        report['error'] = str(exc)
    report['finished_at_utc'] = datetime.now(UTC).isoformat()
    with (root / 'manifest.yaml').open('x') as handle:
        yaml.safe_dump(report, handle, sort_keys=False)
    print(report['status'])
    raise SystemExit(0 if report['status'] == 'CCN-archive-directory-acquired-independent-verification-pending' else 1)


if __name__ == '__main__':
    main()
