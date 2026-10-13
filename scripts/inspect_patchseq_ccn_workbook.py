"""Read only a pinned nested reference workbook's structure and first rows."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import xml.etree.ElementTree as ET
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import yaml

REG = Path('docs/validation-results/post2008-patchseq-CCN-workbook-schema-registration-1211.yaml')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_member(path, name, expected_bytes):
    with zipfile.ZipFile(path) as archive:
        matches = [m for m in archive.infolist() if m.filename == name]
        if (len(matches) != 1 or matches[0].is_dir() or matches[0].file_size != expected_bytes
            or matches[0].file_size > 1048576 or matches[0].flag_bits & 1
            or (matches[0].external_attr >> 16) & 0o170000 == 0o120000):
            raise ValueError('exact bounded regular workbook member required')
        data = archive.read(matches[0])
        if len(data) != expected_bytes:
            raise ValueError('actual member byte extent differs')
        return data


def main():
    if digest(REG) != '98e6d41f7b719f9ec16016dafcc2531272f319802b00c9451de7655d7b5aab6d':
        raise ValueError('sealed reference header registration changed')
    reg = yaml.safe_load(REG.read_text())
    archive, reader = Path(reg['archive']), Path(reg['header_reader'])
    if digest(archive) != reg['archive_sha256'] or digest(reader) != reg['header_reader_sha256']:
        raise ValueError('pinned reference archive or header reader changed')
    root = Path(reg['output_directory'])
    root.mkdir()
    report = {'status': 'CCN-workbook-schema-failure-retained', 'registration_sha256': digest(REG),
              'collector_sha256': digest(__file__), 'started_at_utc': datetime.now(UTC).isoformat(),
              'network_requests': 0, 'reference_data_rows_interpreted': False,
              'physiological_arrays_read': False, 'biological_eligibility_established': False}
    try:
        payload = source_member(archive, reg['member'], reg['expected_member_bytes'])
        with (root / 'reference.xlsx').open('xb') as handle:
            handle.write(payload)
        report.update(source_member_sha256=hashlib.sha256(payload).hexdigest(), source_member_bytes=len(payload))
        spec = importlib.util.spec_from_file_location('sealed_header_reader', reader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        report['schema'] = module.inspect(io.BytesIO(payload))
        if digest(archive) != reg['archive_sha256'] or digest(reader) != reg['header_reader_sha256']:
            raise ValueError('pinned source changed during header inspection')
        report['status'] = 'CCN-workbook-first-rows-inspected-independent-verification-pending'
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile, ET.ParseError) as exc:
        report['error'] = str(exc)
    report['finished_at_utc'] = datetime.now(UTC).isoformat()
    with (root / 'manifest.yaml').open('x') as handle:
        yaml.safe_dump(report, handle, sort_keys=False)
    print(report['status'])
    raise SystemExit(0 if report['status'] == 'CCN-workbook-first-rows-inspected-independent-verification-pending' else 1)


if __name__ == '__main__':
    main()
