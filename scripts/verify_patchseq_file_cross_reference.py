"""Independent raw CSV projection and relational cross-reference verification."""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import yaml

REG = Path('docs/validation-results/post2008-patchseq-file-cross-reference-registration-1185.yaml')
RESULT = Path('results/patchseq-file-cross-reference-1186/manifest.yaml')
OUTPUT = Path('results/patchseq-file-cross-reference-independent-1187')
FIELDS = ('file_id', 'file_name', 'file_size', 'sha256_checksum', 'md5_checksum', 'technique',
          'file_type', 'archive', 'archive_uri', 'project', 'cell_specimen_id')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(65536), b''):
            h.update(block)
    return h.hexdigest()


def decimal(value):
    return int(value) if type(value) is str and value and value[0] != '0' and all(c in '0123456789' for c in value) else None


def session(name):
    if name.endswith('.nwb') and decimal(name[:-4]) is not None:
        return decimal(name[:-4])
    if name.startswith('sub-') and name.endswith('_icephys.nwb'):
        parts = name[4:-len('_icephys.nwb')].split('_ses-')
        if len(parts) == 2 and all(decimal(p) is not None for p in parts):
            return decimal(parts[1])
    return None


def project(path):
    if path.stat().st_size > 6000000:
        raise ValueError('manifest byte bound exceeded')
    limit = csv.field_size_limit(4096)
    try:
        with path.open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle, strict=True)
            if reader.fieldnames != list(FIELDS):
                raise ValueError('manifest header changed')
            rows = []
            for number, raw in enumerate(reader, 2):
                if None in raw or any(v is None for v in raw.values()) or len(rows) >= 50000:
                    raise ValueError('source row shape or count changed')
                rows.append({'source_row_number': number, 'raw_fields': raw,
                             'canonical_cell_specimen_id': decimal(raw['cell_specimen_id']),
                             'recognized_file_session': session(raw['file_name'])})
            return rows
    finally:
        csv.field_size_limit(limit)


def reconstruct(rows, assets):
    with sqlite3.connect(':memory:') as db:
        db.execute('CREATE TABLE files (row INTEGER, technique TEXT, session TEXT)')
        for r in rows:
            db.execute('INSERT INTO files VALUES (?, ?, ?)', (r['source_row_number'], r['raw_fields']['technique'],
                       None if r['recognized_file_session'] is None else str(r['recognized_file_session'])))
        by_row = {r['source_row_number']: r for r in rows}
        outputs, used = [], set()
        for asset in assets:
            numbers = [n for (n,) in db.execute('SELECT row FROM files WHERE technique=? AND session=? ORDER BY row',
                       ('intracellular_electrophysiology', str(asset['path_session_token'])))]
            used.update(numbers)
            parents = asset['annotation_candidates']
            unique = len(numbers) == len(parents) == 1
            cell = by_row[numbers[0]]['canonical_cell_specimen_id'] if unique else None
            outputs.append({'asset_id': asset['asset_id'], 'path': asset['path'],
                            'manifest_source_row_numbers': numbers,
                            'one_manifest_and_annotation_candidate': unique,
                            'source_cell_ID_agrees': unique and cell is not None and cell == parents[0]['canonical_ids']['cell_specimen_id'],
                            'internal_recording_identity_confirmed': False, 'biological_eligibility_established': False})
        duplicates = sorted(int(s) for (s,) in db.execute('SELECT session FROM files WHERE technique=? AND session IS NOT NULL GROUP BY session HAVING COUNT(*)>1',
                            ('intracellular_electrophysiology',)))
        recognized = db.execute('SELECT COUNT(*) FROM files WHERE technique=? AND session IS NOT NULL',
                                ('intracellular_electrophysiology',)).fetchone()[0]
    counts = {}
    for key in ('technique', 'archive', 'file_type'):
        values = sorted({r['raw_fields'][key] for r in rows})
        counts[key] = {v: sum(r['raw_fields'][key] == v for r in rows) for v in values}
    return {'source_rows': rows, 'assets': outputs,
            'unreferenced_source_row_numbers': [r['source_row_number'] for r in rows if r['source_row_number'] not in used],
            'duplicate_source_file_sessions': duplicates, 'raw_value_counts': counts,
            'missing_field_counts': {key: sum(not r['raw_fields'][key] for r in rows) for key in FIELDS},
            'summary': {'source_rows': len(rows), 'assets': len(assets), 'recognized_ephys_file_rows': recognized,
                        'agreeing_cell_references': sum(o['source_cell_ID_agrees'] for o in outputs),
                        'unmatched_assets': sum(not o['manifest_source_row_numbers'] for o in outputs),
                        'ambiguous_assets': sum(len(o['manifest_source_row_numbers']) > 1 for o in outputs),
                        'unique_candidate_cell_conflicts': sum(o['one_manifest_and_annotation_candidate'] and not o['source_cell_ID_agrees'] for o in outputs)}}


def compare(expected, result):
    for key, value in expected.items():
        if json.dumps(value, sort_keys=True) != json.dumps(result.get(key), sort_keys=True):
            raise ValueError('independent reconstruction differs: ' + key)


def main():
    pins = {REG: '3a4e1000ba72d893f4b01951f9631dec6156ff35497a413245313175badfc659',
            RESULT: '0836ab32c4fe94ef1428f8d8a7249a35d8ab8f291014d665af08aa5938599af5',
            Path('scripts/cross_reference_patchseq_files.py'): 'ff6bfd94c8f4216af02a4a43e505dd16b95db4dad04657b245728e8086099c0e'}
    reg = yaml.safe_load(REG.read_text())
    pins.update({Path(reg[key]): reg[key + '_sha256'] for key in ('raw_manifest', 'identity_result', 'identity_verification')})
    if any(digest(p) != h for p, h in pins.items()):
        raise ValueError('pinned evidence changed')
    OUTPUT.mkdir()
    report = {'status': 'file-cross-reference-verification-failure-retained', 'verifier_sha256': digest(__file__),
              'source_hashes': {str(p): h for p, h in pins.items()}, 'collector_imported': False,
              'network_requests': 0, 'physiological_arrays_read': False, 'parameter_fitting_performed': False,
              'biological_eligibility_established': False, 'started_at_utc': datetime.now(UTC).isoformat()}
    try:
        result = yaml.load(RESULT.read_text(), Loader=yaml.CSafeLoader)
        identity = yaml.load(Path(reg['identity_result']).read_text(), Loader=yaml.CSafeLoader)
        if (result['status'] != 'file-cross-references-reconstructed-independent-verification-pending'
            or result['registration_sha256'] != pins[REG] or result['collector_sha256'] != pins[Path('scripts/cross_reference_patchseq_files.py')]
            or result['network_requests'] != 0 or result['physiological_arrays_read'] is not False
            or result['parameter_fitting_performed'] is not False or result['biological_eligibility_established'] is not False):
            raise ValueError('terminal scope or source identity differs')
        expected = reconstruct(project(Path(reg['raw_manifest'])), identity['assets'])
        compare(expected, result)
        if any(digest(p) != h for p, h in pins.items()):
            raise ValueError('source changed during verification')
        report.update(status='file-cross-references-independently-verified', summary=expected['summary'])
    except (OSError, ValueError, TypeError, KeyError, csv.Error) as exc:
        report['error'] = str(exc)
    report['finished_at_utc'] = datetime.now(UTC).isoformat()
    with (OUTPUT / 'manifest.yaml').open('x') as handle:
        yaml.safe_dump(report, handle, sort_keys=False)
    print(report['status'])
    raise SystemExit(0 if report['status'] == 'file-cross-references-independently-verified' else 1)


if __name__ == '__main__':
    main()
