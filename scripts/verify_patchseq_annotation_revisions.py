"""Independent original-CSV and SQL reconstruction of revision comparisons."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import sqlite3
import tarfile
from datetime import UTC, datetime
from pathlib import Path

import yaml

REG = Path('docs/validation-results/post2008-patchseq-July-June-comparison-registration-1197.yaml')
RESULT = Path('results/patchseq-July-June-comparison-1198/manifest.yaml')
ARCHIVE = Path('results/patchseq-annotation-archive-schema-1171/annotations.csv.tar')
OUTPUT = Path('results/patchseq-July-June-comparison-independent-1199')
COMMON = ('project', 'cell_specimen_id', 'donor_id', 'ephys_session_id',
          'transcriptomics_sample_id', 'structure', 'full_genotype')
PAIRS = tuple((k, k) for k in COMMON) + (('T-type Accession ID', 'corresponding_AIT2.3.1_ID'),
                                      ('T-type Label', 'corresponding_AIT2.3.1_alias'))
IDS = ('cell_specimen_id', 'donor_id', 'ephys_session_id')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(65536), b''):
            h.update(block)
    return h.hexdigest()


def decimal(value):
    if type(value) is not str or not value or value[0] == '0' or any(c not in '0123456789' for c in value):
        return None
    return int(value)


def project(data, fields):
    if len(data) > 2097152:
        raise ValueError('CSV byte bound exceeded')
    limit = csv.field_size_limit(4096)
    try:
        reader = csv.DictReader(io.StringIO(data.decode('utf-8-sig'), newline=''), strict=True)
        if not set(fields) <= set(reader.fieldnames) or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError('projection schema differs')
        rows = []
        for number, row in enumerate(reader, 2):
            if None in row or any(v is None for v in row.values()) or len(rows) >= 10000:
                raise ValueError('source row shape or extent differs')
            raw = {k: row[k] for k in fields}
            rows.append({'source_row_number': number, 'raw_fields': raw,
                         'canonical_ids': {k: decimal(raw[k]) for k in IDS}})
        return rows
    finally:
        csv.field_size_limit(limit)


def reconstruct(july, june):
    with sqlite3.connect(':memory:') as db:
        db.execute('CREATE TABLE old (ordinal INTEGER, cell TEXT, session TEXT)')
        db.execute('CREATE TABLE new (cell TEXT, donor TEXT, session TEXT)')
        for i, row in enumerate(june):
            db.execute('INSERT INTO old VALUES (?,?,?)', (i, *(None if row['canonical_ids'][k] is None else str(row['canonical_ids'][k]) for k in ('cell_specimen_id', 'ephys_session_id'))))
        for row in july:
            db.execute('INSERT INTO new VALUES (?,?,?)', tuple(None if row['canonical_ids'][k] is None else str(row['canonical_ids'][k]) for k in IDS))
        outputs, used = [], set()
        for row in july:
            cell = [june[i] for (i,) in db.execute('SELECT ordinal FROM old WHERE cell=? ORDER BY ordinal', (str(row['canonical_ids']['cell_specimen_id']),))]
            session = [june[i] for (i,) in db.execute('SELECT ordinal FROM old WHERE session=? ORDER BY ordinal', (str(row['canonical_ids']['ephys_session_id']),))]
            numbers = [r['source_row_number'] for r in cell]
            session_numbers = [r['source_row_number'] for r in session]
            used.update(numbers)
            outputs.append({'July_source_row_number': row['source_row_number'], 'June_cell_candidates': numbers,
                            'June_session_candidates': session_numbers,
                            'one_cell_and_session_correspondence': len(numbers) == len(session_numbers) == 1 and numbers == session_numbers,
                            'raw_field_differences': {a: row['raw_fields'][a] != cell[0]['raw_fields'][b] for a, b in PAIRS} if len(cell) == 1 else None,
                            'biological_eligibility_established': False})
        duplicates = {k: sorted(int(v) for (v,) in db.execute(f'SELECT {column} FROM new WHERE {column} IS NOT NULL GROUP BY {column} HAVING COUNT(*)>1'))
                      for k, column in zip(IDS, ('cell', 'donor', 'session'), strict=True)}
    labels = sorted({r['raw_fields']['MET-type Label'] for r in july})
    return {'July_rows': july, 'comparisons': outputs,
            'unreferenced_June_rows_by_cell': [r['source_row_number'] for r in june if r['source_row_number'] not in used],
            'July_duplicate_IDs': duplicates,
            'July_noncanonical_ID_counts': {k: sum(r['canonical_ids'][k] is None for r in july) for k in IDS},
            'raw_field_change_counts': {k: sum(o['raw_field_differences'] is not None and o['raw_field_differences'][k] for o in outputs) for k, _ in PAIRS},
            'raw_MET_label_counts': {label: sum(r['raw_fields']['MET-type Label'] == label for r in july) for label in labels},
            'summary': {'July_rows': len(july), 'June_rows': len(june),
                        'one_cell_and_session_correspondences': sum(o['one_cell_and_session_correspondence'] for o in outputs),
                        'July_rows_without_June_cell': sum(not o['June_cell_candidates'] for o in outputs),
                        'July_rows_with_ambiguous_June_cell': sum(len(o['June_cell_candidates']) > 1 for o in outputs),
                        'July_rows_without_June_session': sum(not o['June_session_candidates'] for o in outputs)}}


def compare(expected, observed):
    for key, value in expected.items():
        if json.dumps(value, sort_keys=True) != json.dumps(observed.get(key), sort_keys=True):
            raise ValueError('independent revision reconstruction differs: ' + key)


def main():
    pins = {REG: '527859a8bb2966e04f21c21409110af7207b5e7c7582b3ec05d021a1eac31520',
            RESULT: '4c31f9b2cf447b0c82388795417402563c979291b4a223c3fb6e2d109302bcb3',
            ARCHIVE: '6854981d0cdf24f837b8d2b8adf0243aaa53871f4f1cc19f89f56d7430d15c0a',
            Path('scripts/compare_patchseq_annotation_revisions.py'): '0d390967da3ce9355a44e025e2c5464ea2265cc26f98b1f6f25ab52215b26c03'}
    reg = yaml.safe_load(REG.read_text())
    pins.update({Path(reg[k]): reg[k + '_sha256'] for k in ('July_assessment', 'July_csv', 'July_schema', 'June_identity', 'June_verification')})
    if any(digest(p) != h for p, h in pins.items()):
        raise ValueError('pinned evidence changed')
    OUTPUT.mkdir()
    report = {'status': 'revision-verification-failure-retained', 'verifier_sha256': digest(__file__),
              'source_hashes': {str(p): h for p, h in pins.items()}, 'collector_imported': False,
              'network_requests': 0, 'physiological_arrays_read': False, 'parameter_fitting_performed': False,
              'biological_eligibility_established': False, 'started_at_utc': datetime.now(UTC).isoformat()}
    try:
        result = yaml.load(RESULT.read_text(), Loader=yaml.CSafeLoader)
        if (result['status'] != 'annotation-revisions-compared-independent-verification-pending'
            or result['registration_sha256'] != pins[REG] or result['collector_sha256'] != pins[Path('scripts/compare_patchseq_annotation_revisions.py')]
            or result['physiological_arrays_read'] is not False or result['parameter_fitting_performed'] is not False
            or result['biological_eligibility_established'] is not False):
            raise ValueError('terminal scope or source identity differs')
        with tarfile.open(ARCHIVE, 'r:') as archive:
            member = archive.getmember('20200625_patchseq_metadata_mouse/20200625_patchseq_metadata_mouse.csv')
            if not member.isfile() or member.size > 2097152:
                raise ValueError('exact bounded June CSV member required')
            june = project(archive.extractfile(member).read(), [b for _, b in PAIRS])
        july = project(Path(reg['July_csv']).read_bytes(), [a for a, _ in PAIRS] + ['MET-type Label'])
        expected = reconstruct(july, june)
        compare(expected, result)
        if any(digest(p) != h for p, h in pins.items()):
            raise ValueError('sources changed during verification')
        report.update(status='annotation-revisions-independently-verified', summary=expected['summary'],
                      raw_field_change_counts=expected['raw_field_change_counts'], raw_MET_label_counts=expected['raw_MET_label_counts'])
    except (OSError, ValueError, KeyError, TypeError, csv.Error, tarfile.TarError) as exc:
        report['error'] = str(exc)
    report['finished_at_utc'] = datetime.now(UTC).isoformat()
    with (OUTPUT / 'manifest.yaml').open('x') as handle:
        yaml.safe_dump(report, handle, sort_keys=False)
    print(report['status'])
    raise SystemExit(0 if report['status'] == 'annotation-revisions-independently-verified' else 1)


if __name__ == '__main__':
    main()
