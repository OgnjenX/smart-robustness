"""Offline reconstruction of raw annotations and joins using relational queries."""

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

FIELDS = ('project', 'cell_specimen_id', 'donor_id', 'ephys_session_id',
          'transcriptomics_sample_id', 'structure', 'full_genotype',
          'corresponding_AIT2.3.1_ID', 'corresponding_AIT2.3.1_alias')
IDS = ('cell_specimen_id', 'donor_id', 'ephys_session_id')
REG = Path('docs/validation-results/post2008-patchseq-identity-join-registration-1175.yaml')
RESULT = Path('results/patchseq-identity-annotations-1176/manifest.yaml')
OUTPUT = Path('results/patchseq-identity-independent-1179')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(65536), b''):
            h.update(block)
    return h.hexdigest()


def decimal(raw):
    if (type(raw) is not str or not raw or raw[0] == '0'
        or any(c not in '0123456789' for c in raw)):
        return None
    return int(raw)


def path_tokens(path):
    parts = path.split('/')
    if len(parts) != 2 or not parts[0].startswith('sub-'):
        return None, None
    subject = parts[0][4:]
    prefix = parts[0] + '_ses-'
    if not parts[1].startswith(prefix) or not parts[1].endswith('_icephys.nwb'):
        return None, None
    session = parts[1][len(prefix):-len('_icephys.nwb')]
    if decimal(subject) is None or decimal(session) is None:
        return None, None
    return decimal(subject), decimal(session)


def raw_rows(path, schema):
    limit = csv.field_size_limit(4096)
    try:
        with tarfile.open(path, 'r:') as archive:
            candidates = [m for m in archive if m.name == schema['CSV_member']]
            if len(candidates) != 1 or not candidates[0].isfile() or candidates[0].size > 2097152:
                raise ValueError('bounded verified CSV member required')
            data = archive.extractfile(candidates[0]).read()
        reader = csv.DictReader(io.StringIO(data.decode('utf-8-sig'), newline=''), strict=True)
        if reader.fieldnames != schema['columns'] or not set(FIELDS) <= set(reader.fieldnames):
            raise ValueError('source header changed')
        output = []
        for index, row in enumerate(reader, 2):
            if len(output) >= 10000 or None in row or any(v is None for v in row.values()):
                raise ValueError('source row shape or bound differs')
            raw = {field: row[field] for field in FIELDS}
            output.append({'source_row_number': index, 'raw_fields': raw,
                           'canonical_ids': {field: decimal(raw[field]) for field in IDS}})
        return output
    finally:
        csv.field_size_limit(limit)


def rebuild(assets, rows, old):
    with sqlite3.connect(':memory:') as db:
        db.execute('CREATE TABLE annotations (ordinal INTEGER, cell TEXT, donor TEXT, session TEXT)')
        db.execute('CREATE TABLE catalog (ordinal INTEGER, subject TEXT, session TEXT)')
        for index, row in enumerate(rows):
            db.execute('INSERT INTO annotations VALUES (?, ?, ?, ?)',
                       (index, *(None if row['canonical_ids'][key] is None else str(row['canonical_ids'][key]) for key in IDS)))
        tokens = [path_tokens(asset['path']) for asset in assets]
        for index, (subject, session) in enumerate(tokens):
            db.execute('INSERT INTO catalog VALUES (?, ?, ?)',
                       (index, None if subject is None else str(subject), None if session is None else str(session)))
        outputs, referenced = [], set()
        for index, asset in enumerate(assets):
            subject, session = tokens[index]
            parents = [rows[i] for (i,) in db.execute(
                'SELECT a.ordinal FROM annotations a JOIN catalog c ON a.session=c.session WHERE c.ordinal=? ORDER BY a.ordinal', (index,))]
            referenced.update(row['source_row_number'] for row in parents)
            ids = parents[0]['canonical_ids'] if len(parents) == 1 else dict.fromkeys(IDS)
            cell, donor = ids['cell_specimen_id'], ids['donor_id']
            cell_count = db.execute('SELECT COUNT(*) FROM annotations WHERE cell=?', (str(cell),)).fetchone()[0]
            session_count = db.execute('SELECT COUNT(*) FROM annotations WHERE session=?', (str(session),)).fetchone()[0]
            catalog_count = db.execute('SELECT COUNT(*) FROM catalog WHERE session=?', (str(session),)).fetchone()[0]
            flags = {'canonical_path_shape': subject is not None, 'one_annotation_candidate': len(parents) == 1,
                     'canonical_specimen_ID': cell is not None, 'canonical_donor_ID': donor is not None,
                     'unique_source_specimen': cell is not None and cell_count == 1,
                     'unique_source_session': session is not None and session_count == 1,
                     'unique_catalog_session': session is not None and catalog_count == 1,
                     'donor_matches_path_subject': subject is not None and donor == subject}
            outputs.append({'asset_id': asset['asset_id'], 'path': asset['path'],
                            'path_subject_token': subject, 'path_session_token': session,
                            'annotation_candidates': parents, 'identity_flags': flags,
                            'path_annotation_consistent': all(flags.values()),
                            'internal_recording_identity_confirmed': False, 'molecular_class_qualified': False,
                            'measured_layer_qualified': False, 'biological_eligibility_established': False})
        duplicates = {}
        for key, table, column in [('duplicate_source_specimen_IDs', 'annotations', 'cell'),
                                   ('duplicate_source_session_IDs', 'annotations', 'session'),
                                   ('duplicate_catalog_session_tokens', 'catalog', 'session')]:
            duplicates[key] = sorted(int(value) for (value,) in db.execute(
                f'SELECT {column} FROM {table} WHERE {column} IS NOT NULL GROUP BY {column} HAVING COUNT(*)>1'))
        cells = {int(value) for (value,) in db.execute('SELECT DISTINCT cell FROM annotations WHERE cell IS NOT NULL')}
    return {'assets': outputs, 'annotation_rows': rows,
            'unreferenced_source_row_numbers': [r['source_row_number'] for r in rows if r['source_row_number'] not in referenced],
            **duplicates,
            'prior_specimen_ID_equalities': sorted(cells & {r['specimen_id'] for r in old}),
            'prior_included_specimen_ID_equalities': sorted(cells & {r['specimen_id'] for r in old if r['included'] is True}),
            'summary': {'assets': len(assets), 'annotation_rows': len(rows),
                        'path_annotation_consistent_assets': sum(r['path_annotation_consistent'] for r in outputs),
                        'assets_without_annotation_candidates': sum(not r['annotation_candidates'] for r in outputs),
                        'assets_with_multiple_annotation_candidates': sum(len(r['annotation_candidates']) > 1 for r in outputs),
                        'biological_eligibility_established': False}}


def compare(expected, result):
    for key, value in expected.items():
        if json.dumps(value, sort_keys=True) != json.dumps(result.get(key), sort_keys=True):
            raise ValueError(f'independent reconstruction differs: {key}')


def main():
    reg = yaml.safe_load(REG.read_text())
    pins = {REG: '4eed2c338f08dedc397b9536c923b2b20ac428bc89bafa962779651290479caf',
            RESULT: 'ed51183655f1ecca7a2fe0d6c1c3e00ae8657862a9d67e5ef585bb3942fbd8b0',
            Path('scripts/join_patchseq_identity.py'): 'd339e934564cba34f2c4a3fb3ed46dc01ccf177eb8e81f868791129d55736cd2'}
    for key in ('schema_assessment', 'annotation_archive', 'schema_verification', 'catalog_verification', 'previous_specimen_inventory'):
        pins[Path(reg[key])] = reg[key + '_sha256']
    if any(digest(path) != value for path, value in pins.items()):
        raise ValueError('sealed source or result hash differs')
    OUTPUT.mkdir()
    report = {'status': 'identity-verification-failure-retained', 'verifier_sha256': digest(__file__),
              'collector_imported': False, 'join_engine': 'independent-in-memory-SQLite',
              'network_requests': 0, 'physiological_arrays_read': False, 'parameter_fitting_performed': False,
              'started_at_utc': datetime.now(UTC).isoformat(),
              'source_hashes': {str(path): value for path, value in pins.items()}}
    try:
        load = lambda path: yaml.load(Path(path).read_text(), Loader=yaml.CSafeLoader)
        result, schema, catalog = load(RESULT), load(reg['schema_verification']), load(reg['catalog_verification'])
        if (result['registration_sha256'] != pins[REG] or result['collector_sha256'] != pins[Path('scripts/join_patchseq_identity.py')]
            or result['status'] != 'identity-annotations-joined-independent-verification-pending'
            or result['source_version'] != '20200625-June-annotation'
            or result['physiological_arrays_read'] is not False or result['parameter_fitting_performed'] is not False
            or schema['status'] != 'annotation-archive-and-header-independently-verified'
            or catalog['status'] != 'patchseq-catalog-independently-verified'):
            raise ValueError('terminal source identity or scope differs')
        rows = raw_rows(Path(reg['annotation_archive']), schema['schema'])
        expected = rebuild(catalog['assets'], rows, load(reg['previous_specimen_inventory'])['records'])
        compare(expected, result)
        if any(digest(path) != value for path, value in pins.items()):
            raise ValueError('source changed during independent verification')
        report.update(summary=expected['summary'], checked_assets=len(expected['assets']), checked_annotation_rows=len(rows),
                      literal_donor_equality_passes=sum(r['identity_flags']['donor_matches_path_subject'] for r in expected['assets']),
                      biological_eligibility_established=False,
                      status='identity-annotations-independently-verified')
    except (OSError, ValueError, KeyError, TypeError, csv.Error, tarfile.TarError) as exc:
        report['error'] = str(exc)
    report['finished_at_utc'] = datetime.now(UTC).isoformat()
    with (OUTPUT / 'manifest.yaml').open('x') as handle:
        yaml.safe_dump(report, handle, sort_keys=False)
    print(report['status'])
    raise SystemExit(0 if report['status'] == 'identity-annotations-independently-verified' else 1)


if __name__ == '__main__':
    main()
