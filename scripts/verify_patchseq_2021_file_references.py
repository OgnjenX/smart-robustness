"""Independent relational reconstruction of every sealed file-reference result."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import yaml

REG = Path('docs/validation-results/post2008-patchseq-2021-file-reference-registration-1233.yaml')
RESULT = Path('results/patchseq-2021-file-reference-1234/manifest.yaml')
OUTPUT = Path('results/patchseq-2021-file-reference-independent-1236/manifest.yaml')
FLAGS = ('literal_filename_equals_catalog_basename', 'filename_subject_agrees', 'file_size_agrees',
         'manifest_cell_agrees_with_unique_July_session', 'unique_July_cell_and_session_same_row',
         'cross_source_checksum_agreement')


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def decimal(raw):
    return int(raw) if raw and raw[0] in '123456789' and all(c in '0123456789' for c in raw) else None


def name_tokens(name):
    if not isinstance(name, str):
        return None
    if name.endswith('.nwb') and decimal(name[:-4]) is not None:
        return None, decimal(name[:-4]), 'legacy-session-only'
    if name.startswith('sub-') and name.endswith('_icephys.nwb'):
        parts = name[4:-12].split('_ses-')
        if len(parts) == 2 and all(decimal(p) is not None for p in parts):
            return decimal(parts[0]), decimal(parts[1]), 'subject-session'
    return None


def rebuild(rows, assets, july):
    def literal(row, key):
        cell = row['cells'][key]
        return cell['literal_text'] if cell else None

    with sqlite3.connect(':memory:') as db:
        db.execute('CREATE TABLE files (pos INTEGER, rowid_source INTEGER, session TEXT, subject TEXT, kind TEXT)')
        db.execute('CREATE TABLE annotations (pos INTEGER, rowid_source INTEGER, session TEXT, cell TEXT)')
        by_number = {r['source_row_number']: r for r in rows}
        if len(by_number) != len(rows) or len({a['asset_id'] for a in assets}) != len(assets):
            raise ValueError('duplicate source identifiers')
        if len({r['source_row_number'] for r in july}) != len(july):
            raise ValueError('duplicate annotation row')
        for pos, row in enumerate(rows):
            parsed = name_tokens(literal(row, 'file_name'))
            if literal(row, 'technique') == 'intracellular_electrophysiology' and parsed:
                subject, session, kind = parsed
                db.execute('INSERT INTO files VALUES (?, ?, ?, ?, ?)',
                           (pos, row['source_row_number'], str(session), None if subject is None else str(subject), kind))
        for pos, row in enumerate(july):
            ids = row['canonical_ids']
            db.execute('INSERT INTO annotations VALUES (?, ?, ?, ?)',
                       (pos, row['source_row_number'], None if ids['ephys_session_id'] is None else str(ids['ephys_session_id']),
                        None if ids['cell_specimen_id'] is None else str(ids['cell_specimen_id'])))
        outputs, used = [], set()
        for asset in assets:
            parts = asset['path'].split('/')
            token = name_tokens(parts[-1])
            if len(parts) != 2 or token is None or token[2] != 'subject-session' or parts[0] != f'sub-{token[0]}':
                raise ValueError('invalid catalog path')
            if type(asset['size']) is not int or asset['size'] <= 0:
                raise ValueError('invalid catalog size')
            subject, session, _ = token
            parents = list(db.execute('SELECT rowid_source, cell FROM annotations WHERE session=? ORDER BY pos', (str(session),)))
            candidates = []
            for number, file_subject, kind in db.execute('SELECT rowid_source, subject, kind FROM files WHERE session=? ORDER BY pos', (str(session),)):
                used.add(number)
                row = by_number[number]
                cell_id = row['canonical_integers']['cell_specimen_id']
                size = row['canonical_integers']['file_size']
                cells = [r[0] for r in db.execute('SELECT rowid_source FROM annotations WHERE cell=? ORDER BY pos',
                                                 (None if cell_id is None else str(cell_id),))]
                same = len(cells) == len(parents) == 1 and cells[0] == parents[0][0]
                candidates.append({'manifest_source_row_number': number, 'filename_kind': kind,
                    'literal_filename_equals_catalog_basename': literal(row, 'file_name') == parts[-1],
                    'filename_subject_agrees': int(file_subject) == subject if file_subject is not None else None,
                    'file_size_agrees': size == asset['size'] if size is not None else None,
                    'July_cell_source_row_numbers': cells,
                    'manifest_cell_agrees_with_unique_July_session': str(cell_id) == parents[0][1]
                        if cell_id is not None and len(parents) == 1 else None,
                    'unique_July_cell_and_session_same_row': same, 'cross_source_checksum_agreement': None})
            complete = len(candidates) == len(parents) == 1 and all(candidates[0][key] is True for key in FLAGS[:-1])
            outputs.append({'asset_id': asset['asset_id'], 'path': asset['path'],
                            'July_session_source_row_numbers': [r[0] for r in parents],
                            'manifest_candidates': candidates, 'complete_external_file_reference_agreement': complete,
                            'internal_recording_identity_confirmed': False, 'specimen_eligibility_established': False})
        duplicate = lambda table, field: sorted(int(r[0]) for r in db.execute(
            f'SELECT {field} FROM {table} WHERE {field} IS NOT NULL GROUP BY {field} HAVING COUNT(*)>1'))
        comparisons = [c for a in outputs for c in a['manifest_candidates']]
        summary = {'source_rows': len(rows), 'catalog_assets': len(assets), 'July_rows': len(july),
                   'recognized_ephys_source_rows': db.execute('SELECT COUNT(*) FROM files').fetchone()[0],
                   'unmatched_assets': sum(not a['manifest_candidates'] for a in outputs),
                   'ambiguous_manifest_assets': sum(len(a['manifest_candidates']) > 1 for a in outputs),
                   'assets_without_July_session': sum(not a['July_session_source_row_numbers'] for a in outputs),
                   'complete_external_file_reference_agreements': sum(a['complete_external_file_reference_agreement'] for a in outputs),
                   'candidate_flag_counts': {key: {label: sum(c[key] is value for c in comparisons)
                                                   for label, value in [('true', True), ('false', False), ('unknown', None)]}
                                             for key in FLAGS}}
        return {'assets': outputs, 'unreferenced_manifest_source_row_numbers': [r['source_row_number'] for r in rows if r['source_row_number'] not in used],
                'duplicate_manifest_sessions': duplicate('files', 'session'),
                'duplicate_July_sessions': duplicate('annotations', 'session'),
                'duplicate_July_cells': duplicate('annotations', 'cell'), 'summary': summary}


def compare(expected, observed):
    for key, value in expected.items():
        if json.dumps(value, sort_keys=True) != json.dumps(observed.get(key), sort_keys=True):
            raise ValueError('independent reconstruction differs: '+key)


def main():
    pins = {REG: '90ca072aa980bf91804d991bc94fdaaa3d53b3dbea48ddf03bb7d8306673ae4d',
            RESULT: '5c701da3f8d443a9e2a5fe488ddf59b4a7ea648058c0922aa373b2104027bb78',
            Path('scripts/cross_reference_patchseq_2021_files.py'): '04241a7e72696a70c7868afe86f5b7c2df36ef3fd936910726652169c2d7cdae'}
    reg = yaml.load(REG.read_text(), Loader=yaml.CSafeLoader)
    pins.update({Path(v['path']): v['sha256'] for v in reg['sources'].values()})
    if any(digest(p) != h for p, h in pins.items()):
        raise ValueError('pinned evidence changed')
    sources = {k: yaml.load(Path(v['path']).read_text(), Loader=yaml.CSafeLoader) for k, v in reg['sources'].items()}
    observed = yaml.load(RESULT.read_text(), Loader=yaml.CSafeLoader)
    if (observed['status'] != '2021-file-reference-comparison-independent-verification-pending'
        or observed['registration_sha256'] != pins[REG]
        or observed['collector_sha256'] != pins[Path('scripts/cross_reference_patchseq_2021_files.py')]
        or observed['source_hashes'] != reg['sources'] or observed['network_requests'] != 0
        or any(observed[k] is not False for k in ('physiological_arrays_read', 'specimen_qualification_performed', 'frozen_baseline_modified'))):
        raise ValueError('terminal provenance or scope mismatch')
    expected = rebuild(sources['manifest']['rows'], sources['catalog']['assets'], sources['July']['July_rows'])
    compare(expected, observed)
    if any(digest(p) != h for p, h in pins.items()):
        raise ValueError('evidence changed during reconstruction')
    report = {'status': 'all-external-file-reference-comparisons-independently-reconstructed',
              'verifier_sha256': digest(__file__), 'verified_at_utc': datetime.now(UTC).isoformat(),
              'source_hashes': {str(p): h for p, h in pins.items()}, 'collector_imported': False,
              'summary': expected['summary'], 'unreferenced_source_rows': len(expected['unreferenced_manifest_source_row_numbers']),
              'network_requests': 0, 'physiological_arrays_read': False, 'specimen_qualification_performed': False}
    OUTPUT.parent.mkdir()
    with OUTPUT.open('x') as handle:
        yaml.safe_dump(report, handle, sort_keys=False)
    print(yaml.safe_dump(report, sort_keys=False))


if __name__ == '__main__':
    main()
