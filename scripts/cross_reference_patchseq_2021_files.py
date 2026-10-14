"""Sealed file references with all candidates and unresolved identity preserved."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import yaml

REG = Path('docs/validation-results/post2008-patchseq-2021-file-reference-registration-1233.yaml')
REG_SHA = '90ca072aa980bf91804d991bc94fdaaa3d53b3dbea48ddf03bb7d8306673ae4d'
DEC = r'[1-9][0-9]*'


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def literal(row, field):
    cell = row['cells'][field]
    return None if cell is None else cell['literal_text']


def filename_tokens(name):
    if type(name) is not str:
        return None
    match = re.fullmatch(f'({DEC})\\.nwb', name)
    if match:
        return {'subject': None, 'session': int(match[1]), 'kind': 'legacy-session-only'}
    match = re.fullmatch(f'sub-({DEC})_ses-({DEC})_icephys\\.nwb', name)
    return {'subject': int(match[1]), 'session': int(match[2]), 'kind': 'subject-session'} if match else None


def path_tokens(path):
    match = re.fullmatch(f'sub-({DEC})/sub-({DEC})_ses-({DEC})_icephys\\.nwb', path)
    if match is None or match[1] != match[2]:
        raise ValueError('exact consistent catalog subject/session path required')
    return int(match[1]), int(match[3])


def compare(rows, assets, july):
    if len(rows) > 50000 or len(assets) > 10000 or len(july) > 10000:
        raise ValueError('source count exceeds registered bound')
    for records, key in ((rows, 'source_row_number'), (assets, 'asset_id'), (july, 'source_row_number')):
        if len({r[key] for r in records}) != len(records):
            raise ValueError('duplicate source row or catalog asset identity')
    files, sessions, cells = defaultdict(list), defaultdict(list), defaultdict(list)
    row_tokens = {}
    for row in rows:
        tokens = filename_tokens(literal(row, 'file_name'))
        if literal(row, 'technique') == 'intracellular_electrophysiology' and tokens:
            files[tokens['session']].append(row)
            row_tokens[row['source_row_number']] = tokens
    for row in july:
        for key, table in (('ephys_session_id', sessions), ('cell_specimen_id', cells)):
            value = row['canonical_ids'][key]
            if value is not None:
                table[value].append(row)
    outputs, used = [], set()
    for asset in assets:
        subject, session = path_tokens(asset['path'])
        if type(asset['size']) is not int or asset['size'] <= 0:
            raise ValueError('positive exact catalog size required')
        parent = sessions.get(session, [])
        candidates = []
        for row in files.get(session, []):
            number = row['source_row_number']
            used.add(number)
            token = row_tokens[number]
            ids = row['canonical_integers']
            cell_id, size = ids['cell_specimen_id'], ids['file_size']
            by_cell = cells.get(cell_id, []) if cell_id is not None else []
            unique = len(parent) == len(by_cell) == 1
            same = unique and parent[0]['source_row_number'] == by_cell[0]['source_row_number']
            candidates.append({'manifest_source_row_number': number, 'filename_kind': token['kind'],
                'literal_filename_equals_catalog_basename': literal(row, 'file_name') == asset['path'].split('/')[-1],
                'filename_subject_agrees': token['subject'] == subject if token['subject'] is not None else None,
                'file_size_agrees': size == asset['size'] if size is not None else None,
                'July_cell_source_row_numbers': [r['source_row_number'] for r in by_cell],
                'manifest_cell_agrees_with_unique_July_session': cell_id == parent[0]['canonical_ids']['cell_specimen_id']
                    if len(parent) == 1 and cell_id is not None else None,
                'unique_July_cell_and_session_same_row': same,
                'cross_source_checksum_agreement': None})
        unique_reference = len(candidates) == len(parent) == 1
        confirmed = unique_reference and all(candidates[0][key] is True for key in (
            'literal_filename_equals_catalog_basename', 'filename_subject_agrees', 'file_size_agrees',
            'manifest_cell_agrees_with_unique_July_session', 'unique_July_cell_and_session_same_row'))
        outputs.append({'asset_id': asset['asset_id'], 'path': asset['path'],
                        'July_session_source_row_numbers': [r['source_row_number'] for r in parent],
                        'manifest_candidates': candidates, 'complete_external_file_reference_agreement': confirmed,
                        'internal_recording_identity_confirmed': False, 'specimen_eligibility_established': False})
    pairs = [c for a in outputs for c in a['manifest_candidates']]
    return {'assets': outputs, 'unreferenced_manifest_source_row_numbers': [r['source_row_number'] for r in rows if r['source_row_number'] not in used],
            'duplicate_manifest_sessions': sorted(s for s, r in files.items() if len(r) > 1),
            'duplicate_July_sessions': sorted(s for s, r in sessions.items() if len(r) > 1),
            'duplicate_July_cells': sorted(s for s, r in cells.items() if len(r) > 1),
            'summary': {'source_rows': len(rows), 'catalog_assets': len(assets), 'July_rows': len(july),
                        'recognized_ephys_source_rows': len(row_tokens),
                        'unmatched_assets': sum(not a['manifest_candidates'] for a in outputs),
                        'ambiguous_manifest_assets': sum(len(a['manifest_candidates']) > 1 for a in outputs),
                        'assets_without_July_session': sum(not a['July_session_source_row_numbers'] for a in outputs),
                        'complete_external_file_reference_agreements': sum(a['complete_external_file_reference_agreement'] for a in outputs),
                        'candidate_flag_counts': {key: {'true': sum(c[key] is True for c in pairs),
                            'false': sum(c[key] is False for c in pairs), 'unknown': sum(c[key] is None for c in pairs)}
                            for key in ('literal_filename_equals_catalog_basename', 'filename_subject_agrees',
                                'file_size_agrees', 'manifest_cell_agrees_with_unique_July_session',
                                'unique_July_cell_and_session_same_row', 'cross_source_checksum_agreement')}}}


def main():
    if digest(REG) != REG_SHA:
        raise ValueError('sealed registration changed')
    reg = yaml.load(REG.read_text(), Loader=yaml.CSafeLoader)
    sources = {}
    for key, pin in reg['sources'].items():
        if digest(pin['path']) != pin['sha256']:
            raise ValueError('pinned input changed')
        sources[key] = yaml.load(Path(pin['path']).read_text(), Loader=yaml.CSafeLoader)
    if sources['manifest_verification']['result_sha256'] != reg['sources']['manifest']['sha256']:
        raise ValueError('independent verification does not cover manifest')
    root = Path(reg['output_directory'])
    root.mkdir()
    report = {'status': '2021-file-reference-comparison-failure-retained',
              'registration_sha256': digest(REG), 'collector_sha256': digest(__file__),
              'source_hashes': reg['sources'], 'started_at_utc': datetime.now(UTC).isoformat(),
              'network_requests': 0, 'physiological_arrays_read': False,
              'specimen_qualification_performed': False, 'frozen_baseline_modified': False}
    try:
        report.update(compare(sources['manifest']['rows'], sources['catalog']['assets'], sources['July']['July_rows']))
        if any(digest(pin['path']) != pin['sha256'] for pin in reg['sources'].values()):
            raise ValueError('evidence changed during comparison')
        report['status'] = '2021-file-reference-comparison-independent-verification-pending'
    except (ValueError, TypeError, KeyError, OSError) as exc:
        report['error'] = str(exc)
    report['finished_at_utc'] = datetime.now(UTC).isoformat()
    with (root/'manifest.yaml').open('x') as handle:
        yaml.safe_dump(report, handle, sort_keys=False)
    print(report['status'])
    print(yaml.safe_dump(report.get('summary', {}), sort_keys=False))
    raise SystemExit(0 if report['status'] == '2021-file-reference-comparison-independent-verification-pending' else 1)


if __name__ == '__main__':
    main()
