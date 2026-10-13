"""Preserve official file records and check sealed session-to-cell references."""

from __future__ import annotations

import csv
import hashlib
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import yaml

REG = Path('docs/validation-results/post2008-patchseq-file-cross-reference-registration-1185.yaml')
FIELDS = ['file_id', 'file_name', 'file_size', 'sha256_checksum', 'md5_checksum', 'technique',
          'file_type', 'archive', 'archive_uri', 'project', 'cell_specimen_id']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(65536), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return int(value) if type(value) is str and re.fullmatch('[1-9][0-9]*', value) else None


def file_session(name):
    match = re.fullmatch(r'([1-9][0-9]*)\.nwb', name)
    if match:
        return int(match[1])
    match = re.fullmatch(r'sub-[1-9][0-9]*_ses-([1-9][0-9]*)_icephys\.nwb', name)
    return int(match[1]) if match else None


def read_rows(path):
    if path.stat().st_size > 6000000:
        raise ValueError('source byte bound exceeded')
    limit = csv.field_size_limit(4096)
    try:
        with path.open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.reader(handle, strict=True)
            if next(reader) != FIELDS:
                raise ValueError('verified header differs')
            rows = []
            for number, values in enumerate(reader, 2):
                if len(values) != len(FIELDS) or len(rows) >= 50000:
                    raise ValueError('row shape or count bound exceeded')
                raw = dict(zip(FIELDS, values, strict=True))
                rows.append({'source_row_number': number, 'raw_fields': raw,
                             'canonical_cell_specimen_id': canonical(raw['cell_specimen_id']),
                             'recognized_file_session': file_session(raw['file_name'])})
            return rows
    finally:
        csv.field_size_limit(limit)


def cross_reference(rows, assets):
    sessions = defaultdict(list)
    for row in rows:
        if row['raw_fields']['technique'] == 'intracellular_electrophysiology' and row['recognized_file_session'] is not None:
            sessions[row['recognized_file_session']].append(row)
    outputs, used = [], set()
    for asset in assets:
        candidates = sessions.get(asset['path_session_token'], [])
        used.update(r['source_row_number'] for r in candidates)
        parents = asset['annotation_candidates']
        unique = len(candidates) == len(parents) == 1
        cell = candidates[0]['canonical_cell_specimen_id'] if unique else None
        agreement = unique and cell is not None and cell == parents[0]['canonical_ids']['cell_specimen_id']
        outputs.append({'asset_id': asset['asset_id'], 'path': asset['path'],
                        'manifest_source_row_numbers': [r['source_row_number'] for r in candidates],
                        'one_manifest_and_annotation_candidate': unique,
                        'source_cell_ID_agrees': agreement,
                        'internal_recording_identity_confirmed': False,
                        'biological_eligibility_established': False})
    return {'source_rows': rows, 'assets': outputs,
            'unreferenced_source_row_numbers': [r['source_row_number'] for r in rows if r['source_row_number'] not in used],
            'duplicate_source_file_sessions': sorted(s for s, records in sessions.items() if len(records) > 1),
            'raw_value_counts': {key: dict(sorted(Counter(r['raw_fields'][key] for r in rows).items()))
                                 for key in ('technique', 'archive', 'file_type')},
            'missing_field_counts': {key: sum(r['raw_fields'][key] == '' for r in rows) for key in FIELDS},
            'summary': {'source_rows': len(rows), 'assets': len(assets),
                        'recognized_ephys_file_rows': sum(map(len, sessions.values())),
                        'agreeing_cell_references': sum(a['source_cell_ID_agrees'] for a in outputs),
                        'unmatched_assets': sum(not a['manifest_source_row_numbers'] for a in outputs),
                        'ambiguous_assets': sum(len(a['manifest_source_row_numbers']) > 1 for a in outputs),
                        'unique_candidate_cell_conflicts': sum(a['one_manifest_and_annotation_candidate'] and not a['source_cell_ID_agrees'] for a in outputs)}}


def main():
    if digest(REG) != '3a4e1000ba72d893f4b01951f9631dec6156ff35497a413245313175badfc659':
        raise ValueError('sealed cross-reference registration changed')
    reg = yaml.safe_load(REG.read_text())
    if reg['fields'] != FIELDS or any(reg[k] is not False for k in (
        'external_requests_authorized', 'recording_download_authorized', 'physiological_array_reads_authorized',
        'cell_class_or_layer_qualification_authorized', 'parameter_fitting_authorized', 'cell_or_network_execution_authorized')):
        raise ValueError('sealed interpretation scope differs')
    pins = {Path(reg[key]): reg[key + '_sha256'] for key in ('raw_manifest', 'identity_result', 'identity_verification')}
    if any(digest(path) != value for path, value in pins.items()):
        raise ValueError('pinned source or identity evidence changed')
    root = Path(reg['output_directory'])
    root.mkdir()
    report = {'registration_sha256': digest(REG), 'collector_sha256': digest(__file__),
              'source_hashes': {str(p): v for p, v in pins.items()}, 'started_at_utc': datetime.now(UTC).isoformat(),
              'status': 'file-cross-reference-failure-retained', 'network_requests': 0,
              'physiological_arrays_read': False, 'parameter_fitting_performed': False,
              'biological_eligibility_established': False}
    try:
        verified = yaml.load(Path(reg['identity_verification']).read_text(), Loader=yaml.CSafeLoader)
        identity = yaml.load(Path(reg['identity_result']).read_text(), Loader=yaml.CSafeLoader)
        if verified['status'] != 'identity-annotations-independently-verified':
            raise ValueError('independent identity source gate absent')
        report.update(cross_reference(read_rows(Path(reg['raw_manifest'])), identity['assets']))
        if any(digest(path) != value for path, value in pins.items()):
            raise ValueError('sources changed during cross-reference')
        report['status'] = 'file-cross-references-reconstructed-independent-verification-pending'
    except (OSError, ValueError, KeyError, TypeError, csv.Error) as exc:
        report['error'] = str(exc)
    report['finished_at_utc'] = datetime.now(UTC).isoformat()
    with (root / 'manifest.yaml').open('x') as handle:
        yaml.safe_dump(report, handle, sort_keys=False)
    print(report['status'])
    raise SystemExit(0 if report['status'] == 'file-cross-references-reconstructed-independent-verification-pending' else 1)


if __name__ == '__main__':
    main()
