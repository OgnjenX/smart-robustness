"""Retain source annotation revisions, ambiguity and literal disagreements."""

from __future__ import annotations

import csv
import hashlib
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import yaml

REG = Path('docs/validation-results/post2008-patchseq-July-June-comparison-registration-1197.yaml')
COMMON = ('project', 'cell_specimen_id', 'donor_id', 'ephys_session_id',
          'transcriptomics_sample_id', 'structure', 'full_genotype')
FIELDS = (*COMMON, 'T-type Accession ID', 'T-type Label', 'MET-type Label')
IDS = ('cell_specimen_id', 'donor_id', 'ephys_session_id')
PAIRS = {**{k: k for k in COMMON}, 'T-type Accession ID': 'corresponding_AIT2.3.1_ID',
         'T-type Label': 'corresponding_AIT2.3.1_alias'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(65536), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return int(value) if type(value) is str and re.fullmatch('[1-9][0-9]*', value) else None


def read_rows(path, header):
    if path.stat().st_size > 2097152:
        raise ValueError('July annotation byte bound exceeded')
    limit = csv.field_size_limit(4096)
    try:
        with path.open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.reader(handle, strict=True)
            if next(reader) != header or not set(FIELDS) <= set(header):
                raise ValueError('exact July schema required')
            positions = {k: header.index(k) for k in FIELDS}
            rows = []
            for number, values in enumerate(reader, 2):
                if len(values) != len(header) or len(rows) >= 10000:
                    raise ValueError('source shape or count bound exceeded')
                raw = {k: values[i] for k, i in positions.items()}
                rows.append({'source_row_number': number, 'raw_fields': raw,
                             'canonical_ids': {k: canonical(raw[k]) for k in IDS}})
            return rows
    finally:
        csv.field_size_limit(limit)


def compare_rows(july, june):
    cells, sessions = defaultdict(list), defaultdict(list)
    for row in june:
        for key, table in (('cell_specimen_id', cells), ('ephys_session_id', sessions)):
            value = row['canonical_ids'][key]
            if value is not None:
                table[value].append(row)
    outputs, used = [], set()
    for row in july:
        cell = cells.get(row['canonical_ids']['cell_specimen_id'], [])
        session = sessions.get(row['canonical_ids']['ephys_session_id'], [])
        used.update(r['source_row_number'] for r in cell)
        differences = {k: row['raw_fields'][k] != cell[0]['raw_fields'][v] for k, v in PAIRS.items()} if len(cell) == 1 else None
        consistent = (len(cell) == len(session) == 1
                      and cell[0]['source_row_number'] == session[0]['source_row_number'])
        outputs.append({'July_source_row_number': row['source_row_number'],
                        'June_cell_candidates': [r['source_row_number'] for r in cell],
                        'June_session_candidates': [r['source_row_number'] for r in session],
                        'one_cell_and_session_correspondence': consistent,
                        'raw_field_differences': differences, 'biological_eligibility_established': False})
    duplicate = {}
    for key in IDS:
        counts = Counter(r['canonical_ids'][key] for r in july if r['canonical_ids'][key] is not None)
        duplicate[key] = sorted(v for v, n in counts.items() if n > 1)
    return {'July_rows': july, 'comparisons': outputs,
            'unreferenced_June_rows_by_cell': [r['source_row_number'] for r in june if r['source_row_number'] not in used],
            'July_duplicate_IDs': duplicate,
            'July_noncanonical_ID_counts': {k: sum(r['canonical_ids'][k] is None for r in july) for k in IDS},
            'raw_field_change_counts': {k: sum(r['raw_field_differences'] is not None and r['raw_field_differences'][k] for r in outputs) for k in PAIRS},
            'raw_MET_label_counts': dict(sorted(Counter(r['raw_fields']['MET-type Label'] for r in july).items())),
            'summary': {'July_rows': len(july), 'June_rows': len(june),
                        'one_cell_and_session_correspondences': sum(r['one_cell_and_session_correspondence'] for r in outputs),
                        'July_rows_without_June_cell': sum(not r['June_cell_candidates'] for r in outputs),
                        'July_rows_with_ambiguous_June_cell': sum(len(r['June_cell_candidates']) > 1 for r in outputs),
                        'July_rows_without_June_session': sum(not r['June_session_candidates'] for r in outputs)}}


def main():
    if digest(REG) != '527859a8bb2966e04f21c21409110af7207b5e7c7582b3ec05d021a1eac31520':
        raise ValueError('sealed revision comparison registration changed')
    reg = yaml.safe_load(REG.read_text())
    pins = {Path(reg[k]): reg[k + '_sha256'] for k in ('July_assessment', 'July_csv', 'July_schema', 'June_identity', 'June_verification')}
    if any(digest(p) != h for p, h in pins.items()):
        raise ValueError('source evidence changed')
    root = Path(reg['output_directory'])
    root.mkdir()
    load = lambda p: yaml.load(Path(p).read_text(), Loader=yaml.CSafeLoader)
    report = {'status': 'annotation-revision-comparison-failure-retained', 'registration_sha256': digest(REG),
              'collector_sha256': digest(__file__), 'source_hashes': {str(p): h for p, h in pins.items()},
              'started_at_utc': datetime.now(UTC).isoformat(), 'network_requests': 0,
              'physiological_arrays_read': False, 'parameter_fitting_performed': False,
              'biological_eligibility_established': False}
    try:
        if load(reg['June_verification'])['status'] != 'identity-annotations-independently-verified':
            raise ValueError('June independent gate absent')
        report.update(compare_rows(read_rows(Path(reg['July_csv']), load(reg['July_schema'])['schema']['columns']),
                                   load(reg['June_identity'])['annotation_rows']))
        if any(digest(p) != h for p, h in pins.items()):
            raise ValueError('source changed during comparison')
        report['status'] = 'annotation-revisions-compared-independent-verification-pending'
    except (OSError, ValueError, KeyError, TypeError, csv.Error) as exc:
        report['error'] = str(exc)
    report['finished_at_utc'] = datetime.now(UTC).isoformat()
    with (root / 'manifest.yaml').open('x') as handle:
        yaml.safe_dump(report, handle, sort_keys=False)
    print(report['status'])
    raise SystemExit(0 if report['status'] == 'annotation-revisions-compared-independent-verification-pending' else 1)


if __name__ == '__main__':
    main()
