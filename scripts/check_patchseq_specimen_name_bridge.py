"""Separate exact named-field crosswalk; the strict sample-string failure stays intact."""

from __future__ import annotations

import csv
import hashlib
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import yaml

REG = Path('docs/validation-results/post2008-patchseq-specimen-name-bridge-registration-1253.yaml')
REG_HASH = '1769ec1fcf846b340686a83f6e49500943f8f5ecb39ce66e4293c9f06e2a596c'
FIELDS = ['cell_specimen_id', 'cell_specimen_name', 'transcriptomics_sample_id']


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read_names(path, header):
    limit = csv.field_size_limit(4096)
    try:
        with Path(path).open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.reader(handle, strict=True)
            if next(reader) != header or not set(FIELDS) <= set(header):
                raise ValueError('verified identifier header required')
            positions = {k: header.index(k) for k in FIELDS}
            rows = []
            for number, values in enumerate(reader, 2):
                if len(rows) >= 10000 or len(values) != len(header):
                    raise ValueError('bounded identifier row shape required')
                raw = {k: values[p] for k, p in positions.items()}
                identity = raw['cell_specimen_id']
                rows.append({'source_row_number': number, 'raw_fields': raw,
                             'canonical_cell_specimen_id': int(identity) if re.fullmatch('[1-9][0-9]*', identity) else None})
            return rows
    finally:
        csv.field_size_limit(limit)


def bridge(names, july, methods):
    previous = {r['source_row_number']: r for r in july}
    if len(previous) != len(july) or len({r['source_row_number'] for r in names}) != len(names):
        raise ValueError('duplicate July source coordinate')
    if set(previous) != {r['source_row_number'] for r in names}:
        raise ValueError('complete July source reconciliation required')
    cells, samples = defaultdict(list), defaultdict(list)
    for row in methods:
        if row['canonical_cell_id'] is not None:
            cells[row['canonical_cell_id']].append(row)
        if row['raw_fields']['sample_id']:
            samples[row['raw_fields']['sample_id']].append(row)
    result, quality, used = [], Counter(), set()
    for row in names:
        raw = row['raw_fields']
        old = previous[row['source_row_number']]
        if (old['canonical_ids']['cell_specimen_id'] != row['canonical_cell_specimen_id']
            or old['raw_fields']['cell_specimen_id'] != raw['cell_specimen_id']
            or old['raw_fields']['transcriptomics_sample_id'] != raw['transcriptomics_sample_id']):
            raise ValueError('previous July source identifiers disagree')
        by_cell = cells.get(row['canonical_cell_specimen_id'], [])
        by_name = samples.get(raw['cell_specimen_name'], []) if raw['cell_specimen_name'] else []
        unique = len(by_cell) == len(by_name) == 1
        same = unique and by_cell[0]['source_row_number'] == by_name[0]['source_row_number']
        used.update(r['source_row_number'] for r in (*by_cell, *by_name))
        result.append({'July_source_row_number': row['source_row_number'], 'July_raw_identifier_fields': raw,
                       'July_canonical_cell_specimen_id': row['canonical_cell_specimen_id'],
                       'methods_cell_candidates': by_cell, 'methods_name_candidates': by_name,
                       'unique_explicit_specimen_name_bridge': same,
                       'unique_cell_name_conflict': unique and not same,
                       'methods_sample_equals_July_transcriptomics_sample': by_cell[0]['raw_fields']['sample_id'] == raw['transcriptomics_sample_id']
                           if len(by_cell) == 1 else None,
                       'mapping_confidence_assigned': False, 'specimen_eligibility_established': False})
        if same:
            quality[by_cell[0]['raw_fields']['mapping_quality']] += 1
    return {'July_name_rows': names, 'bridges': result,
            'unreferenced_methods_source_row_numbers': [r['source_row_number'] for r in methods if r['source_row_number'] not in used],
            'duplicate_methods_cell_ids': sorted(k for k, r in cells.items() if len(r) > 1),
            'duplicate_methods_nonempty_sample_ids': sorted(k for k, r in samples.items() if len(r) > 1),
            'summary': {'July_rows': len(names), 'methods_rows': len(methods),
                'unique_explicit_specimen_name_bridges': sum(r['unique_explicit_specimen_name_bridge'] for r in result),
                'July_rows_without_methods_cell': sum(not r['methods_cell_candidates'] for r in result),
                'July_rows_without_methods_name': sum(not r['methods_name_candidates'] for r in result),
                'ambiguous_methods_cell_candidates': sum(len(r['methods_cell_candidates']) > 1 for r in result),
                'ambiguous_methods_name_candidates': sum(len(r['methods_name_candidates']) > 1 for r in result),
                'unique_cell_name_conflicts': sum(r['unique_cell_name_conflict'] for r in result),
                'literal_transcriptomics_sample_agreements': sum(r['methods_sample_equals_July_transcriptomics_sample'] is True for r in result),
                'raw_quality_counts_for_unique_name_bridges': dict(sorted(quality.items()))}}


def main():
    if sha(REG) != REG_HASH:
        raise ValueError('sealed name bridge registration changed')
    reg = yaml.load(REG.read_text(), Loader=yaml.CSafeLoader)
    for pin in reg['sources'].values():
        if sha(pin['path']) != pin['sha256']:
            raise ValueError('pinned source changed')
    def load(key):
        return yaml.load(Path(reg['sources'][key]['path']).read_text(), Loader=yaml.CSafeLoader)
    header = load('July_schema')['schema']['columns']
    july, methods = load('July_projection')['July_rows'], load('methods_inventory')['rows']
    root = Path(reg['output_directory'])
    root.mkdir()
    report = {'status': 'specimen-name-bridge-failure-retained', 'registration_sha256': REG_HASH,
              'collector_sha256': sha(__file__), 'source_hashes': reg['sources'],
              'started_at_utc': datetime.now(UTC).isoformat(), 'network_requests': 0,
              'physiological_features_interpreted': False, 'mapping_confidence_assigned': False,
              'specimen_eligibility_established': False, 'failed_strict_test_modified': False}
    try:
        report.update(bridge(read_names(reg['sources']['July_csv']['path'], header), july, methods))
        if any(sha(pin['path']) != pin['sha256'] for pin in reg['sources'].values()):
            raise ValueError('source changed during name bridge')
        report['status'] = 'specimen-name-bridge-independent-verification-pending'
    except (OSError, ValueError, csv.Error, UnicodeError, KeyError, TypeError) as exc:
        report['error'] = str(exc)
    report['finished_at_utc'] = datetime.now(UTC).isoformat()
    with (root/'manifest.yaml').open('x') as handle:
        yaml.safe_dump(report, handle, sort_keys=False)
    print(report['status'])
    print(yaml.safe_dump(report.get('summary', {}), sort_keys=False))
    raise SystemExit(0 if report['status'] == 'specimen-name-bridge-independent-verification-pending' else 1)


if __name__ == '__main__':
    main()
