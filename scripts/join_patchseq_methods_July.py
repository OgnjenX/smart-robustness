"""Exact source annotation correspondence, without confidence assignment."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import yaml

REG = Path('docs/validation-results/post2008-patchseq-methods-July-correspondence-registration-1249.yaml')
REG_HASH = '14b99dc2c2b4207a048447124ce245b79fa313feb6c49545250cf8c0a7866930'


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def join(methods, july):
    for records in (methods, july):
        if len({r['source_row_number'] for r in records}) != len(records):
            raise ValueError('duplicate source row coordinate')
    cells, samples = defaultdict(list), defaultdict(list)
    for row in methods:
        if row['canonical_cell_id'] is not None:
            cells[row['canonical_cell_id']].append(row)
        if row['raw_fields']['sample_id']:
            samples[row['raw_fields']['sample_id']].append(row)
    outputs, used, qualities, indicators = [], set(), Counter(), Counter()
    def candidates(records, structure):
        return [{'source_row_number': r['source_row_number'], 'raw_fields': r['raw_fields'],
                 'canonical_cell_id': r['canonical_cell_id'],
                 'literal_structure_agrees': r['raw_fields']['structure'] == structure} for r in records]

    for row in july:
        raw = row['raw_fields']
        by_cell = cells.get(row['canonical_ids']['cell_specimen_id'], [])
        by_sample = samples.get(raw['transcriptomics_sample_id'], []) if raw['transcriptomics_sample_id'] else []
        used.update(r['source_row_number'] for r in (*by_cell, *by_sample))
        unique = len(by_cell) == len(by_sample) == 1
        consistent = unique and by_cell[0]['source_row_number'] == by_sample[0]['source_row_number']

        outputs.append({'July_source_row_number': row['source_row_number'],
                        'July_canonical_cell_specimen_id': row['canonical_ids']['cell_specimen_id'],
                        'July_raw_sample_id': raw['transcriptomics_sample_id'], 'July_raw_structure': raw['structure'],
                        'methods_cell_candidates': candidates(by_cell, raw['structure']),
                        'methods_sample_candidates': candidates(by_sample, raw['structure']),
                        'consistent_cell_sample_correspondence': consistent,
                        'unique_cell_sample_conflict': unique and not consistent,
                        'mapping_confidence_assigned': False, 'specimen_eligibility_established': False})
        if consistent:
            qualities[by_cell[0]['raw_fields']['mapping_quality']] += 1
            indicators[by_cell[0]['raw_fields']['Gouwens_etal_2020']] += 1
    return {'correspondences': outputs,
            'unreferenced_methods_source_row_numbers': [r['source_row_number'] for r in methods if r['source_row_number'] not in used],
            'duplicate_methods_cell_ids': sorted(k for k, rows in cells.items() if len(rows) > 1),
            'duplicate_methods_nonempty_sample_ids': sorted(k for k, rows in samples.items() if len(rows) > 1),
            'summary': {'methods_rows': len(methods), 'July_rows': len(july),
                'consistent_cell_sample_correspondences': sum(r['consistent_cell_sample_correspondence'] for r in outputs),
                'July_rows_without_methods_cell': sum(not r['methods_cell_candidates'] for r in outputs),
                'July_rows_without_methods_sample': sum(not r['methods_sample_candidates'] for r in outputs),
                'ambiguous_methods_cell_candidates': sum(len(r['methods_cell_candidates']) > 1 for r in outputs),
                'ambiguous_methods_sample_candidates': sum(len(r['methods_sample_candidates']) > 1 for r in outputs),
                'unique_cell_sample_conflicts': sum(r['unique_cell_sample_conflict'] for r in outputs),
                'consistent_correspondence_structure_conflicts': sum(r['consistent_cell_sample_correspondence']
                    and not r['methods_cell_candidates'][0]['literal_structure_agrees'] for r in outputs),
                'raw_mapping_quality_counts_for_consistent_correspondences': dict(sorted(qualities.items())),
                'raw_paper_indicator_counts_for_consistent_correspondences': dict(sorted(indicators.items()))}}


def main():
    if sha(REG) != REG_HASH:
        raise ValueError('sealed correspondence registration changed')
    reg = yaml.load(REG.read_text(), Loader=yaml.CSafeLoader)
    sources = {}
    for key, pin in reg['sources'].items():
        if sha(pin['path']) != pin['sha256']:
            raise ValueError('pinned annotation source changed')
        sources[key] = yaml.load(Path(pin['path']).read_text(), Loader=yaml.CSafeLoader)
    root = Path(reg['output_directory'])
    root.mkdir()
    report = {'status': 'methods-July-correspondence-failure-retained',
              'registration_sha256': REG_HASH, 'collector_sha256': sha(__file__),
              'source_hashes': reg['sources'], 'started_at_utc': datetime.now(UTC).isoformat(),
              'network_requests': 0, 'physiological_features_interpreted': False,
              'mapping_confidence_assigned': False, 'specimen_eligibility_established': False}
    try:
        report.update(join(sources['methods_inventory']['rows'], sources['July']['July_rows']))
        if any(sha(pin['path']) != pin['sha256'] for pin in reg['sources'].values()):
            raise ValueError('source changed during correspondence')
        report['status'] = 'methods-July-correspondence-independent-verification-pending'
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report['error'] = str(exc)
    report['finished_at_utc'] = datetime.now(UTC).isoformat()
    with (root/'manifest.yaml').open('x') as handle:
        yaml.safe_dump(report, handle, sort_keys=False)
    print(report['status'])
    print(yaml.safe_dump(report.get('summary', {}), sort_keys=False))
    raise SystemExit(0 if report['status'] == 'methods-July-correspondence-independent-verification-pending' else 1)


if __name__ == '__main__':
    main()
