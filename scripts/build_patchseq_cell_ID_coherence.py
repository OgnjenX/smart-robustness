"""Nonqualifying cell-ID-linked annotations with failed identity evidence intact."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import yaml

REG = Path('docs/validation-results/post2008-patchseq-cell-ID-annotation-coherence-registration-1257.yaml')
REG_HASH = '23a03d75a07f3b8ae1bed9b3d526d613ea5b991dbddc35fde8dd0f4f2c6446ad'


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def index_unique(rows, key):
    result = {r[key]: r for r in rows}
    if len(result) != len(rows):
        raise ValueError('duplicate source row coordinate')
    return result


def validate_id(value):
    if value is not None and (type(value) is not int or value <= 0):
        raise ValueError('positive canonical cell integer or unresolved ID required')


def build(methods, july, strict, names):
    index_unique(methods, 'source_row_number')
    prior = index_unique(strict, 'July_source_row_number')
    name_prior = index_unique(names, 'July_source_row_number')
    current = index_unique(july, 'source_row_number')
    if set(current) != set(prior) or set(current) != set(name_prior):
        raise ValueError('complete prior-test reconciliation required')
    by_cell = defaultdict(list)
    for row in methods:
        identity = row['canonical_cell_id']
        validate_id(identity)
        if identity is not None:
            by_cell[identity].append(row)
    records, used, counts = [], set(), {k: Counter() for k in ('mapping_quality', 'Gouwens_etal_2020', 'class', 'dataset')}
    for row in july:
        number, raw = row['source_row_number'], row['raw_fields']
        identity = row['canonical_ids']['cell_specimen_id']
        validate_id(identity)
        earlier, named = prior[number], name_prior[number]
        candidates = by_cell.get(identity, []) if identity is not None else []
        candidate_numbers = [r['source_row_number'] for r in candidates]
        if (earlier['July_canonical_cell_specimen_id'] != identity or named['July_canonical_cell_specimen_id'] != identity
            or earlier['July_raw_sample_id'] != raw['transcriptomics_sample_id']
            or named['July_raw_identifier_fields']['transcriptomics_sample_id'] != raw['transcriptomics_sample_id']
            or named['July_raw_identifier_fields']['cell_specimen_id'] != raw['cell_specimen_id']
            or earlier['July_raw_structure'] != raw['structure']
            or [r['source_row_number'] for r in earlier['methods_cell_candidates']] != candidate_numbers
            or [r['source_row_number'] for r in named['methods_cell_candidates']] != candidate_numbers):
            raise ValueError('prior identity evidence differs')
        for candidate, previous in zip(candidates, named['methods_cell_candidates'], strict=True):
            if candidate != previous:
                raise ValueError('prior raw annotation candidate differs')
        unique = len(candidates) == 1
        source = candidates[0]['raw_fields'] if unique else None
        structure = source['structure'] == raw['structure'] if unique and source['structure'] and raw['structure'] else None
        used.update(candidate_numbers)
        records.append({'July_source_row_number': number, 'July_raw_fields': raw,
            'canonical_cell_id': identity, 'methods_candidates': candidates,
            'unique_cell_ID_annotation_association': unique, 'link_evidence': 'canonical-cell-ID-only',
            'literal_structure_agrees': structure,
            'prior_strict_cell_sample_correspondence': earlier['consistent_cell_sample_correspondence'],
            'prior_specimen_name_bridge': named['unique_explicit_specimen_name_bridge'],
            'prior_literal_transcriptomics_sample_equality': named['methods_sample_equals_July_transcriptomics_sample'],
            'RNA_sample_alias_equivalence_established': False, 'internal_recording_identity_confirmed': False,
            'donor_namespace_equivalence_established': False, 'mapping_confidence_assigned': False,
            'specimen_eligibility_established': False})
        if unique:
            for key, table in counts.items():
                table[source[key]] += 1
    return {'records': records,
        'unreferenced_methods_source_row_numbers': [r['source_row_number'] for r in methods if r['source_row_number'] not in used],
        'summary': {'July_rows': len(july), 'methods_rows': len(methods),
            'unique_cell_ID_annotation_associations': sum(r['unique_cell_ID_annotation_association'] for r in records),
            'unmatched_July_cells': sum(not r['methods_candidates'] for r in records),
            'ambiguous_July_cell_associations': sum(len(r['methods_candidates']) > 1 for r in records),
            'literal_structure_agreements': sum(r['literal_structure_agrees'] is True for r in records),
            'literal_structure_conflicts': sum(r['literal_structure_agrees'] is False for r in records),
            'unknown_structure_comparisons': sum(r['literal_structure_agrees'] is None for r in records),
            'prior_strict_correspondence_passes': sum(r['prior_strict_cell_sample_correspondence'] is True for r in records),
            'prior_specimen_name_bridge_passes': sum(r['prior_specimen_name_bridge'] is True for r in records),
            'raw_source_category_counts_for_unique_cell_ID_associations': {k: dict(sorted(v.items())) for k, v in counts.items()},
            'eligible_specimens': 0}}


def main():
    if sha(REG) != REG_HASH:
        raise ValueError('sealed coherence registration changed')
    reg = yaml.load(REG.read_text(), Loader=yaml.CSafeLoader)
    sources = {}
    for key, pin in reg['sources'].items():
        if sha(pin['path']) != pin['sha256']:
            raise ValueError('pinned source changed')
        sources[key] = yaml.load(Path(pin['path']).read_text(), Loader=yaml.CSafeLoader)
    root = Path(reg['output_directory'])
    root.mkdir()
    report = {'status': 'cell-ID-coherence-failure-retained', 'registration_sha256': REG_HASH,
              'collector_sha256': sha(__file__), 'source_hashes': reg['sources'],
              'started_at_utc': datetime.now(UTC).isoformat(), 'network_requests': 0,
              'physiological_features_interpreted': False, 'mapping_confidence_assigned': False,
              'specimen_eligibility_established': False, 'prior_failed_tests_modified': False}
    try:
        report.update(build(sources['methods_inventory']['rows'], sources['July']['July_rows'],
                            sources['strict_sample_test']['correspondences'], sources['specimen_name_test']['bridges']))
        if any(sha(pin['path']) != pin['sha256'] for pin in reg['sources'].values()):
            raise ValueError('source changed during coherence inventory')
        report['status'] = 'nonqualifying-cell-ID-coherence-independent-verification-pending'
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report['error'] = str(exc)
    report['finished_at_utc'] = datetime.now(UTC).isoformat()
    with (root/'manifest.yaml').open('x') as handle:
        yaml.safe_dump(report, handle, sort_keys=False)
    print(report['status'])
    print(yaml.safe_dump(report.get('summary', {}), sort_keys=False))
    raise SystemExit(0 if report['status'] == 'nonqualifying-cell-ID-coherence-independent-verification-pending' else 1)


if __name__ == '__main__':
    main()
