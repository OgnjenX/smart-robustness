"""Prospective author-annotated targets, never physiology qualification."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import yaml
from verify_patchseq_whole_file_integrity import digest

REG = Path('docs/validation-results/post2008-patchseq-source-candidate-registration-1303.yaml')


def unique_index(records, key):
    indexed = {r[key]: r for r in records}
    if len(indexed) != len(records):
        raise ValueError('duplicate source row coordinates')
    return indexed


def build(reg, ledger, coherence, split, association):
    by_row = unique_index(coherence['records'], 'July_source_row_number')
    membership = {}
    for r in split['records']:
        for number in r['July_candidate_source_rows']:
            membership.setdefault(number, []).append(r)
    if {r['July_source_row_number'] for r in ledger['specimens']} != set(by_row):
        raise ValueError('complete source-row scope mismatch')
    outputs = []
    for specimen in ledger['specimens']:
        number = specimen['July_source_row_number']
        annotation = by_row[number]
        identity = specimen['canonical_ids']['cell_specimen_id']
        if (type(identity) is not int or identity <= 0 or type(annotation['canonical_cell_id']) is not int
            or identity != annotation['canonical_cell_id'] or specimen['raw_fields'] != annotation['July_raw_fields']):
            raise ValueError('source identity or original annotation disagreement')
        reasons = []
        group, anatomy = specimen['source_core_inhibitory_group'], specimen['raw_fields']['structure']
        layer = reg['target_anatomy_labels'].get(anatomy)
        if group not in reg['target_groups']:
            reasons.append('source-group-not-target')
        if layer is None:
            reasons.append('anatomy-label-not-target')
        if specimen['one_agreeing_source_reference_leaf'] is not True:
            reasons.append('source-reference-leaf-not-agreeing')
        candidates = annotation['methods_candidates']
        source = candidates[0]['raw_fields'] if len(candidates) == 1 else None
        if source is None or annotation['unique_cell_ID_annotation_association'] is not True:
            reasons.append('methods-cell-association-not-unique')
        if annotation['literal_structure_agrees'] is not True:
            reasons.append('literal-anatomy-agreement-not-established')
        predicates = reg['source_label_predicates']
        for field, policy in [('class', 'raw_methods_class'), ('Gouwens_etal_2020', 'raw_methods_Gouwens_etal_2020'),
                              ('mapping_quality', 'raw_methods_mapping_quality')]:
            if source is None or source[field] != predicates[policy]:
                reasons.append('raw-source-label-not-required:'+field)
        groups = membership.get(number, [])
        frozen = groups[0] if len(groups) == 1 else None
        if frozen is None or frozen['unresolved']:
            reasons.append('frozen-membership-not-unique-and-resolved')
        if frozen is not None and 'author_cell:'+str(identity) not in frozen['group_keys']:
            raise ValueError('frozen membership has different author cell')
        verified = None
        if str(identity) == association['source_cell_specimen_id']:
            if (number != association['July_source_row_number'] or frozen is None
                or frozen['file_path'] != association['file_path']):
                raise ValueError('verified pilot association disagrees')
            verified = True
        outputs.append({'July_source_row_number': number, 'source_cell_specimen_id': str(identity),
            'source_core_group': group, 'raw_anatomy_label': anatomy, 'target_layer_label': layer,
            'raw_author_labels': None if source is None else {k: source[k] for k in ('class', 'Gouwens_etal_2020', 'mapping_quality')},
            'source_annotation_candidate': not reasons, 'candidate_failures_or_unknowns': reasons,
            'frozen_group_id': None if frozen is None else frozen['group_id'],
            'frozen_partition': None if frozen is None else frozen['partition'],
            'file_path': None if frozen is None else frozen['file_path'],
            'recording_content_session_author_cell_provenance_verified': verified,
            'prior_strict_cell_sample_correspondence': annotation['prior_strict_cell_sample_correspondence'],
            'prior_specimen_name_bridge': annotation['prior_specimen_name_bridge'],
            'RNA_alias_equivalence_established': False, 'mapping_confidence_assigned': False,
            'isolated_physiology_validated': False, 'specimen_eligibility_established': False})
    strata = []
    for group in reg['target_groups']:
        for layer in sorted(set(reg['target_anatomy_labels'].values())):
            for partition in ('fitting', 'held-out'):
                rows = [r for r in outputs if r['source_annotation_candidate'] and r['source_core_group'] == group
                        and r['target_layer_label'] == layer and r['frozen_partition'] == partition]
                strata.append({'source_group': group, 'anatomy_layer_label': layer, 'partition': partition,
                    'candidate_records': len(rows), 'conservative_source_groups': len({r['frozen_group_id'] for r in rows})})
    return {'records': outputs, 'strata': strata, 'summary': {'records': len(outputs),
        'source_annotation_candidates': sum(r['source_annotation_candidate'] for r in outputs),
        'failure_reason_counts_nonexclusive': dict(sorted(Counter(reason for r in outputs for reason in r['candidate_failures_or_unknowns']).items())),
        'qualified_specimens': 0}}


def main():
    reg = yaml.safe_load(REG.read_text())
    pins = {REG: digest(REG), Path(__file__): digest(__file__)}
    for pin in reg['sources'].values():
        pins[Path(pin['path'])] = pin['sha256']
    if any(digest(p) != value for p, value in pins.items()):
        raise ValueError('registered sources changed')
    data = {k: yaml.load(Path(v['path']).read_text(), Loader=yaml.CSafeLoader) for k, v in reg['sources'].items()}
    if (data['independent_split']['status'] != 'conservative-source-groups-and-partitions-independently-reconstructed'
        or data['pilot_association']['status'] != 'external-author-cell-association-independently-reconstructed'):
        raise ValueError('independent prerequisites missing')
    report = build(reg, data['source_ledger'], data['coherence'], data['split'], data['pilot_association']['association'])
    if len(report['records']) != 4435 or any(digest(p) != value for p, value in pins.items()):
        raise ValueError('source scope or hashes changed')
    output = Path(reg['output'])
    output.parent.mkdir()
    report.update(status='source-annotated-candidates-built-independent-verification-pending',
        source_hashes={str(p): value for p, value in pins.items()}, finished_at_utc=datetime.now(UTC).isoformat(),
        network_requests=0, physiology_accessed=False, specimen_eligibility_established=False)
    with output.open('x') as handle:
        yaml.safe_dump(report, handle, sort_keys=False)
    print('Source-annotated targets retained with frozen splits; no physiology cohort qualified')


if __name__ == '__main__':
    main()
