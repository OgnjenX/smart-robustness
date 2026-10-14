"""Nonqualifying source-core specimen annotations; retain unresolved evidence."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import yaml

REG = Path('docs/validation-results/post2008-patchseq-specimen-source-ledger-registration-1226.yaml')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(65536), b''):
            h.update(block)
    return h.hexdigest()


def build(july, memberships, identity, references):
    by_pair = defaultdict(list)
    for row in memberships:
        by_pair[(row['legacy_accession'], row['legacy_label'])].append(row)
    assets = {a['asset_id']: a for a in identity}
    covered_cells = set()
    for ref in references:
        if ref['source_cell_ID_agrees'] is True:
            parents = assets[ref['asset_id']]['annotation_candidates']
            if len(parents) != 1:
                raise ValueError('agreeing cell reference with ambiguous source annotations')
            covered_cells.add(parents[0]['canonical_ids']['cell_specimen_id'])
    outputs = []
    for row in july:
        raw = row['raw_fields']
        candidates = by_pair.get((raw['T-type Accession ID'], raw['T-type Label']), [])
        parent = candidates[0] if len(candidates) == 1 else None
        agreeing = parent is not None and parent['source_leaf_identity_agrees'] is True
        core = parent['exclusive_named_inhibitory_subgroup'] if agreeing else None
        outputs.append({'July_source_row_number': row['source_row_number'], 'raw_fields': raw,
                        'canonical_ids': row['canonical_ids'], 'type_membership_candidates': candidates,
                        'one_agreeing_source_reference_leaf': agreeing,
                        'source_core_inhibitory_group': core,
                        'source_groups': parent['source_groups'] if agreeing else [],
                        'June_official_manifest_cell_reference_present': row['canonical_ids']['cell_specimen_id'] in covered_cells,
                        'transcriptomic_mapping_confidence': None,
                        'mapping_confidence_available_in_release': False,
                        'internal_recording_identity_confirmed': False,
                        'isolated_physiology_validated': False,
                        'specimen_eligibility_established': False})
    counts = defaultdict(Counter)
    for row in outputs:
        counts[row['source_core_inhibitory_group'] or '__unassigned__'][row['raw_fields']['structure']] += 1
    return {'specimens': outputs,
            'raw_structure_counts_by_source_core_group': {k: dict(sorted(v.items())) for k, v in sorted(counts.items())},
            'summary': {'specimens': len(outputs),
                        'agreeing_source_reference_leaf_annotations': sum(r['one_agreeing_source_reference_leaf'] for r in outputs),
                        'source_core_group_annotated_specimens': sum(r['source_core_inhibitory_group'] is not None for r in outputs),
                        'unassigned_source_core_groups': sum(r['source_core_inhibitory_group'] is None for r in outputs),
                        'June_manifest_covered_specimens': sum(r['June_official_manifest_cell_reference_present'] for r in outputs),
                        'eligible_specimens': 0}}


def main():
    if digest(REG) != '74d973e48e8abd6ab2dfdaa076c079dfb2d0463dfe8323f860c87698342731d0':
        raise ValueError('sealed specimen ledger registration changed')
    reg = yaml.safe_load(REG.read_text())
    keys = ('July_comparison', 'memberships', 'June_identity', 'June_manifest_cross_reference')
    pins = {Path(reg[k]): reg[k + '_sha256'] for k in keys}
    if any(digest(p) != h for p, h in pins.items()):
        raise ValueError('pinned specimen source changed')
    root = Path(reg['output_directory'])
    root.mkdir()
    report = {'status': 'specimen-ledger-failure-retained', 'registration_sha256': digest(REG),
              'collector_sha256': digest(__file__), 'source_hashes': {str(p): h for p, h in pins.items()},
              'started_at_utc': datetime.now(UTC).isoformat(), 'network_requests': 0,
              'physiological_arrays_read': False, 'parameter_fitting_performed': False,
              'specimen_qualification_performed': False}
    try:
        sources = {k: yaml.load(Path(reg[k]).read_text(), Loader=yaml.CSafeLoader) for k in keys}
        report.update(build(sources['July_comparison']['July_rows'], sources['memberships']['memberships'],
                            sources['June_identity']['assets'], sources['June_manifest_cross_reference']['assets']))
        if any(digest(p) != h for p, h in pins.items()):
            raise ValueError('source changed during specimen ledger construction')
        report['status'] = 'specimen-source-core-ledger-built-independent-verification-pending'
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report['error'] = str(exc)
    report['finished_at_utc'] = datetime.now(UTC).isoformat()
    with (root / 'manifest.yaml').open('x') as handle:
        yaml.safe_dump(report, handle, sort_keys=False)
    print(report['status'])
    raise SystemExit(0 if report['status'] == 'specimen-source-core-ledger-built-independent-verification-pending' else 1)


if __name__ == '__main__':
    main()
