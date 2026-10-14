"""Reconstruct append ranks from an independently ranked doubled prefix."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml
from verify_patchseq_intrinsic_subset import reconcile
from verify_patchseq_whole_file_integrity import digest, same

REG = Path('docs/validation-results/post2008-patchseq-append-selection-registration-1372.yaml')


def reconstruct_append(base, candidates, references, manifest, original):
    doubled = copy.deepcopy(base)
    selection = doubled['selection']
    for key in ('fitting_groups_per_stratum', 'held_out_groups_per_stratum',
                'minimum_required_selection_slots', 'maximum_unique_recordings'):
        selection[key] *= 2
    # The doubled prefix is reconstruction only, not an acquisition budget.
    doubled['preflight']['maximum_unique_selected_recording_bytes'] *= 2
    combined = reconcile(doubled, candidates, references, manifest)
    old_slots, new_slots, strata = [], [], []
    for entry in combined['strata']:
        count = base['selection']['fitting_groups_per_stratum' if entry['partition'] == 'fitting'
                                  else 'held_out_groups_per_stratum']
        matching = [s for s in combined['slots'] if all(s[k] == entry[k] for k in
                    ('source_group', 'layer_label', 'partition'))]
        for index, slot in enumerate(matching):
            target = old_slots if index < count else new_slots
            slot = copy.deepcopy(slot)
            slot['slot'] = len(target)+1
            slot['group_rank_in_stratum'] = index+1 if index < count else index-count+1
            target.append(slot)
        strata.append({**entry, 'available_groups': entry['available_groups']-count,
                       'required_groups': count, 'selected_group_ids': entry['selected_group_ids'][count:]})
    same(old_slots, original['slots'])
    total = sum(s['expected_recording_bytes'] for s in new_slots)
    if total > base['preflight']['maximum_unique_selected_recording_bytes']:
        raise ValueError('append aggregate budget exceeded')
    if {s['file_path'] for s in new_slots} & {s['file_path'] for s in old_slots}:
        raise ValueError('append/original body overlap')
    return {'slots': new_slots, 'strata': strata, 'problems': [],
            'summary': {'selection_slots': len(new_slots), 'unique_recordings': len({s['file_path'] for s in new_slots}),
                        'unique_source_groups': len({s['frozen_group_id'] for s in new_slots}),
                        'known_selected_recording_bytes': total, 'slot_preflight_failures': 0,
                        'qualified_specimens': 0}}


def main():
    reg = yaml.safe_load(REG.read_text())
    result_path = Path(reg['output'])
    report = yaml.load(result_path.read_text(), Loader=yaml.CSafeLoader)
    pins = {Path(p): h for p, h in report['source_hashes'].items()}
    pins.update({result_path: digest(result_path), Path(__file__): digest(__file__),
                 Path('scripts/verify_patchseq_intrinsic_subset.py'): '9bd8b545a7a0095bfe60aaf3b6e5c6a0edec68712f6f5c8a6857c14a7b5727b3'})
    pins[Path('scripts/verify_patchseq_author_cell_association.py')] = 'f80c618ae7ed6639c97467f57aa7b8648820b65f31fcb8835853933ff7f379a4'
    if any(digest(p) != h for p, h in pins.items()):
        raise ValueError('append selection evidence changed')
    base = yaml.safe_load(Path(reg['base_registration']).read_text())
    original = yaml.load(Path(reg['original_subset']).read_text(), Loader=yaml.CSafeLoader)
    data = {k: yaml.load(Path(base[k]).read_text(), Loader=yaml.CSafeLoader)
            for k in ('candidates', 'file_references', 'typed_manifest')}
    expected = reconstruct_append(base, data['candidates'], data['file_references'], data['typed_manifest'], original)
    for key in expected:
        same(expected[key], report[key])
    if report['network_requests'] != 0 or any(report[k] is not False for k in
        ('physiology_accessed', 'recording_downloaded', 'specimen_eligibility_established')):
        raise ValueError('append scope differs')
    if any(digest(p) != h for p, h in pins.items()):
        raise ValueError('append evidence changed during verification')
    output = Path(reg['independent_output'])
    output.parent.mkdir()
    with output.open('x') as handle:
        yaml.safe_dump({'status': 'finite-intrinsic-subset-and-author-preflight-independently-reconstructed',
                       'source_hashes': {str(p): h for p, h in pins.items()}, 'summary': expected['summary'],
                       'selector_imported': False, 'network_requests': 0,
                       'physiology_accessed': False, 'original_cases_replaced': 0}, handle, sort_keys=False)
    print('Append ranks, original prefix, author hashes and byte budget independently verified')


if __name__ == '__main__':
    main()
