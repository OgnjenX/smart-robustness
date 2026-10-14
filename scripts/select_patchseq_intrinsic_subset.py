"""Seal a finite metadata-selected acquisition subset; do not read physiology."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

import yaml
from read_patchseq_2021_manifest import canonical_integer
from verify_patchseq_whole_file_integrity import digest

REG = Path('docs/validation-results/post2008-patchseq-intrinsic-subset-registration-1308.yaml')
REG_HASH = '8e26374dbd7a3e0876b18d85c2ff55854668ac7c43fb61f4c05e14777f772de7'


def cell_order(record):
    value = record['source_cell_specimen_id']
    if type(value) is not str or re.fullmatch('[1-9][0-9]*', value) is None:
        raise ValueError('noncanonical source-cell selector key')
    return int(value), record['file_path']


def select(reg, candidates, references, manifest):
    selection, limits = reg['selection'], reg['preflight']
    slots, strata, problems = [], [], []
    for group in selection['source_groups']:
        for layer in selection['source_layer_labels']:
            for partition, quota in [('fitting', selection['fitting_groups_per_stratum']),
                                     ('held-out', selection['held_out_groups_per_stratum'])]:
                choices = [r for r in candidates['records'] if r['source_annotation_candidate'] is True
                    and r['source_core_group'] == group and r['target_layer_label'] == layer
                    and r['frozen_partition'] == partition]
                by_group = {}
                for r in choices:
                    if type(r['frozen_group_id']) is not str:
                        raise ValueError('frozen group selector key missing')
                    by_group.setdefault(r['frozen_group_id'], []).append(r)
                ordered = sorted(by_group)
                if len(ordered) < quota:
                    problems.append(f'insufficient-groups:{group}:{layer}:{partition}')
                chosen = ordered[:quota]
                strata.append({'source_group': group, 'layer_label': layer, 'partition': partition,
                    'available_groups': len(ordered), 'required_groups': quota, 'selected_group_ids': chosen})
                for rank, group_id in enumerate(chosen, 1):
                    record = min(by_group[group_id], key=cell_order)
                    slots.append({'slot': len(slots)+1, 'source_group': group, 'layer_label': layer,
                        'partition': partition, 'group_rank_in_stratum': rank, 'frozen_group_id': group_id,
                        'source_cell_specimen_id': record['source_cell_specimen_id'],
                        'July_source_row_number': record['July_source_row_number'], 'file_path': record['file_path'],
                        'preflight_failures': [], 'specimen_eligibility_established': False})
    paths = [s['file_path'] for s in slots]
    if len(slots) != selection['minimum_required_selection_slots']:
        problems.append('incomplete-selection-slot-count')
    if len(set(paths)) > selection['maximum_unique_recordings'] or len(set(paths)) != len(paths):
        problems.append('unique-recording-count-or-duplicate-path')
    group_partitions = {}
    for slot in slots:
        previous = group_partitions.setdefault(slot['frozen_group_id'], slot['partition'])
        if previous != slot['partition']:
            raise ValueError('frozen group crosses selector partitions')
    total = 0
    for slot in slots:
        refs = [r for r in references['assets'] if r['path'] == slot['file_path']]
        slot['asset_reference_candidates'] = [r['asset_id'] for r in refs]
        if len(refs) != 1 or refs[0]['complete_external_file_reference_agreement'] is not True or len(refs[0]['manifest_candidates']) != 1:
            slot['preflight_failures'].append('file-reference-not-unique-and-complete')
            continue
        reference = refs[0]
        slot['asset_id'] = reference['asset_id']
        number = reference['manifest_candidates'][0]['manifest_source_row_number']
        rows = [r for r in manifest['rows'] if r['source_row_number'] == number]
        slot['author_manifest_source_row_candidates'] = [r['source_row_number'] for r in rows]
        if len(rows) != 1:
            slot['preflight_failures'].append('author-row-not-unique')
            continue
        cells = rows[0]['cells']
        cell, size = canonical_integer(cells['cell_specimen_id']), canonical_integer(cells['file_size'])
        if cell is None or size is None or str(cell) != slot['source_cell_specimen_id']:
            slot['preflight_failures'].append('author-cell-or-size-not-canonical-and-agreeing')
            continue
        if (type(rows[0]['canonical_integers']['cell_specimen_id']) is not int
            or type(rows[0]['canonical_integers']['file_size']) is not int
            or rows[0]['canonical_integers']['cell_specimen_id'] != cell or rows[0]['canonical_integers']['file_size'] != size):
            slot['preflight_failures'].append('canonical-source-projection-disagrees')
            continue
        if (cells['file_name'] is None or cells['file_name']['formula_present']
            or cells['file_name']['literal_text'] != slot['file_path'].split('/')[-1]
            or cells['sha256_checksum'] is None or cells['sha256_checksum']['formula_present']
            or type(cells['sha256_checksum']['literal_text']) is not str
            or re.fullmatch('[0-9a-fA-F]{64}', cells['sha256_checksum']['literal_text']) is None):
            slot['preflight_failures'].append('literal-author-filename-or-SHA256-invalid')
            continue
        if reference['July_session_source_row_numbers'] != [slot['July_source_row_number']]:
            slot['preflight_failures'].append('author-reference-July-row-disagrees')
        slot.update(expected_recording_bytes=size, expected_SHA256=cells['sha256_checksum']['literal_text'],
                    author_manifest_source_row_number=number)
        if size > limits['maximum_single_recording_bytes']:
            slot['preflight_failures'].append('single-recording-size-limit-exceeded')
        total += size
    if total > limits['maximum_unique_selected_recording_bytes']:
        problems.append('aggregate-recording-size-limit-exceeded')
    if any(s['preflight_failures'] for s in slots):
        problems.append('selected-slot-preflight-failure-no-replacement')
    return {'status': 'finite-subset-and-size-preflight-complete-independent-verification-pending' if not problems else 'finite-subset-preflight-incomplete-retained',
        'slots': slots, 'strata': strata, 'problems': problems,
        'summary': {'selection_slots': len(slots), 'unique_recordings': len(set(paths)),
            'unique_source_groups': len(group_partitions), 'known_selected_recording_bytes': total,
            'slot_preflight_failures': sum(bool(s['preflight_failures']) for s in slots), 'qualified_specimens': 0}}


def main():
    if digest(REG) != REG_HASH:
        raise ValueError('sealed subset registration changed')
    reg = yaml.safe_load(REG.read_text())
    keys = ('candidates', 'independent_candidates', 'file_references', 'typed_manifest')
    pins = {REG: REG_HASH, Path(__file__): digest(__file__),
        Path('scripts/read_patchseq_2021_manifest.py'): '2518610d766da2c0765b2152037cd9e24ad44378132affba538ef3b6ffdee826'}
    pins.update({Path(reg[k]): reg[k+'_sha256'] for k in keys})
    if any(digest(p) != value for p, value in pins.items()):
        raise ValueError('registered metadata or numeric parser changed')
    data = {k: yaml.load(Path(reg[k]).read_text(), Loader=yaml.CSafeLoader) for k in keys}
    if data['independent_candidates']['status'] != 'all-source-candidate-records-and-strata-independently-reconstructed':
        raise ValueError('independent candidate prerequisite missing')
    report = select(reg, data['candidates'], data['file_references'], data['typed_manifest'])
    if any(digest(p) != value for p, value in pins.items()):
        raise ValueError('metadata changed during selection')
    output = Path(reg['output'])
    output.parent.mkdir()
    report.update(source_hashes={str(p): value for p, value in pins.items()}, finished_at_utc=datetime.now(UTC).isoformat(),
        network_requests=0, physiology_accessed=False, recording_downloaded=False, specimen_eligibility_established=False)
    with output.open('x') as handle:
        yaml.safe_dump(report, handle, sort_keys=False)
    print(report['status'])


if __name__ == '__main__':
    main()
