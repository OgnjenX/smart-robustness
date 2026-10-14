"""Subset selection is fixed before size preflight and never repairs bad slots."""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import pytest

scripts = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(scripts))
try:
    spec = importlib.util.spec_from_file_location('intrinsic_subset', scripts/'select_patchseq_intrinsic_subset.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
finally:
    sys.path.pop(0)


def inputs():
    reg = {'selection': {'source_groups': ['Vip'], 'source_layer_labels': ['L23'],
        'fitting_groups_per_stratum': 1, 'held_out_groups_per_stratum': 1,
        'minimum_required_selection_slots': 2, 'maximum_unique_recordings': 2},
        'preflight': {'maximum_single_recording_bytes': 64, 'maximum_unique_selected_recording_bytes': 128}}
    records, assets, rows = [], [], []
    for index, (cell, group, partition) in enumerate([(43, 'a', 'fitting'), (42, 'a', 'fitting'),
                                                   (41, 'b', 'fitting'), (50, 'c', 'held-out')], 2):
        name = f'sub-1_ses-{cell}_icephys.nwb'
        path = 'sub-1/'+name
        records.append({'source_annotation_candidate': True, 'source_core_group': 'Vip', 'target_layer_label': 'L23',
            'frozen_partition': partition, 'frozen_group_id': group, 'source_cell_specimen_id': str(cell),
            'July_source_row_number': index, 'file_path': path})
        assets.append({'path': path, 'asset_id': 'asset-'+str(cell), 'complete_external_file_reference_agreement': True,
            'manifest_candidates': [{'manifest_source_row_number': index}], 'July_session_source_row_numbers': [index]})
        cells = {key: {'formula_present': False, 'storage_type': None if key in ('cell_specimen_id', 'file_size') else 's',
            'raw_value': value, 'literal_text': value} for key, value in [('cell_specimen_id', str(cell)),
                ('file_size', '60'), ('file_name', name), ('sha256_checksum', 'a'*64)]}
        rows.append({'source_row_number': index, 'cells': cells, 'canonical_integers': {'cell_specimen_id': cell, 'file_size': 60}})
    return [reg, {'records': records}, {'assets': assets}, {'rows': rows}]


def test_group_order_then_numeric_cell_order_not_lowest_cell_across_groups():
    result = module.select(*inputs())
    assert not result['problems']
    assert [s['source_cell_specimen_id'] for s in result['slots']] == ['42', '50']
    assert [s['frozen_group_id'] for s in result['slots']] == ['a', 'c']
    assert result['summary']['known_selected_recording_bytes'] == 120
    assert all(s['specimen_eligibility_established'] is False for s in result['slots'])


@pytest.mark.parametrize('case', ['oversize', 'aggregate', 'formula', 'missing_ref', 'ambiguous_ref', 'missing_group'])
def test_failure_retains_selected_slot_without_smaller_or_later_replacement(case):
    values = inputs()
    selected = values[3]['rows'][1]
    if case == 'oversize':
        selected['cells']['file_size']['raw_value'] = '65'
        selected['canonical_integers']['file_size'] = 65
    elif case == 'aggregate':
        values[0]['preflight']['maximum_unique_selected_recording_bytes'] = 100
    elif case == 'formula':
        selected['cells']['sha256_checksum']['formula_present'] = True
    elif case == 'missing_ref':
        values[2]['assets'].pop(1)
    elif case == 'ambiguous_ref':
        values[2]['assets'].append(copy.deepcopy(values[2]['assets'][1]))
    else:
        values[1]['records'] = [r for r in values[1]['records'] if r['frozen_partition'] != 'held-out']
    result = module.select(*values)
    assert result['problems'] and result['status'] == 'finite-subset-preflight-incomplete-retained'
    assert result['slots'][0]['source_cell_specimen_id'] == '42'
    assert result['summary']['qualified_specimens'] == 0


def test_source_identifiers_not_numeric_rounded_or_trimmed():
    record = {'source_cell_specimen_id': '42.0', 'file_path': 'unused'}
    with pytest.raises(ValueError):
        module.cell_order(record)
