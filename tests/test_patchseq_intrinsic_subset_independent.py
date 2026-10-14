"""Independent SQL ranking uses numeric cells within fixed source groups."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

scripts = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(scripts))
try:
    spec = importlib.util.spec_from_file_location('independent_subset', scripts/'verify_patchseq_intrinsic_subset.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
finally:
    sys.path.pop(0)


def inputs():
    reg = {'selection': {'source_groups': ['Vip'], 'source_layer_labels': ['L23'],
        'fitting_groups_per_stratum': 1, 'held_out_groups_per_stratum': 1}}
    rows = []
    for cell, group, partition in [('10', 'a', 'fitting'), ('2', 'a', 'fitting'),
                                    ('1', 'b', 'fitting'), ('20', 'c', 'held-out')]:
        rows.append({'source_annotation_candidate': True, 'source_core_group': 'Vip', 'target_layer_label': 'L23',
            'frozen_partition': partition, 'frozen_group_id': group, 'source_cell_specimen_id': cell,
            'July_source_row_number': int(cell), 'file_path': f'file-{cell}.nwb'})
    return reg, {'records': rows}


def test_independent_rank_is_group_first_numeric_cell_second():
    slots, strata = module.ranked_subset(*inputs())
    assert [s['source_cell_specimen_id'] for s in slots] == ['2', '20']
    assert [s['frozen_group_id'] for s in slots] == ['a', 'c']
    assert [s['available_groups'] for s in strata] == [2, 1]


def test_sql_rank_does_not_depend_on_original_record_order():
    reg, data = inputs()
    original = module.ranked_subset(reg, data)
    assert module.ranked_subset(reg, {'records': list(reversed(data['records']))}) == original


def test_missing_held_out_support_fails_without_fitting_group_replacement():
    reg, data = inputs()
    data['records'] = [r for r in data['records'] if r['frozen_partition'] == 'fitting']
    with pytest.raises(ValueError, match='support insufficient'):
        module.ranked_subset(reg, data)


@pytest.mark.parametrize('cell', ['02', '2.0', '２', True])
def test_noncanonical_cell_rank_keys_are_not_normalized(cell):
    reg, data = inputs()
    data['records'][0]['source_cell_specimen_id'] = cell
    with pytest.raises(ValueError):
        module.ranked_subset(reg, data)
