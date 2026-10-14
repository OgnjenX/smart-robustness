"""Batches do not double-count shared groups, discard failures or mix partitions."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
from test_patchseq_fixed_subset_ceiling import fixture

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
try:
    from audit_patchseq_combined_source_coverage import combine
finally:
    sys.path.pop(0)


def batches():
    slots, waves, health = fixture()
    for s in slots:
        s.update(file_path='old-'+str(s['slot']), asset_id='old-asset-'+str(s['slot']))
    new = [{**slots[1], 'slot': 1, 'file_path': 'new-1', 'asset_id': 'new-asset-1',
            'source_cell_specimen_id': '101', 'frozen_group_id': 'new-group'}]
    nw = [{'slot': 1, 'source_cell_specimen_id': '101', 'sweeps': [{'sweep_QC_pass': True}]}]
    nh = [{'slot': 1, 'mandatory_gate': 'pass'}]
    return [('original', slots, waves, health), ('append', new, nw, nh)]


def test_append_support_does_not_erase_original_failure():
    result = combine(batches())
    assert result['recordings_retained'] == 4
    original_held = next(r for r in result['batch_reports'][0]['strata'] if r['layer'] == 'L23' and r['partition'] == 'held-out')
    combined_held = next(r for r in result['combined']['strata'] if r['source_group'] == 'Pvalb' and r['layer'] == 'L23' and r['partition'] == 'held-out')
    assert original_held['joint_groups'] == 0
    assert combined_held['selected_groups'] == 2 and combined_held['joint_groups'] == 1
    assert result['combined']['records'][1]['any_passing_waveform'] is False
    assert result['neuron_model_eligibility_established'] is False
    assert result['combined']['all_strata_nonempty_after_fixed_QC'] is False


def test_shared_group_not_double_counted():
    data = batches()
    data[1][1][0]['frozen_group_id'] = 'b'
    result = combine(data)
    row = next(r for r in result['combined']['strata'] if r['source_group'] == 'Pvalb' and r['layer'] == 'L23' and r['partition'] == 'held-out')
    assert row['selected_groups'] == 1 and row['joint_groups'] == 1


@pytest.mark.parametrize('field', ['file_path', 'asset_id'])
def test_duplicate_body_not_admitted(field):
    data = batches()
    data[1][1][0][field] = data[0][1][0][field]
    with pytest.raises(ValueError, match='duplicated'):
        combine(data)


def test_global_group_cannot_cross_partitions():
    data = batches()
    data[1][1][0]['frozen_group_id'] = 'a'
    with pytest.raises(ValueError, match='crosses partitions'):
        combine(data)


def test_unknown_health_is_not_usable_support_and_inputs_not_mutated():
    data = batches()
    data[1][3][0]['mandatory_gate'] = 'unknown'
    original = copy.deepcopy(data)
    result = combine(data)
    assert data == original
    row = next(r for r in result['combined']['strata'] if r['source_group'] == 'Pvalb' and r['layer'] == 'L23' and r['partition'] == 'held-out')
    assert row['joint_groups'] == 0
