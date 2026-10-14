"""Empty held-out stratum cannot be rescued by fitting cases or other layers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

scripts = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(scripts))
try:
    spec = importlib.util.spec_from_file_location('subset_ceiling', scripts/'audit_patchseq_fixed_subset_ceiling.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
finally:
    sys.path.pop(0)


def fixture():
    slots = [{'slot': k, 'source_cell_specimen_id': str(k), 'source_group': 'Pvalb', 'layer_label': layer,
              'partition': partition, 'frozen_group_id': group}
             for k, layer, partition, group in [(1, 'L23', 'fitting', 'a'), (2, 'L23', 'held-out', 'b'),
                                                (3, 'L5', 'held-out', 'c')]]
    waves = [{'slot': s['slot'], 'source_cell_specimen_id': str(s['slot']), 'sweeps': [{'sweep_QC_pass': s['slot'] != 2}]} for s in slots]
    health = [{'slot': s['slot'], 'mandatory_gate': 'pass'} for s in slots]
    return slots, waves, health


def test_no_layer_or_partition_pooling_and_no_invented_fit_minimum():
    out = module.audit(*fixture())
    assert out['all_strata_nonempty_after_fixed_QC'] is False
    empty = out['strata_without_any_joint_supported_group']
    assert len(empty) == 1 and empty[0]['layer'] == 'L23' and empty[0]['partition'] == 'held-out'
    assert out['fit_target_diversity_established'] is False


def test_group_counts_not_cell_counts():
    slots, waves, health = fixture()
    slots.append({**slots[0], 'slot': 4, 'source_cell_specimen_id': '4'})
    waves.append({'slot': 4, 'source_cell_specimen_id': '4', 'sweeps': [{'sweep_QC_pass': True}]})
    health.append({'slot': 4, 'mandatory_gate': 'pass'})
    row = module.audit(slots, waves, health)['strata'][0]
    assert row['selected_groups'] == 1 and row['joint_groups'] == 1


def test_scope_disagreement_rejected():
    slots, waves, health = fixture()
    with pytest.raises(ValueError):
        module.audit(slots, waves[:1], health)


def test_entirely_absent_design_stratum_not_ignored():
    expected = {('Pvalb', 'L23', 'fitting'), ('Pvalb', 'L23', 'held-out'),
                ('Pvalb', 'L5', 'held-out'), ('Pvalb', 'L5', 'fitting')}
    out = module.audit(*fixture(), expected)
    assert any(r['layer'] == 'L5' and r['partition'] == 'fitting' and r['selected_groups'] == 0
               for r in out['strata_without_any_joint_supported_group'])
