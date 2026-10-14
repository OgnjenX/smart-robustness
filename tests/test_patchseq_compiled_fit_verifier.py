"""Synthetic compiled-route orchestration and prerequisite controls."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

scripts = Path(__file__).parents[1]/'scripts'
names = ('verify_patchseq_whole_file_integrity', 'verify_patchseq_fitting_targets',
         'audit_patchseq_fit_checkpoint', 'patchseq_hh_independent_replay',
         'patchseq_hh_compiled_prototype', 'patchseq_hh_compiled_driver',
         'patchseq_independent_fit_inputs', 'patchseq_independent_fit_loss',
         'verify_patchseq_fit_checkpoint_structure', 'verify_patchseq_isolated_fit_replay',
         'verify_patchseq_compiled_fit')
for name in names:
    spec = importlib.util.spec_from_file_location(name, scripts/(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
mod = sys.modules['verify_patchseq_compiled_fit']


def test_missing_or_partial_equivalence_blocks_route():
    for value in ({}, {'status': 'synthetic-compiled-equivalence-retained', 'all_cases_pass': True,
                      'biological_data_used': False, 'cases': [{'passed': True}]}):
        with pytest.raises(ValueError): mod.require_equivalence(value)


def test_full_equivalence_required():
    value = {'status': 'synthetic-compiled-equivalence-retained', 'all_cases_pass': True,
             'biological_data_used': False, 'cases': [{'passed': True} for _ in range(386)]}
    mod.require_equivalence(value)
    value['cases'][0]['passed'] = False
    with pytest.raises(ValueError): mod.require_equivalence(value)


def fixture():
    feature = {'baseline_mean_mV': -65., 'early_step_mean_mV': -60., 'steady_step_mean_mV': -60.,
               'detected_event_count': 0, 'first_event_latency_s': None, 'last_to_first_interval_ratio': None}
    records = [{'partition': 'fitting', 'source_group': 'Sst', 'layer_label': 'L5',
                'source_batch': 'original', 'slot': 1, 'frozen_group_id': 'g',
                'sweeps': [{'state': 'fitting-response-characterized', 'sweep_number': 4, 'features': feature}]}]
    inputs = {('original', 1, 4): (np.zeros(4), 50000., 0., {})}
    expected = {'loss': 0., 'group_losses': {'g': 0.}, 'simulation_failures': [],
                'all_training_observable_gates_pass': True}
    return feature, records, inputs, expected


def test_injected_engine_explicit_not_hidden_global_swap(monkeypatch):
    feature, records, inputs, expected = fixture()
    monkeypatch.setattr(mod, 'reconstruct', lambda *args: feature)
    assert mod.verify_candidate(lambda *args: np.zeros((1, 4)), {}, records, inputs, expected, 1.)['passed']


def test_numerical_failure_retained(monkeypatch):
    _, records, inputs, expected = fixture()

    def failure(*args):
        raise ValueError('compiled failure')

    expected.update(loss=4000/6, group_losses={'g': 4000/6}, all_training_observable_gates_pass=False,
                    simulation_failures=[{'source_key': ['original', 1, 4]}])
    result = mod.verify_candidate(failure, {}, records, inputs, expected, 1.)
    assert result['passed'] and result['all_training_observable_gates_pass'] is False
