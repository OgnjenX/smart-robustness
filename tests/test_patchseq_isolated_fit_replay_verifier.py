"""Synthetic orchestration tests; no real fit checkpoint, HDF5 or neuron replay."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

scripts = Path(__file__).parents[1]/'scripts'
for name in ('verify_patchseq_whole_file_integrity', 'verify_patchseq_fitting_targets',
             'audit_patchseq_fit_checkpoint', 'patchseq_hh_independent_replay',
             'patchseq_independent_fit_inputs', 'patchseq_independent_fit_loss',
             'verify_patchseq_fit_checkpoint_structure', 'verify_patchseq_isolated_fit_replay'):
    spec = importlib.util.spec_from_file_location(name, scripts/(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
mod = sys.modules['verify_patchseq_isolated_fit_replay']


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


def test_complete_candidate_matching(monkeypatch):
    feature, records, inputs, expected = fixture()
    monkeypatch.setattr(mod, 'replay', lambda *args: np.zeros((1, 4)))
    monkeypatch.setattr(mod, 'reconstruct', lambda *args: feature)
    result = mod.verify_candidate({}, records, inputs, expected, 1.)
    assert result['passed'] and all(result['checks'].values())


def test_disagreement_retained_not_repaired(monkeypatch):
    feature, records, inputs, expected = fixture()
    monkeypatch.setattr(mod, 'replay', lambda *args: np.zeros((1, 4)))
    monkeypatch.setattr(mod, 'reconstruct', lambda *args: feature | {'baseline_mean_mV': -62.})
    result = mod.verify_candidate({}, records, inputs, expected, 1.)
    assert result['passed'] is False and result['checks']['total_loss'] is False


def test_matching_numerical_failure_is_verified_as_failure(monkeypatch):
    _, records, inputs, expected = fixture()

    def fails(*args):
        raise FloatingPointError('independent overflow')

    monkeypatch.setattr(mod, 'replay', fails)
    expected.update(loss=4000/6, group_losses={'g': 4000/6},
                    all_training_observable_gates_pass=False,
                    simulation_failures=[{'source_key': ['original', 1, 4]}])
    result = mod.verify_candidate({}, records, inputs, expected, 1.)
    assert result['passed'] and not result['all_training_observable_gates_pass']
    assert len(result['simulation_failures']) == 1


def test_unexpected_failure_is_disagreement(monkeypatch):
    _, records, inputs, expected = fixture()

    def fails(*args):
        raise ValueError('unexpected failure')

    monkeypatch.setattr(mod, 'replay', fails)
    result = mod.verify_candidate({}, records, inputs, expected, 1.)
    assert not result['passed'] and not result['checks']['failure_support']


@pytest.mark.parametrize('left,right', [(float('nan'), 0), (float('inf'), 0), (1., 2.)])
def test_nonfinite_or_disagreeing_loss(left, right):
    assert not mod.close(left, right)


def test_atomic_verifier_checkpoint(tmp_path):
    import json
    path = tmp_path/'checkpoint.json'
    state = {'all_reconstructions_agree': False}
    mod.save(path, state)
    assert json.loads(path.read_text()) == state and not path.with_suffix('.pending').exists()
