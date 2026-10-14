"""Synthetic search/replay controls, not fitted biological outcomes."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

scripts = Path(__file__).parents[1]/'scripts'
for name in ('patchseq_response_features', 'patchseq_isolated_hh', 'patchseq_fit_search'):
    spec = importlib.util.spec_from_file_location(name, scripts/(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
mod = sys.modules['patchseq_fit_search']


def search():
    return {'candidates_per_stratum': 4, 'bounds': {
        'capacitance_pF': [20, 400], 'leak_nS': [1, 40], 'leak_reversal_mV': [-85, -45],
        'sodium_nS': [200, 20000], 'potassium_nS': [100, 5000], 'm_current_nS': [0, 200],
        'threshold_shift_mV': [-75, -45], 'm_tau_max_ms': [400, 2500]}}


def test_candidate_exact_repeat_and_bounds():
    s = search(); first = mod.candidates(s)
    assert first == mod.candidates(s) and len(first) == 4
    for p in first:
        for name, value in mod.candidate_dict(p).items():
            lo, hi = s['bounds'][name]
            assert lo <= value <= hi


@pytest.mark.parametrize('count', [0, 1, 3, 4097, True])
def test_count_rejected(count):
    s = search(); s['candidates_per_stratum'] = count
    with pytest.raises(ValueError): mod.candidates(s)


def test_dimension_rejected():
    s = search(); s['bounds'].pop('capacitance_pF')
    with pytest.raises(ValueError): mod.candidates(s)


def test_replay_initial_state_is_model_not_observed_and_hold_is_once():
    p = mod.Parameters(100, 10, -65, 0, 0, 0, -60, 1000)
    v = mod.replay([p], np.full(100, 30.), 50000, 20, warmup_s=.020)[0]
    t = .020+np.arange(100)/50000
    np.testing.assert_allclose(v, -60-5*np.exp(-t/.010), rtol=0, atol=1e-9)
    refined = mod.replay([p], np.full(100, 30.), 50000, 20, warmup_s=.020, refinement=2)[0]
    np.testing.assert_allclose(v, refined, rtol=0, atol=1e-9)


def test_incomplete_or_failed_table_cannot_select_favorable_subset():
    with pytest.raises(ValueError): mod.selected_candidate({}, 4)
    assert mod.selected_candidate({i: {'state': 'failed'} for i in range(4)}, 4) is None


def test_tie_break_uses_parameters_then_index():
    ps = mod.candidates(search())
    rows = {i: {'state': 'scored', 'loss': 0., 'parameters': mod.candidate_dict(p)}
            for i, p in enumerate(ps)}
    assert mod.selected_candidate(rows, 4) == 0


def test_numerical_controls_do_not_hide_failed_exact_repeat(monkeypatch):
    calls = []

    def fake_replay(*args, **kwargs):
        calls.append(1)
        return np.full((1, 10), -65. if len(calls) != 2 else -64.)

    monkeypatch.setattr(mod, 'replay', fake_replay)
    monkeypatch.setattr(mod, 'characterize', lambda *args: {
        'baseline_mean_mV': -65., 'detected_event_count': 0})
    result = mod.numerical_controls(None, np.zeros(10), 50000, 0, {})
    assert len(calls) == 4
    assert not result['gates']['exact_repeat'] and not result['passed']
