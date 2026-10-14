"""Synthetic per-feature and selected-control checks, without real source access."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

scripts = Path(__file__).parents[1]/'scripts'
for name in ('verify_patchseq_whole_file_integrity', 'verify_patchseq_fitting_targets',
             'patchseq_response_features', 'patchseq_isolated_hh', 'patchseq_fit_search',
             'patchseq_hh_independent_replay', 'patchseq_selected_fit_agreement'):
    spec = importlib.util.spec_from_file_location(name, scripts/(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
mod = sys.modules['patchseq_selected_fit_agreement']


@pytest.mark.parametrize('left,right', [(None, 0), ([1], [2]), (True, 1),
                                     ({'a': 1}, {'b': 1}), (float('nan'), float('nan'))])
def test_discrete_missing_nonfinite_and_scope_not_averaged_away(left, right):
    assert not mod.feature_agreement(left, right)


def test_small_arithmetic_float_difference_only():
    assert mod.feature_agreement({'v': -65.}, {'v': -65.+1e-6})
    assert not mod.feature_agreement({'v': -65.}, {'v': -65.+1e-3})


def setup(monkeypatch, primary_delta=0.):
    parameter = {'capacitance_pF': 100., 'leak_nS': 10., 'leak_reversal_mV': -65.,
                 'sodium_nS': 2000., 'potassium_nS': 500., 'm_current_nS': 10.,
                 'threshold_shift_mV': -60., 'm_tau_max_ms': 1000.}
    feature = {'baseline_mean_mV': -65., 'detected_event_count': 0}
    monkeypatch.setattr(mod, 'primary_replay', lambda *args: np.full((1, 10), -65.+primary_delta))
    monkeypatch.setattr(mod, 'independent_replay', lambda *args, **kwargs: np.full((1, 10), -65.))
    monkeypatch.setattr(mod, 'characterize', lambda *args: feature)
    monkeypatch.setattr(mod, 'reconstruct', lambda *args: feature)
    control = {'gates': {name: True for name in ('exact_repeat', 'refinement_voltage',
               'refinement_event_count', 'warmup_baseline', 'warmup_event_count')},
               'maximum_refinement_voltage_error_mV': 0., 'warmup_baseline_difference_mV': 0.,
               'passed': True}
    return parameter, control


def test_agreement_does_not_promote_network(monkeypatch):
    p, control = setup(monkeypatch)
    result = mod.selected_agreement(p, np.zeros(10), 50000, 0, {}, control)
    assert result['all_reconstruction_checks_pass'] and result['network_promotion'] is False


def test_waveform_disagreement_not_hidden_by_equal_features(monkeypatch):
    p, control = setup(monkeypatch, .1)
    result = mod.selected_agreement(p, np.zeros(10), 50000, 0, {}, control)
    assert result['agreement_checks']['feature_agreement']
    assert not result['all_reconstruction_checks_pass']


def test_failure_kept_unresolved_even_if_primary_failed(monkeypatch):
    p, _ = setup(monkeypatch)

    def failure(*args, **kwargs):
        raise ValueError('synthetic independent failure')

    monkeypatch.setattr(mod, 'independent_replay', failure)
    result = mod.selected_agreement(p, np.zeros(10), 50000, 0, {}, {'passed': False})
    assert result['state'] == 'selected-sweep-reconstruction-unresolved'
    assert not result['all_reconstruction_checks_pass']
