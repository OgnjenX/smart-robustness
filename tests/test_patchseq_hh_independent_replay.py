"""Independent numerical replay tested only against synthetic analytical/adaptive controls."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('patchseq_hh_independent_replay', Path(__file__).parents[1]/'scripts/patchseq_hh_independent_replay.py')
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def parameters(**changes):
    return {'capacitance_pF': 100., 'leak_nS': 10., 'leak_reversal_mV': -65.,
            'sodium_nS': 0., 'potassium_nS': 0., 'm_current_nS': 0.,
            'threshold_shift_mV': -60., 'm_tau_max_ms': 1000.} | changes


def test_passive_step_timing_and_exact_repeat():
    current = np.zeros((1000, 1)); current[500:] = 50.
    v = mod.integrate([parameters()], current, 50000, [-65])[0]
    np.testing.assert_array_equal(v[:500], np.full(500, -65.))
    np.testing.assert_allclose(v[500:], -60-5*np.exp(-np.arange(500)*.002), rtol=0, atol=1e-9)
    np.testing.assert_array_equal(v, mod.integrate([parameters()], current, 50000, [-65])[0])


def test_holding_once_and_warmup_refinement():
    command = np.full(100, 30.)
    v = mod.replay([parameters()], command, 50000, 20, .02)[0]
    exact = -60-5*np.exp(-(.02+np.arange(100)/50000)/.01)
    np.testing.assert_allclose(v, exact, rtol=0, atol=1e-9)
    np.testing.assert_allclose(v, mod.replay([parameters()], command, 50000, 20, .02, 2)[0], rtol=0, atol=1e-9)


def test_active_trajectory_against_dop853():
    from scipy.integrate import solve_ivp
    p = parameters(sodium_nS=2000., potassium_nS=500., m_current_nS=10.)
    matrix = mod.parameters([p])
    a, b, slow, _ = mod.kinetics(np.array([-65.]), np.array([-60.]), np.array([1000.]))
    initial = np.r_[-65, (a/(a+b))[:, 0], slow[0]]
    t = np.arange(1000)*.02
    reference = solve_ivp(lambda time, state: mod.derivative(state[:, None], np.array([500.]), matrix)[:, 0],
                          (0, 20), initial, t_eval=t, method='DOP853', rtol=1e-10, atol=1e-11)
    assert reference.success and reference.y[0].max() > 0
    v = mod.integrate([p], np.full((1000, 1), 500.), 50000, [-65])[0]
    np.testing.assert_allclose(v, reference.y[0], rtol=0, atol=.1)


def test_singularities_have_source_limits():
    a, b, _, _ = mod.kinetics(np.array([-47., -45., -20.]), np.full(3, -60.), np.full(3, 1000.))
    assert a[0, 0] == pytest.approx(1.28)
    assert a[2, 1] == pytest.approx(.16)
    assert b[0, 2] == pytest.approx(1.4)


def test_cross_engine_synthetic_active_voltage():
    path = Path(__file__).parents[1]/'scripts/patchseq_isolated_hh.py'
    spec = importlib.util.spec_from_file_location('hh_primary_synthetic_comparison', path)
    primary = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = primary
    spec.loader.exec_module(primary)
    p = parameters(sodium_nS=2000., potassium_nS=500., m_current_nS=10.)
    current = np.full((1000, 1), 500.)
    independent = mod.integrate([p], current, 50000, [-65])
    brian = primary.simulate([primary.Parameters(**p)], current, 50000, [-65])
    np.testing.assert_allclose(independent, brian, rtol=0, atol=1e-5)


@pytest.mark.parametrize('change', [{'capacitance_pF': 0}, {'sodium_nS': -1},
                                  {'m_tau_max_ms': float('nan')}, {'extra': 1}])
def test_invalid_parameters(change):
    with pytest.raises(ValueError): mod.parameters([parameters(**change)])


@pytest.mark.parametrize('rate', [0, 20000, float('nan'), 1000001])
def test_invalid_rate(rate):
    with pytest.raises(ValueError): mod.integrate([parameters()], np.zeros((2, 1)), rate, [-65])
