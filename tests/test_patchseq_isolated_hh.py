"""Equation and numerical controls on synthetic inputs only."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('patchseq_isolated_hh', Path(__file__).parents[1]/'scripts/patchseq_isolated_hh.py')
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def params(**changes):
    fields = {'capacitance_pF': 100., 'leak_nS': 10., 'leak_reversal_mV': -65.,
              'sodium_nS': 0., 'potassium_nS': 0., 'm_current_nS': 0.,
              'threshold_shift_mV': -60., 'm_tau_max_ms': 1000.}
    return mod.Parameters(**(fields | changes))


def test_passive_analytic_and_exact_repeat():
    current = np.full((1000, 1), 50.)
    result = mod.simulate([params()], current, 50000., [-65.])
    t = np.arange(1000)/50000
    exact = -60-5*np.exp(-t/.010)
    np.testing.assert_allclose(result[0], exact, rtol=0, atol=1e-9)
    np.testing.assert_array_equal(result, mod.simulate([params()], current, 50000., [-65.]))


def test_rate_singularities_against_direct_source_formula():
    import brian2 as b
    shifted = np.array([-30., 13., 15., 40., 70.])
    group = mod.population([params()]*5, b.Clock(dt=.02*b.ms),
                           b.TimedArray(np.zeros((2, 5))*b.pA, dt=.02*b.ms), shifted-60, b)

    def trap(x, y):
        return y if x == 0 else x/np.expm1(x/y)

    expected = {'am': [.32*trap(13-v, 4) for v in shifted],
                'bm': [.28*trap(v-40, 5) for v in shifted],
                'an': [.032*trap(15-v, 5) for v in shifted],
                'ah': .128*np.exp((17-shifted)/18),
                'bh': 4/(1+np.exp((40-shifted)/5)),
                'bn': .5*np.exp((10-shifted)/40)}
    for name, value in expected.items():
        np.testing.assert_allclose(getattr(group, name)*b.ms, value, rtol=1e-12)
    np.testing.assert_allclose(group.pinf[:], 1/(1+np.exp(-((shifted-60)+35)/10)))
    np.testing.assert_allclose(group.taup/b.ms,
        1000/(3.3*np.exp(((shifted-60)+35)/20)+np.exp(-((shifted-60)+35)/20)))


def test_step_uses_sample_hold_without_early_boundary_current():
    current = np.zeros((1000, 1)); current[500:] = 50.
    result = mod.simulate([params()], current, 50000., [-65.])[0]
    t = np.arange(500)/50000
    np.testing.assert_allclose(result[:500], -65., rtol=0, atol=1e-9)
    np.testing.assert_allclose(result[500:], -60-5*np.exp(-t/.010), rtol=0, atol=1e-9)


def test_active_trajectory_against_independent_adaptive_ode():
    from scipy.integrate import solve_ivp

    def rates(v):
        u = v+60

        def trap(x, y):
            return y if abs(x/y) < 1e-12 else x/np.expm1(x/y)

        a = np.array([.32*trap(13-u, 4), .128*np.exp((17-u)/18), .032*trap(15-u, 5)])
        b = np.array([.28*trap(u-40, 5), 4/(1+np.exp((40-u)/5)), .5*np.exp((10-u)/40)])
        return a, b

    def rhs(t, state):
        v, m, h, n, p = state
        a, b = rates(v)
        pinf = 1/(1+np.exp(-(v+35)/10))
        tau = 1000/(3.3*np.exp((v+35)/20)+np.exp(-(v+35)/20))
        dv = (10*(-65-v)+2000*m**3*h*(50-v)+500*n**4*(-90-v)+10*p*(-90-v)+500)/100
        gates = a*(1-np.array([m, h, n]))-b*np.array([m, h, n])
        return np.r_[dv, gates, (pinf-p)/tau]

    a, b = rates(-65)
    initial = np.r_[-65, a/(a+b), 1/(1+np.exp(3))]
    time_ms = np.arange(1000)*.02
    reference = solve_ivp(rhs, (0, 20), initial, t_eval=time_ms,
                          method='DOP853', rtol=1e-10, atol=1e-11)
    assert reference.success and reference.y[0].max() > 0
    current = np.full((1000, 1), 500.)
    result = mod.simulate([params(sodium_nS=2000, potassium_nS=500, m_current_nS=10)],
                          current, 50000, [-65])[0]
    np.testing.assert_allclose(result, reference.y[0], rtol=0, atol=.1)


@pytest.mark.parametrize('changes', [{'capacitance_pF': 0}, {'leak_nS': -1},
                                   {'sodium_nS': -1}, {'m_tau_max_ms': 0},
                                   {'threshold_shift_mV': float('nan')}])
def test_reject_parameters(changes):
    with pytest.raises(ValueError): params(**changes)


@pytest.mark.parametrize('rate', [0, 20000, float('nan'), 1000001])
def test_reject_rate(rate):
    with pytest.raises(ValueError): mod.simulate([params()], np.zeros((2, 1)), rate, [-65])
