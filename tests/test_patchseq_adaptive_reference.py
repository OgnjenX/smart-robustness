"""Analytic passive current-edge and scalar RHS tests, not biological validation."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
import patchseq_adaptive_reference as reference
from patchseq_hh_independent_replay import derivative as old_derivative
from patchseq_hh_independent_replay import parameters


def passive():
    return dict(zip(reference.FIELDS, [100., 10., -65., 0., 0., 0., -60., 1000.], strict=True))


@pytest.mark.parametrize('method', ['DOP853', 'Radau'])
def test_analytic_piecewise_passive_with_last_sample_edge(method):
    p = passive(); current = np.r_[np.full(100, 50.), np.full(100, -20.), 70.]
    v, receipt = reference.integrate(p, current, 50000., method, 1e-10, 1e-10, 1e-12)
    t = np.arange(100)/50000
    expected = np.r_[-60-5*np.exp(-t/.01),
                     -67+(-60-5*np.exp(-.002/.01)+67)*np.exp(-np.arange(101)/50000/.01)]
    np.testing.assert_allclose(v, expected, atol=1e-8, rtol=0)
    assert [(r['first_index'],r['last_index']) for r in receipt['segments']] == [(0,100),(100,200)]


@pytest.mark.parametrize('method', ['DOP853', 'Radau'])
def test_warmup_and_native_holding_added_once(method):
    v, _ = reference.replay(passive(), np.full(100, 30.), 50000., 20., warmup_s=.02,
                           method=method, rtol=1e-10, voltage_atol=1e-10, gate_atol=1e-12)
    expected = -60-5*np.exp(-(.02+np.arange(100)/50000)/.01)
    np.testing.assert_allclose(v, expected, atol=1e-8, rtol=0)


def test_scalar_rhs_matches_existing_unit_explicit_equations():
    p = passive();p.update(sodium_nS=8243.75,potassium_nS=253.125,m_current_nS=143.75)
    for v in (-85., -65., -40., 0., 40.):
        y = np.array([v,.2,.6,.3,.1])
        expected = old_derivative(y[:,None], np.array([30.]), parameters([p]))[:,0]
        np.testing.assert_allclose(reference.derivative(y,30.,p), expected, rtol=1e-12, atol=1e-12)


def test_evaluation_limit_retained_as_failure():
    with pytest.raises(RuntimeError, match='evaluation budget'):
        reference.integrate(passive(), np.zeros(100), 50000., 'DOP853',1e-10,1e-10,1e-12,
                            maximum_evaluations=1)
