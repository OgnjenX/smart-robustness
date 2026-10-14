"""Synthetic numerical equivalence only; does not enable compiled biological replay."""

import importlib.util
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

scripts = Path(__file__).parents[1]/'scripts'
for name in ('patchseq_hh_independent_replay', 'patchseq_hh_compiled_prototype'):
    spec = importlib.util.spec_from_file_location(name, scripts/(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
reference = sys.modules['patchseq_hh_independent_replay']
mod = sys.modules['patchseq_hh_compiled_prototype']


@pytest.fixture(scope='module')
def engine():
    if shutil.which('clang++') is None:
        pytest.skip('optional compiler unavailable; compiled biological replay is not enabled')
    with mod.compiled_engine() as integrate:
        yield integrate


def params(active=False):
    return {'capacitance_pF': 100., 'leak_nS': 10., 'leak_reversal_mV': -65.,
            'sodium_nS': 2000. if active else 0., 'potassium_nS': 500. if active else 0.,
            'm_current_nS': 10. if active else 0., 'threshold_shift_mV': -60., 'm_tau_max_ms': 1000.}


def test_passive_step_and_repeat(engine):
    current = np.zeros((1000, 1)); current[500:] = 50.
    actual = engine([params()], current, 50000, [-65.])
    expected = reference.integrate([params()], current, 50000, [-65.])
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-9)
    np.testing.assert_array_equal(actual, engine([params()], current, 50000, [-65.]))


def test_active_mixed_vector_equivalence(engine):
    rows = [params(True), params(False)]
    current = np.column_stack([np.full(1000, 500.), np.full(1000, 50.)])
    actual = engine(rows, current, 50000, [-65., -65.])
    expected = reference.integrate(rows, current, 50000, [-65., -65.])
    assert actual[0].max() > 0
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-5)


@pytest.mark.parametrize('rate', [0, 20000, float('nan'), 1000001])
def test_invalid_inputs(engine, rate):
    with pytest.raises(ValueError): engine([params()], np.zeros((2, 1)), rate, [-65.])


def test_failure_not_zero_filled(engine):
    with pytest.raises(ValueError): engine([params(True)], np.full((1000, 1), 1e20), 50000, [-65.])
