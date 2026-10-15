"""Synthetic build provenance and explicit-state contracts; no recordings read."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
import patchseq_precision_driver as driver


@pytest.fixture(scope='module')
def built(tmp_path_factory):
    path = tmp_path_factory.mktemp('precision-driver')
    try:
        return driver.PrecisionDriver(path), path
    except RuntimeError as exc:
        pytest.skip(str(exc))


def test_build_dependency_closure_and_reuse(built):
    first, path = built
    second = driver.PrecisionDriver(path)
    assert second.provenance == first.provenance
    assert driver.digest(first.provenance['binary_path']) == first.provenance['binary_sha256']
    assert str(driver.SOURCE) in first.provenance['dependencies']
    assert any('/boost/' in p for p in first.provenance['dependencies'])
    assert first.probes == second.probes


def test_partial_build_refused(tmp_path):
    (tmp_path/'kernel.so').write_bytes(b'not a library')
    try:
        with pytest.raises(ValueError, match='partial'):
            driver.prepare_build(tmp_path)
    except RuntimeError as exc:
        pytest.skip(str(exc))


def test_shared_initialization_has_original_rule_and_no_run():
    p = dict(zip(driver.FIELDS, [100., 10., -65., 2000., 500., 10., -60., 1000.], strict=True))
    state = driver.shared_brian_initial_state(p, 50000.)
    np.testing.assert_array_equal(state, driver.shared_brian_initial_state(p, 50000.))
    assert state.shape == (5,) and state[0] == p['leak_reversal_mV']
    assert np.all((state[1:] > 0) & (state[1:] < 1))


@pytest.mark.parametrize('mutation', ['rank', 'initial', 'bits', 'factor', 'nan'])
def test_wrapper_rejects_invalid_specification(built, mutation):
    lib, _ = built
    p = dict(zip(driver.FIELDS, [100., 10., -65., 2000., 500., 10., -60., 1000.], strict=True))
    current, initial, bits, factor = np.zeros(3), [-65., .05, .6, .2, .05], 160, 4
    if mutation == 'rank':
        current = np.zeros((3, 1))
    elif mutation == 'initial':
        initial = initial[:4]
    elif mutation == 'bits':
        bits = 80
    elif mutation == 'factor':
        factor = 3
    else:
        current[0] = np.nan
    with pytest.raises(ValueError):
        lib.integrate(p, current, 50000., initial, bits, factor)
