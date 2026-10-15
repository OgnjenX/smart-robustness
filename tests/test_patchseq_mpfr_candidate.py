"""Short synthetic candidate-backend checks; never decode biological recordings."""

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from patchseq_mpfr_driver import FIELDS, MPFRDriver
from patchseq_precision_driver import PrecisionDriver


@pytest.fixture(scope='module')
def drivers(tmp_path_factory):
    root = tmp_path_factory.mktemp('mpfr-candidate')
    try:
        return MPFRDriver(root/'mpfr'), PrecisionDriver(root/'boost')
    except RuntimeError as exc:
        pytest.skip(str(exc))


def parameters():
    return dict(zip(FIELDS, [100., 10., -65., 2000., 500., 10., -60., 1000.], strict=True))


@pytest.mark.parametrize('bits', [160, 320])
@pytest.mark.parametrize('factor', [1, 4, 8, 16])
def test_short_current_edges_match_sealed_reference(drivers, bits, factor):
    candidate, reference = drivers
    p = parameters()
    current = np.r_[np.zeros(7), np.full(13, -50.), np.full(17, 125.), np.zeros(14)]
    initial = [-65., .05, .6, .2, .05]
    actual = candidate.integrate(p, current, 50000., initial, bits, factor)
    expected = reference.integrate(p, current, 50000., initial, bits, factor)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
    np.testing.assert_array_equal(actual[0], initial)
    assert candidate.probes[bits] == {'digits': bits, 'cancellation': 2.**-100}


@pytest.mark.parametrize('voltage', [-47., -20., -45.])
@pytest.mark.parametrize('bits', [160, 320])
def test_singular_rate_limits_match_independent_oracle(drivers, bits, voltage):
    from test_patchseq_precision_prototype import oracle

    candidate, _ = drivers
    p, initial = parameters(), [voltage, .05, .6, .2, .05]
    current = np.zeros(31)
    actual = candidate.integrate(p, current, 50000., initial, bits, 4)
    expected = oracle([p[k] for k in FIELDS], current, initial, 4)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize('bits', [160, 320])
def test_large_synthetic_excursion_matches_oracle_and_repeats_bits(drivers, bits):
    from test_patchseq_precision_prototype import oracle

    candidate, _ = drivers
    p, initial, current = parameters(), [-65., .05, .6, .2, .05], np.full(501, 2000.)
    actual = candidate.integrate(p, current, 50000., initial, bits, 1)
    repeated = candidate.integrate(p, current, 50000., initial, bits, 1)
    assert actual[:, 0].max() > 0
    np.testing.assert_array_equal(actual.view(np.uint64), repeated.view(np.uint64))
    np.testing.assert_allclose(actual, oracle([p[k] for k in FIELDS], current, initial, 1), atol=1e-12, rtol=0)


def test_thread_local_contexts_and_dependency_versions(drivers):
    candidate, _ = drivers
    assert candidate.versions == {'mpfr': '4.2.2', 'header': '4.2.2', 'gmp': '6.3.0', 'tls': 1}
    p, initial = parameters(), [-65., .05, .6, .2, .05]

    def one(bits):
        return candidate.integrate(p, np.zeros(31), 50000., initial, bits, 4)

    expected = [one(b) for b in (160, 320)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        actual = list(pool.map(one, (160, 320)))
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(a.view(np.uint64), b.view(np.uint64))


@pytest.mark.parametrize('defect', ['native_control', 'factor', 'rank', 'initial', 'nan', 'conductance'])
def test_invalid_inputs_and_native_control_not_silently_reinterpreted(drivers, defect):
    candidate, _ = drivers
    p, current, initial, bits, factor = parameters(), np.zeros(3), [-65., .05, .6, .2, .05], 160, 4
    if defect == 'native_control':
        bits = 53
    elif defect == 'factor':
        factor = 3
    elif defect == 'rank':
        current = np.zeros((3, 1))
    elif defect == 'initial':
        initial = initial[:4]
    elif defect == 'nan':
        current[0] = np.nan
    else:
        p['sodium_nS'] = -1.
    with pytest.raises(ValueError):
        candidate.integrate(p, current, 50000., initial, bits, factor)


def test_simulation_failure_remains_failure_not_clipped(drivers):
    candidate, reference = drivers
    p, initial, current = parameters(), [-65., .05, .6, .2, .05], np.array([1e12, 0.])
    for engine in (candidate, reference):
        with pytest.raises(ValueError):
            engine.integrate(p, current, 50000., initial, 160, 1)
