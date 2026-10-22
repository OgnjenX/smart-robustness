"""Short synthetic HH-adapter units, not the prospective long qualification matrix."""

from __future__ import annotations

import copy
import ctypes
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from patchseq_gbs_driver import Policy, identity
from patchseq_gbs_hh_driver import HHDriver, parameters_array, prepare_build
from patchseq_mpfr_driver import FIELDS
from patchseq_rkf78_reference import RKF78Driver

ROOT = Path(__file__).parents[1]
INITIAL = [-65., .05, .6, .2, .05]


@pytest.fixture(scope='module')
def driver(tmp_path_factory):
    try:
        return HHDriver(tmp_path_factory.mktemp('gbs-hh'))
    except RuntimeError as exc:
        pytest.skip(str(exc))
    except subprocess.CalledProcessError as exc:
        pytest.fail(str(exc.stderr))


@pytest.fixture(scope='module')
def reference(tmp_path_factory):
    try:
        return RKF78Driver(tmp_path_factory.mktemp('gbs-hh-independent-reference'))
    except RuntimeError as exc:
        pytest.skip(str(exc))
    except subprocess.CalledProcessError as exc:
        pytest.fail(str(exc.stderr))


def parameters(passive=False):
    return dict(zip(FIELDS, [100., 10., -65., 0. if passive else 2000.,
        0. if passive else 500., 0. if passive else 10., -60., 1000.], strict=True))


def context(driver, current, bits=160, policy=None, p=None, initial=None, rate=50000.):
    return driver.context(p or parameters(), current, rate, initial or INITIAL, bits, policy or Policy())


def decode(text, mp):
    digits, exponent = text.split('p')
    return mp.mpf(int(digits, 16))*mp.power(2, int(exponent))


def exact(value, bits, mp):
    """Independent canonical encoding of a finite target-width mpf value."""
    value = mp.mpf(value)
    if not value:
        return '0p0'
    sign, mantissa, exponent, width = value._mpf_
    assert width <= bits
    mantissa <<= bits-width
    exponent -= bits-width
    return ('-' if sign else '')+format(mantissa, 'x')+'p'+str(exponent)


def rhs_oracle(p, state, current, mp):
    # Independent equation expression at 150 decimal digits. Inputs are exact
    # binary64 lifts; the PRIMARY adapter intentionally shares the frozen RHS.
    p = [mp.mpf(float(p[key])) for key in FIELDS]
    v, m, h, n, slow = map(mp.mpf, state)
    u = v-p[6]
    def trap(x):
        return x/mp.expm1(x) if x else mp.mpf(1)
    am = mp.mpf(1.28)*trap((13-u)/4)
    bm = mp.mpf(1.4)*trap((u-40)/5)
    ah = mp.mpf(.128)*mp.exp((17-u)/18)
    bh = 4/(1+mp.exp((40-u)/5))
    an = mp.mpf(.16)*trap((15-u)/5)
    bn = mp.mpf(.5)*mp.exp((10-u)/40)
    inf = 1/(1+mp.exp(-(v+35)/10))
    tau = p[7]/(mp.mpf(3.3)*mp.exp((v+35)/20)+mp.exp(-(v+35)/20))
    inward = (p[1]*(p[2]-v)+p[3]*m**3*h*(50-v)
              +p[4]*n**4*(-90-v)+p[5]*slow*(-90-v)+mp.mpf(current))
    return [inward/p[0], am*(1-m)-bm*m, ah*(1-h)-bh*h, an*(1-n)-bn*n, (inf-slow)/tau]


def reseal(value):
    value['sha256'] = identity({k: v for k, v in value.items() if k != 'sha256'})
    return value


def test_preceding_generic_and_frozen_sources_not_rewritten():
    for filename in ('post2008-patchseq-gbs-arithmetic-preparation-1512.yaml',
                     'post2008-patchseq-gbs-generic-preparation-1513.yaml'):
        evidence = yaml.safe_load((ROOT/'docs/validation-results'/filename).read_text())
        for relative, expected in evidence['source_hashes'].items():
            assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() == expected
    assert hashlib.sha256((ROOT/'scripts/patchseq_mpfr_kernel.cpp').read_bytes()).hexdigest() == (
        'b0bfd3496bd77c8303798ff25502628db6c1346332759716f641e718be96f543')
    adapter = (ROOT/'scripts/patchseq_gbs_hh.cpp').read_text()
    assert '#include "patchseq_mpfr_kernel.cpp"' in adapter
    assert 'work_->rhs(' in adapter and 'work_->advance(' not in adapter


@pytest.mark.parametrize('bits', [160, 320])
@pytest.mark.parametrize('voltage', [-200., -65., -47., -45., -20., 40., 400.])
def test_five_derivatives_singular_rates_and_excursions_against_150digit_oracle(driver, bits, voltage):
    mp = pytest.importorskip('mpmath').mp.clone()
    mp.dps = 150
    state = [voltage, .05, .6, .2, .05]
    actual = driver.derivative_exact(parameters(), state, 125., bits)
    expected = rhs_oracle(parameters(), state, 125., mp)
    for encoded, value in zip(actual, expected, strict=True):
        assert abs(decode(encoded, mp)-value) <= max(1, abs(value))*mp.power(2, 12-bits)
    assert driver.environment['precision'+str(bits)] == bits


@pytest.mark.parametrize('bits', [160, 320])
def test_passive_voltage_analytic_at_every_native_sample_with_current_edges(driver, bits):
    current = np.r_[np.zeros(7), np.full(13, -50.), np.full(17, 125.), np.zeros(14)]
    with context(driver, current, bits, p=parameters(passive=True)) as live:
        actual = live.advance(len(current)-1)
        assert live.stats['dense'] > 0
    mp = pytest.importorskip('mpmath').mp.clone()
    mp.dps = 150
    delta, voltage, expected = mp.mpf(1000./50000.), mp.mpf(-65), [-65.]
    for command in current[:-1]:
        steady = mp.mpf(-65)+mp.mpf(float(command))/10
        voltage = steady+(voltage-steady)*mp.exp(-delta/10)
        expected.append(float(voltage))
    np.testing.assert_array_equal(actual[0], INITIAL)
    np.testing.assert_allclose(actual[:, 0], expected, atol=2e-12, rtol=0)
    assert np.all(actual[:, 1:] >= -1e-8) and np.all(actual[:, 1:] <= 1.0+1e-8)


@pytest.mark.parametrize('bits', [160, 320])
@pytest.mark.parametrize('voltage', [-47., -45., -20.])
def test_short_singular_rate_trajectories_against_separate_rkf78(driver, reference, bits, voltage):
    initial, current = [voltage, .05, .6, .2, .05], np.zeros(11)
    with context(driver, current, bits, initial=initial) as live:
        actual = live.advance(10)
    coarse = reference.integrate(parameters(), current, 50000., initial, bits, 2)
    fine = reference.integrate(parameters(), current, 50000., initial, bits, 4)
    np.testing.assert_allclose(coarse.states, fine.states, atol=2e-11, rtol=0)
    np.testing.assert_allclose(actual, fine.states, atol=2e-11, rtol=0)


@pytest.mark.parametrize('voltage', [-47., -45., -20.])
def test_prior_rk4_singular_test_reference_failure_is_retained(reference, voltage):
    from test_patchseq_precision_prototype import oracle

    initial, current = [voltage, .05, .6, .2, .05], np.zeros(11)
    old_reference = oracle([parameters()[key] for key in FIELDS], current, initial, 16)
    independent = reference.integrate(parameters(), current, 50000., initial, 320, 4)
    assert np.max(np.abs(old_reference-independent.states)) > 2e-11


@pytest.fixture(scope='module')
def short_active_reference(reference):
    current = np.full(501, 2000.)
    # Same input as the failed RK4 test-oracle control; no omitted/replaced case.
    coarse = reference.integrate(parameters(), current, 50000., INITIAL, 320, 2)
    fine = reference.integrate(parameters(), current, 50000., INITIAL, 320, 4)
    np.testing.assert_allclose(coarse.states, fine.states, atol=1e-8, rtol=0)
    return fine.states


def test_preexisting_short_rk4_oracle_control_remains_unqualified_at_new_test_bound():
    from test_patchseq_precision_prototype import oracle

    current = np.full(501, 2000.)
    p = [parameters()[key] for key in FIELDS]
    coarse = oracle(p, current, INITIAL, 8)
    fine = oracle(p, current, INITIAL, 16)
    # Retain the observed unfavorable control. It cannot be the unit reference
    # at this bound and no RK4 step, neuron parameter or tolerance is repaired.
    assert np.max(np.abs(coarse-fine)) > 1e-8


@pytest.mark.parametrize('bits', [160, 320])
def test_short_active_excursion_native_array_and_exact_repeat(driver, bits, short_active_reference):
    current = np.full(501, 2000.)
    retained = []
    for _ in range(2):
        with context(driver, current, bits) as live:
            retained.append((live.advance(500), live.snapshot()))
    actual = retained[0][0]
    assert actual[:, 0].max() > 0
    np.testing.assert_allclose(actual, short_active_reference, atol=1e-8, rtol=0)
    np.testing.assert_array_equal(actual.view(np.uint64), retained[1][0].view(np.uint64))
    assert retained[0][1] == retained[1][1]


@pytest.mark.parametrize('bits', [160, 320])
def test_five_state_chunk_resume_and_full_internal_precision(driver, bits):
    current = np.zeros(4101)
    p = parameters(passive=True)
    with context(driver, current, bits, p=p) as whole:
        expected = whole.advance(4100)
        final = whole.snapshot()
    with context(driver, current, bits, p=p) as first:
        prefix = first.advance(4096)
        saved = json.loads(json.dumps(first.snapshot()))
    with context(driver, current, bits, p=p) as second:
        second.restore(saved)
        assert second.snapshot() == saved
        suffix = second.advance(4100)
        assert second.snapshot() == final
    np.testing.assert_array_equal(np.r_[prefix, suffix[1:]].view(np.uint64), expected.view(np.uint64))
    mp = pytest.importorskip('mpmath').mp.clone()
    mp.prec = 500
    assert any(decode(encoded, mp) != mp.mpf(float(value)) for encoded, value in
        zip(final['payload'].splitlines()[7:11], expected[-1, 1:], strict=True))


@pytest.mark.parametrize('bits', [160, 320])
@pytest.mark.parametrize('which', ['lower', 'upper'])
def test_gate_bounds_use_full_precision_before_export_not_rounded_double(driver, bits, which):
    mp = pytest.importorskip('mpmath').mp.clone()
    mp.prec = bits
    bound = mp.mpf(-1e-8 if which == 'lower' else 1.0+1e-8)
    direction = -1 if which == 'lower' else 1
    outside = bound+direction*mp.power(2, 12-bits)
    assert float(outside) == float(bound)  # binary64 alone would miss this violation.
    error = ctypes.create_string_buffer(4096)
    for value, rejected in ((bound, False), (outside, True)):
        state = [-65., .5, .5, .5, .5]
        state[2] = value
        strings = (ctypes.c_char_p*5)(*(exact(v, bits, mp).encode() for v in state))
        code = driver._lib.gbs_hh_gate_probe(bits, strings, error, len(error))
        assert bool(code) == rejected, error.value
        if rejected:
            assert b'gate bounds' in error.value


@pytest.mark.parametrize('bits', [160, 320])
def test_invalid_accepted_internal_gate_poisons_context(driver, bits):
    error = ctypes.create_string_buffer(4096)
    assert driver._lib.gbs_hh_accepted_gate_probe(bits, error, len(error)) == 0, error.value


@pytest.mark.parametrize('bits', [160, 320])
def test_invalid_dense_observed_gate_rejected_even_with_valid_accepted_endpoints(driver, bits):
    error = ctypes.create_string_buffer(4096)
    assert driver._lib.gbs_hh_dense_gate_probe(bits, error, len(error)) == 0, error.value


@pytest.mark.parametrize('bits', [160, 320])
def test_invalid_full_precision_restored_gate_is_rejected_before_mutation(driver, bits):
    current = np.zeros(31)
    with context(driver, current, bits) as original:
        original.advance(30)
        saved = original.snapshot()
    bad = copy.deepcopy(saved)
    mp = pytest.importorskip('mpmath').mp.clone()
    mp.prec = bits
    fields = bad['payload'].splitlines()
    fields[7] = exact(mp.mpf(1.0+1e-8)+mp.power(2, 12-bits), bits, mp)
    bad['payload'] = '\n'.join(fields)+'\n'
    bad = reseal(bad)
    with context(driver, current, bits) as fresh:
        before = fresh.snapshot()
        with pytest.raises(ValueError, match='gate bounds'):
            fresh.restore(bad)
        assert fresh.snapshot() == before
        fresh.restore(saved)
        assert fresh.snapshot() == saved


@pytest.mark.parametrize('field,value', [('capacitance_pF', 0.), ('leak_nS', 0.),
    ('sodium_nS', -1.), ('potassium_nS', -1.), ('m_current_nS', -1.), ('m_tau_max_ms', 0.),
    ('threshold_shift_mV', float('nan')), ('capacitance_pF', True)])
def test_invalid_original_parameter_fields_rejected(field, value):
    p = parameters()
    p[field] = value
    with pytest.raises(ValueError):
        parameters_array(p)


@pytest.mark.parametrize('gate', [1, 2, 3, 4])
@pytest.mark.parametrize('value', [-1.1e-8, 1.00000002])
def test_invalid_initial_gate_never_clipped_or_reinitialized(driver, gate, value):
    initial = INITIAL.copy()
    initial[gate] = value
    with pytest.raises(ValueError, match='gate bounds'):
        context(driver, np.zeros(3), initial=initial)


def test_original_explicit_initial_state_and_terminal_current_exclusion(driver):
    values = []
    for last in (0., 1e308):
        current = np.array([0., -50., 125., 0., last])
        initial = [-64., .04, .7, .3, .2]
        with context(driver, current, initial=initial) as live:
            values.append(live.advance(4))
    np.testing.assert_array_equal(values[0][0].view(np.uint64), np.array(initial).view(np.uint64))
    np.testing.assert_array_equal(values[0].view(np.uint64), values[1].view(np.uint64))


def test_checkpoint_cannot_change_parameters_build_policy_or_initialization(driver):
    current = np.zeros(31)
    with context(driver, current) as old:
        old.advance(30)
        saved = old.snapshot()
    changed = parameters()
    changed['m_current_nS'] += 1
    for kwargs in ({'p': changed}, {'policy': replace(Policy(), maximum_rhs=999999)},
                   {'initial': [-64., .05, .6, .2, .05]}, {'bits': 320}):
        with context(driver, current, **kwargs) as fresh:
            before = fresh.snapshot()
            with pytest.raises(ValueError, match='identity differs'):
                fresh.restore(saved)
            assert fresh.snapshot() == before
    bad = copy.deepcopy(saved)
    bad['configuration']['build_sha256'] = '0'*64
    with context(driver, current) as fresh, pytest.raises(ValueError, match='identity differs'):
        fresh.restore(reseal(bad))


def test_resource_failure_and_nonfinite_derivative_preserve_failure(driver):
    with context(driver, np.zeros(31), policy=replace(Policy(), maximum_rhs=1)) as live:
        with pytest.raises(ValueError, match='RHS evaluation budget'):
            live.advance(30)
        with pytest.raises(ValueError, match='failed context'):
            live.snapshot()
    with context(driver, np.zeros(3), initial=[1e308, .05, .6, .2, .05]) as live:
        with pytest.raises(ValueError, match='nonfinite'):
            live.advance(2)
        with pytest.raises(ValueError, match='failed context'):
            live.advance(2)


def test_independent_owned_workspaces_repeat_under_threads(driver):
    def run(bits):
        with context(driver, np.zeros(31), bits) as live:
            return live.advance(30), live.snapshot()
    serial = [run(160), run(320)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        concurrent = list(pool.map(run, (160, 320)))
    for (a, sa), (b, sb) in zip(serial, concurrent, strict=True):
        np.testing.assert_array_equal(a.view(np.uint64), b.view(np.uint64))
        assert sa == sb


def test_build_receipt_pins_reused_rhs_and_every_python_helper(driver):
    receipt = driver.provenance
    for relative in ('patchseq_mpfr_kernel.cpp', 'patchseq_gbs_guarded_core.hpp',
                     'patchseq_gbs_core.hpp', 'patchseq_gbs_driver.py', 'patchseq_mpfr_driver.py'):
        assert str((ROOT/'scripts'/relative).resolve()) in receipt['source_hashes']
    binary, reused = prepare_build(Path(receipt['binary_path']).parent)
    assert str(binary) == receipt['binary_path'] and reused == receipt
