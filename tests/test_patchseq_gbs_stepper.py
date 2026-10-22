"""Generic exact-boundary/restart tests only; no neuron/reference benchmark."""

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

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from patchseq_gbs_driver import LinearDriver, Policy, identity, prepare_build

ROOT = Path(__file__).parents[1]


@pytest.fixture(scope='module')
def driver(tmp_path_factory):
    try:
        return LinearDriver(tmp_path_factory.mktemp('gbs-linear'))
    except RuntimeError as exc:
        pytest.skip(str(exc))
    except subprocess.CalledProcessError as exc:
        pytest.fail(str(exc.stderr))


def context(driver, current, bits=160, policy=None, rate=40000., initial=(1., -2.),
            rates=(-.125, .025), gains=(.75, -.5)):
    return driver.context(current, rate, initial, rates, gains, bits, policy or Policy())


def decode(text, mp):
    digits, exponent = text.split('p')
    return mp.mpf(int(digits, 16))*mp.power(2, int(exponent))


def analytic(current, rate, initial, rates, gains):
    """Independent closed form, not another discretization or saved-array check."""
    mp = pytest.importorskip('mpmath').mp.clone()
    mp.prec = 500
    dt = mp.mpf(1000./rate)  # Same explicit binary64 time-unit input.
    state = list(map(mp.mpf, initial))
    result = [list(map(float, state))]
    start = 0
    for end in range(1, len(current)):
        elapsed = (end-start)*dt
        values = []
        for j in range(2):
            r, drive = mp.mpf(rates[j]), mp.mpf(gains[j])*mp.mpf(float(current[start]))
            values.append(state[j]+drive*elapsed if r == 0 else
                          state[j]*mp.exp(r*elapsed)+drive*mp.expm1(r*elapsed)/r)
        result.append(list(map(float, values)))
        if end+1 < len(current) and current[end] != current[start]:
            state, start = values, end
    return np.array(result)


def sealed_again(envelope):
    contents = {k: v for k, v in envelope.items() if k != 'sha256'}
    return {**contents, 'sha256': identity(contents)}


def test_local_stepper_has_only_declared_changes_to_pinned_upstream_math():
    local = (ROOT/'scripts/third_party/patchseq_bounded_gbs_dense.hpp').read_text()
    upstream = next((p/'boost/numeric/odeint/stepper/bulirsch_stoer_dense_out.hpp'
        for p in map(Path, ['/opt/homebrew/include', '/usr/include', '/usr/local/include'])
        if (p/'boost/numeric/odeint/stepper/bulirsch_stoer_dense_out.hpp').is_file()), None)
    if upstream is None:
        pytest.skip('optional pinned upstream Boost header not installed')
    assert hashlib.sha256(upstream.read_bytes()).hexdigest() == (
        'fe9852def53e255570f103adc6be7f1f3f038ffe2af1ea1b5f52e7e4bdd829cd')
    body = '\n'.join(local.split('\n')[3:])
    start = body.index('    // Local addition: exact end-bound,')
    stop = body.index('    /* performs the interpolation', start)
    body = body[:start]+body[stop:]
    body = body.replace('patchseq_bounded_gbs_dense_out', 'bulirsch_stoer_dense_out')
    body = body.replace('PATCHSEQ_BOUNDED_GBS_DENSE_OUT_HPP_INCLUDED',
                        'BOOST_NUMERIC_ODEINT_STEPPER_BULIRSCH_STOER_DENSE_OUT_HPP_INCLUDED')
    assert body == upstream.read_text()
    license_text = (ROOT/'scripts/third_party/BOOST_LICENSE_1_0.txt').read_text()
    assert license_text.startswith('Boost Software License - Version 1.0 - August 17th, 2003')
    assert 'Copyright 2011-2015 Mario Mulansky' in local


@pytest.mark.parametrize('bits', [160, 320])
@pytest.mark.parametrize('difficult', [0, 1])
def test_uncapped_trial_sequence_endpoints_and_dense_bits_identical_to_stock(driver, bits, difficult):
    counts = np.empty(3, dtype=np.uint64)
    error = ctypes.create_string_buffer(4096)
    code = driver._lib.gbs_linear_stock_probe(bits, difficult, counts, error, len(error))
    assert code == 0, error.value
    assert counts[0] == counts[1] > 0
    if difficult:
        assert counts[2] > 0


@pytest.mark.parametrize('bits', [160, 320])
def test_nonfinite_derivative_is_terminal_before_any_accepted_step(driver, bits):
    error = ctypes.create_string_buffer(4096)
    assert driver._lib.gbs_linear_nonfinite_probe(bits, error, len(error)) == 0, error.value


@pytest.mark.parametrize('bits', [160, 320])
@pytest.mark.parametrize('kind', ['zero', 'constant', 'edges', 'pulse', 'adjacent'])
def test_full_native_grid_against_independent_two_dimensional_closed_form(driver, bits, kind):
    current = np.zeros(193)
    if kind == 'constant':
        current[:] = .37
    if kind == 'edges':
        current[1:31], current[31:70], current[70:] = -.6, .8, .2
    if kind == 'pulse':
        current[79] = 1.5
    if kind == 'adjacent':
        current[11:16] = [.2, -.7, 0., 1., -.3]
    initial, rates, gains = (1., -2.), (-.125, .025), (.75, -.5)
    with context(driver, current, bits) as live:
        values = live.advance(len(current)-1)
        assert live.stats['rhs'] > 0
        assert live.stats['dense'] > 0
        assert live.stats['position'] == len(current)-1
        checkpoint = live.snapshot()
    expected = analytic(current, 40000., initial, rates, gains)
    np.testing.assert_allclose(values, expected, atol=2e-12, rtol=2e-13)
    assert np.array_equal(values[0].view(np.uint64), np.array(initial).view(np.uint64))
    mp = pytest.importorskip('mpmath').mp.clone()
    mp.prec = 500
    fields = checkpoint['payload'].splitlines()
    assert decode(fields[5], mp) == (len(current)-1)*mp.mpf(1000./40000.)


@pytest.mark.parametrize('bits', [160, 320])
@pytest.mark.parametrize('rate', [20001., 44100., 1000000.])
def test_zero_derivative_exact_and_affine_near_zero_error(driver, bits, rate):
    current = np.full(81, .375)
    with context(driver, current, bits, rate=rate, rates=(0., 0.), gains=(0., .5)) as live:
        output = live.advance(80)
    expected = analytic(current, rate, (1., -2.), (0., 0.), (0., .5))
    np.testing.assert_array_equal(output[:, 0], np.ones(81))
    np.testing.assert_allclose(output, expected, atol=1e-14, rtol=0)


@pytest.mark.parametrize('bits', [160, 320])
def test_last_command_not_integrated_and_one_sample_edges_force_exact_landings(driver, bits):
    first = np.array([0., -.5, 0., 1.25, -.25, 0., 0.])
    second = first.copy()
    second[-1] = 1e308
    retained = []
    for current in (first, second):
        with context(driver, current, bits, rates=(0., 0.), gains=(1., -1.)) as live:
            assert live.barriers == (0, 1, 2, 3, 4, 5, 6)
            retained.append((live.advance(6), live.stats))
    np.testing.assert_array_equal(retained[0][0].view(np.uint64), retained[1][0].view(np.uint64))
    assert retained[0][1] == retained[1][1]
    expected = analytic(first, 40000., (1., -2.), (0., 0.), (1., -1.))
    np.testing.assert_array_equal(retained[0][0], expected)


def test_signed_zero_changes_are_preserved_as_input_boundaries(driver):
    current = np.array([0., -0., -0., 0., 1., 0., 0.])
    with context(driver, current) as live:
        assert live.barriers == (0, 1, 3, 4, 5, 6)
        for stop in live.barriers:
            live.advance(stop)
        assert live.stats['position'] == 6


@pytest.mark.parametrize('bits', [160, 320])
def test_fixed_barriers_chunk_resume_full_precision_and_cumulative_counters(driver, bits):
    current = np.full(8203, .075)
    with context(driver, current, bits) as whole:
        expected = whole.advance(8202)
        expected_snapshot = whole.snapshot()
        expected_stats = whole.stats
    pieces = []
    live = context(driver, current, bits)
    try:
        assert live.barriers == (0, 4096, 8192, 8202)
        for stop in (4096, 8192, 8202):
            output = live.advance(stop)
            pieces.append(output if not pieces else output[1:])
            saved = json.loads(json.dumps(live.snapshot()))
            stats = live.stats
            live.close()
            live = context(driver, current, bits)
            live.restore(saved)
            assert live.stats == stats
            assert live.snapshot() == saved
        assert live.stats == expected_stats
        assert live.snapshot() == expected_snapshot
    finally:
        live.close()
    actual = np.concatenate(pieces)
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
    # Accepted endpoint has significant digits lost by a binary64 export.
    mp = pytest.importorskip('mpmath').mp.clone()
    mp.prec = 500
    assert any(decode(text, mp) != mp.mpf(float(value)) for text, value in
               zip(saved['payload'].splitlines()[6:8], actual[-1], strict=True))


def test_all_at_once_vs_multiple_current_barriers_identical(driver):
    current = np.zeros(105)
    current[30:32], current[32:70], current[70:] = .5, -.75, .25
    with context(driver, current) as a, context(driver, current) as b:
        full = a.advance(104)
        pieces = [b.advance(30), b.advance(32)[1:], b.advance(70)[1:], b.advance(104)[1:]]
        np.testing.assert_array_equal(full.view(np.uint64), np.concatenate(pieces).view(np.uint64))
        assert a.snapshot() == b.snapshot()


@pytest.mark.parametrize('stop', [-1, 1, 2, 3, 4095, 4097, True, 2.5])
def test_invalid_stop_rejected_before_mutation_not_a_silent_reset(driver, stop):
    with context(driver, np.zeros(4100)) as live:
        before = live.snapshot()
        with pytest.raises(ValueError, match='prescribed restart barrier'):
            live.advance(stop)
        assert live.snapshot() == before
        assert live.advance(0).shape == (1, 2)
        assert live.snapshot() == before


@pytest.mark.parametrize('field,value', [
    ('absolute', '0'), ('relative', '-1'), ('minimum_step_ms', 'nan'), ('initial_step_ms', 'inf'),
    ('absolute', .01), ('maximum_step_ms', '0.001'), ('minimum_step_ms', '1'),
    ('maximum_rhs', True), ('maximum_steps', -1), ('maximum_rejections', 2**63),
    ('relative', 'oops'), ('absolute', '1\x00'),
])
def test_invalid_policy_has_no_fallback(field, value):
    with pytest.raises(ValueError):
        replace(Policy(), **{field: value})


@pytest.mark.parametrize('field,value', [('bits', 53), ('bits', True), ('rate', True), ('rate', 20000.),
    ('rate', float('nan')), ('initial', (1., float('inf'))), ('gains', ('1', '2')),
    ('rates', (1.,)), ('current', [1.]), ('current', [0., float('nan')]), ('current', [True, False])])
def test_invalid_explicit_inputs_fail_without_run(driver, field, value):
    kwargs = {'current': np.zeros(10), field: value}
    with pytest.raises(ValueError):
        context(driver, **kwargs)


@pytest.mark.parametrize('field,value', [('current', [0, 2**53+1]),
    ('initial', [1, -(2**53+1)]), ('gains', np.array([1, 2**63], dtype=np.uint64))])
def test_integer_inputs_must_not_silently_round_before_the_binary64_lift(driver, field, value):
    with pytest.raises(ValueError):
        context(driver, **{'current': [0., 0.], field: value})


def test_extended_precision_inputs_not_silently_truncated_when_available(driver):
    if np.dtype(np.longdouble).itemsize <= 8:
        pytest.skip('platform long double is already binary64')
    with pytest.raises(ValueError):
        context(driver, np.zeros(2, dtype=np.longdouble))
    with pytest.raises(ValueError):
        context(driver, np.zeros(2), rate=np.longdouble(40000))


@pytest.mark.parametrize('change', ['current', 'initial', 'rates', 'gains', 'bits', 'policy', 'rate'])
def test_checkpoint_cannot_be_imported_under_different_inputs_or_policy(driver, change):
    current = np.zeros(33)
    with context(driver, current) as old:
        old.advance(32)
        checkpoint = old.snapshot()
    kwargs = {'current': current}
    kwargs[change] = {'current': np.ones(33), 'initial': (1.5, -2.), 'rates': (-.12, .025),
        'gains': (.8, -.5), 'bits': 320, 'policy': replace(Policy(), maximum_rhs=999999),
        'rate': 41000.}[change]
    with context(driver, **kwargs) as fresh:
        before = fresh.snapshot()
        with pytest.raises(ValueError, match='identity differs'):
            fresh.restore(checkpoint)
        assert fresh.snapshot() == before


@pytest.mark.parametrize('part', ['payload', 'sha256', 'extra', 'configuration', 'schema'])
def test_envelope_corruption_detected(driver, part):
    with context(driver, np.zeros(33)) as live:
        checkpoint = live.snapshot()
        if part == 'extra':
            checkpoint['extra'] = True
        elif part == 'configuration':
            checkpoint[part]['build_sha256'] = 'f'*64
        else:
            checkpoint[part] += 'corrupt'
        before = live.snapshot()
        with pytest.raises(ValueError):
            live.restore(checkpoint)
        assert live.snapshot() == before


@pytest.mark.parametrize('field,value', [(0, 'gbs-core-v2'), (1, '320'), (2, '3'), (3, '0'*64),
    (4, '1'), (4, '033'), (5, 'NaN'), (5, '0p0'), (6, 'inf'), (8, '1000001'),
    (9, '100001'), (10, '10001'), (11, '34')])
def test_core_checkpoint_structure_exact_time_and_counters_validated_atomically(driver, field, value):
    with context(driver, np.zeros(34)) as old:
        old.advance(33)
        saved = old.snapshot()
    bad = copy.deepcopy(saved)
    fields = bad['payload'].splitlines()
    fields[field] = value
    bad['payload'] = '\n'.join(fields)+'\n'
    # Deliberately recompute checksum to reach the C++ structural validation.
    # This also makes explicit that the checksum is not an authentication key.
    bad = sealed_again(bad)
    with context(driver, np.zeros(34)) as fresh:
        before = fresh.snapshot()
        with pytest.raises(ValueError):
            fresh.restore(bad)
        assert fresh.snapshot() == before
        fresh.restore(saved)
        assert fresh.snapshot() == saved


def test_restore_requires_fresh_context_and_closed_context_cannot_be_reused(driver):
    live = context(driver, np.zeros(34))
    saved = live.snapshot()
    live.advance(33)
    before = live.snapshot()
    with pytest.raises(ValueError, match='fresh context'):
        live.restore(saved)
    assert live.snapshot() == before
    live.close()
    live.close()
    for call in (lambda: live.advance(33), live.snapshot, lambda: live.restore(saved), lambda: live.stats):
        with pytest.raises(ValueError, match='closed'):
            call()


@pytest.mark.parametrize('kind', ['rhs', 'steps', 'rejections', 'minimum', 'export'])
def test_bounded_failure_poisoning_and_no_checkpoint_or_replay(driver, kind):
    policy = Policy()
    kwargs = {}
    current = np.zeros(201)
    match = {'rhs': 'RHS evaluation budget', 'steps': 'accepted-step budget',
             'rejections': 'rejected-trial budget', 'minimum': 'minimum step', 'export': 'nonfinite'}[kind]
    if kind == 'rhs':
        policy = replace(policy, maximum_rhs=1)
    elif kind == 'steps':
        policy = replace(policy, maximum_steps=0)
    elif kind == 'rejections':
        policy = replace(policy, absolute='1e-40', relative='1e-40', initial_step_ms='1',
                         maximum_step_ms='1', maximum_rejections=0)
        kwargs['rates'] = (-1000., -500.)
    elif kind == 'minimum':
        current = np.zeros(2)
        policy = replace(policy, initial_step_ms='0.1', minimum_step_ms='0.1')
    else:
        current = np.full(5, 1e308)
        kwargs.update(initial=(1e308, 1.), rates=(0., 0.), gains=(1e308, 0.))
    with context(driver, current, policy=policy, **kwargs) as live:
        saved = live.snapshot()
        with pytest.raises(ValueError, match=match):
            live.advance(len(current)-1)
        assert live.stats['rhs'] <= policy.maximum_rhs
        assert live.stats['accepted'] <= policy.maximum_steps
        for call in (lambda: live.advance(len(current)-1), live.snapshot, lambda: live.restore(saved)):
            with pytest.raises(ValueError, match='failed context'):
                call()


def test_rhs_budget_cannot_be_reset_by_checkpoint_resume(driver):
    current = np.zeros(8200)
    # Unit-test fault injection: allow exactly the known prefix's work, so the
    # next segment must fail identically with and without a process-style reset.
    # This is not selection of a policy for a neuron experiment.
    with context(driver, current) as prefix:
        prefix.advance(4096)
        prefix_rhs = prefix.stats['rhs']
    policy = replace(Policy(), maximum_rhs=prefix_rhs)
    with context(driver, current, policy=policy) as a:
        a.advance(4096)
        saved = a.snapshot()
        with pytest.raises(ValueError, match='budget'):
            a.advance(8199)
        terminal = a.stats
    with context(driver, current, policy=policy) as b:
        b.restore(saved)
        with pytest.raises(ValueError, match='budget'):
            b.advance(8199)
        assert b.stats == terminal


def test_context_owns_input_arrays_and_independent_threads_agree(driver):
    current = np.zeros(71)
    with context(driver, current) as live:
        current[:] = 999
        baseline = live.advance(70)
    def work(bits):
        with context(driver, np.zeros(71), bits) as live:
            return live.advance(70), live.snapshot()
    serial = [work(bits) for bits in (160, 320)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        parallel = list(pool.map(work, (160, 320)))
    for (first, checkpoint), (second, saved) in zip(serial, parallel, strict=True):
        np.testing.assert_array_equal(first.view(np.uint64), second.view(np.uint64))
        np.testing.assert_array_equal(first.view(np.uint64), baseline.view(np.uint64))
        assert checkpoint == saved


def test_public_provenance_and_environment_are_detached_copies(driver):
    environment, provenance = driver.environment, driver.provenance
    environment['precision160'] = 53
    provenance['source_hashes'].clear()
    assert driver.environment['precision160'] == 160
    assert driver.provenance['source_hashes']
    driver.verify()


def test_source_change_during_call_is_detected_before_output_return_and_poisons_context(driver, monkeypatch):
    with context(driver, np.zeros(4)) as live:
        original = live._call
        def changed(name, *args):
            original(name, *args)
            if name == 'advance':
                monkeypatch.setattr(driver, 'verify', lambda: (_ for _ in ()).throw(
                    ValueError('GBS source changed during operation')))
        monkeypatch.setattr(live, '_call', changed)
        with pytest.raises(ValueError, match='source changed'):
            live.advance(3)
        with pytest.raises(ValueError, match='failed context'):
            live.snapshot()


def test_build_reuse_and_partial_or_altered_receipts_cannot_overwrite(driver, tmp_path):
    provenance = driver.provenance
    binary, receipt = prepare_build(Path(provenance['binary_path']).parent)
    assert str(binary) == provenance['binary_path']
    assert receipt == provenance
    partial = tmp_path/'partial'
    partial.mkdir()
    (partial/'linear.so').write_bytes(b'orphan')
    with pytest.raises(ValueError, match='partial'):
        prepare_build(partial)
    assert (partial/'linear.so').read_bytes() == b'orphan'
    locked = tmp_path/'locked'
    locked.mkdir()
    (locked/'build.lock').write_text('retained-lock')
    with pytest.raises(ValueError, match='lock'):
        prepare_build(locked)
    assert (locked/'build.lock').read_text() == 'retained-lock'
    altered = tmp_path/'altered'
    altered.mkdir()
    (altered/'linear.so').write_bytes(Path(provenance['binary_path']).read_bytes())
    receipt['binary_path'] = str((altered/'linear.so').resolve())
    receipt['binary_sha256'] = '0'*64
    (altered/'build.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='provenance differs'):
        prepare_build(altered)
