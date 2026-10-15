"""Synthetic equivalence and lossless-state controls, not biological validation."""

import copy
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from patchseq_precision_driver import FIELDS, PrecisionDriver
from patchseq_precision_successor_driver import SuccessorDriver, signature


@pytest.fixture(scope='module')
def drivers(tmp_path_factory):
    root = tmp_path_factory.mktemp('successor-synthetic')
    try:
        return SuccessorDriver(root/'new'), PrecisionDriver(root/'old')
    except RuntimeError as exc:
        pytest.skip(str(exc))


def specification():
    p = dict(zip(FIELDS, [100., 10., -65., 2000., 500., 10., -60., 1000.], strict=True))
    current = np.r_[np.zeros(7), np.full(13, -50.), np.full(17, 125.), np.zeros(14)]
    return p, current, 50000., [-65., .05, .6, .2, .05]


def assert_bits(left, right):
    np.testing.assert_array_equal(np.asarray(left, dtype=np.float64).view(np.uint64),
                                  np.asarray(right, dtype=np.float64).view(np.uint64))


@pytest.mark.parametrize('bits', [53, 160, 320])
@pytest.mark.parametrize('factor', [1, 4, 8, 16])
def test_monolithic_and_chunked_exactly_match_sealed_predecessor(drivers, bits, factor):
    new, old = drivers
    p, current, rate, initial = specification()
    expected = old.integrate(p, current, rate, initial, bits, factor)
    with new.session(p, current, rate, initial, bits, factor) as context:
        monolithic = context.advance(len(current)-1)
    with new.session(p, current, rate, initial, bits, factor) as context:
        pieces = [context.advance(7)]
        for stop in (8, 19, 20, 36, 37, len(current)-1):
            pieces.append(context.advance(stop)[1:])
        chunked = np.concatenate(pieces)
    assert_bits(monolithic, expected)
    assert_bits(chunked, expected)


@pytest.mark.parametrize('bits', [53, 160, 320])
def test_serialized_state_restores_exact_internal_digits_and_trajectory(drivers, bits):
    new, old = drivers
    p, current, rate, initial = specification()
    expected = old.integrate(p, current, rate, initial, bits, 4)
    with new.session(p, current, rate, initial, bits, 4) as context:
        prefix = context.advance(19)
        snapshot = context.snapshot()
        tail = context.advance(len(current)-1)
        final = context.snapshot()
    snapshot = json.loads(json.dumps(snapshot))
    with new.session(p, current, rate, initial, bits, 4) as resumed:
        resumed.restore(snapshot)
        assert resumed.snapshot() == snapshot  # Not a binary64 checkpoint.
        actual = resumed.advance(len(current)-1)
        assert resumed.snapshot() == final
    assert_bits(actual, tail)
    assert_bits(np.concatenate([prefix, actual[1:]]), expected)


@pytest.mark.parametrize('mutation', ['hash', 'current', 'parameter', 'initial', 'precision',
                                     'factor', 'position', 'valid_wrong_position', 'boolean', 'state', 'tokens', 'build'])
def test_corrupted_or_different_checkpoint_cannot_resume(drivers, mutation):
    new, _ = drivers
    p, current, rate, initial = specification()
    with new.session(p, current, rate, initial, 160, 4) as context:
        context.advance(7)
        checkpoint = context.snapshot()
    changed = copy.deepcopy(checkpoint)
    if mutation == 'hash':
        changed['sha256'] = 'changed'
    elif mutation == 'position':
        changed['cursor'] = len(current)
    elif mutation == 'valid_wrong_position':
        changed['cursor'] = 6
    elif mutation == 'boolean':
        changed['cursor'] = True
    elif mutation == 'state':
        changed['state'] = 'not a state'
    elif mutation == 'tokens':
        changed['state'] += 'extra'
    elif mutation == 'current':
        current[3] += 1
    elif mutation == 'parameter':
        p['capacitance_pF'] += 1
    elif mutation == 'initial':
        initial[0] += 1
    else:
        field = {'precision': 'bits', 'factor': 'factor', 'build': 'build_identity'}[mutation]
        changed['configuration'][field] = 320 if mutation == 'precision' else 8 if mutation == 'factor' else 'changed'
    if mutation != 'hash':
        changed['sha256'] = signature({k: v for k, v in changed.items() if k != 'sha256'})
    with new.session(p, current, rate, initial, 160, 4) as context:
        before = context.snapshot()
        with pytest.raises(ValueError):
            context.restore(changed)
        assert context.cursor == 0 and context.snapshot() == before


def test_snapshot_does_not_alias_context_configuration(drivers):
    new, _ = drivers
    p, current, rate, initial = specification()
    with new.session(p, current, rate, initial, 160, 4) as context:
        first = context.snapshot()
        bad = context.snapshot()
        bad['configuration']['parameters'][0] += 1
        assert context.snapshot() == first


def test_independent_contexts_can_be_sharded_without_reordering_samples(drivers):
    new, _ = drivers
    p, current, rate, initial = specification()

    def one(bits):
        with new.session(p, current, rate, initial, bits, 4) as context:
            return context.advance(len(current)-1), context.snapshot()

    serial = [one(bits) for bits in (53, 160, 320)]
    with ThreadPoolExecutor(max_workers=3) as pool:
        parallel = list(pool.map(one, (53, 160, 320)))
    for (a, x), (b, y) in zip(serial, parallel, strict=True):
        assert_bits(a, b)
        assert x == y


def test_input_copy_zero_length_boundary_and_lifetime_controls(drivers):
    new, old = drivers
    p, current, rate, initial = specification()
    expected = old.integrate(p, current, rate, initial, 53, 1)
    with new.session(p, current, rate, initial, 53, 1) as context:
        current[:] = 999.
        assert_bits(context.advance(0)[0], initial)
        assert_bits(context.advance(len(current)-1), expected)
        with pytest.raises(ValueError, match='fresh'):
            context.restore(context.snapshot())
        with pytest.raises(ValueError):
            context.advance(len(current))
    with pytest.raises(RuntimeError):
        context.advance(0)


def test_failed_integration_poisoned_not_silently_retried(drivers):
    new, _ = drivers
    p, current, rate, initial = specification()
    current[0] = 1e12
    with new.session(p, current, rate, initial, 53, 1) as context:
        with pytest.raises(ValueError, match='poisoned'):
            context.advance(1)
        with pytest.raises(RuntimeError):
            context.snapshot()
        with pytest.raises(RuntimeError):
            context.advance(1)


@pytest.mark.parametrize('bits', [160, 320])
def test_internal_checkpoint_is_not_a_binary64_export(drivers, bits):
    new, _ = drivers
    p, current, rate, initial = specification()
    with new.session(p, current, rate, initial, bits, 4) as context:
        exported = context.advance(19)[-1]
        precise = context.snapshot()['state'].splitlines()[1:]
    with new.session(p, current, rate, exported, bits, 4) as rounded:
        assert rounded.snapshot()['state'].splitlines()[1:] != precise


@pytest.mark.parametrize('bits', [160, 320])
def test_chunked_synthetic_excursion_against_independent_oracle(drivers, bits):
    from test_patchseq_precision_prototype import oracle

    new, _ = drivers
    p, _, rate, initial = specification()
    current = np.full(501, 2000.)
    with new.session(p, current, rate, initial, bits, 1) as context:
        before = context.advance(249)
        snapshot = context.snapshot()
    with new.session(p, current, rate, initial, bits, 1) as resumed:
        resumed.restore(json.loads(json.dumps(snapshot)))
        states = np.concatenate([before, resumed.advance(500)[1:]])
    assert states[:, 0].max() > 0
    expected = oracle([p[k] for k in FIELDS], current, initial, 1)
    assert_bits(states, expected)


def test_retained_build_reloads_and_restores_in_new_driver(drivers):
    new, _ = drivers
    p, current, rate, initial = specification()
    with new.session(p, current, rate, initial, 160, 4) as context:
        context.advance(19)
        saved = context.snapshot()
        expected = context.advance(len(current)-1)
    other = SuccessorDriver(Path(new.provenance['binary_path']).parent)
    assert other.provenance == new.provenance
    with other.session(p, current, rate, initial, 160, 4) as resumed:
        resumed.restore(saved)
        assert_bits(resumed.advance(len(current)-1), expected)
