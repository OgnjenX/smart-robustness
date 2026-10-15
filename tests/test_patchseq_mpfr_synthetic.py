"""Synthetic matrix, input timing and fail-closed checkpoint tests; no long runs."""

import copy
import fcntl
import json
import sys
from collections import Counter
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
import run_patchseq_mpfr_synthetic as runner
import verify_patchseq_mpfr_synthetic as verifier


def test_fixed_matrix_exact_keys_and_registration():
    rows = runner.case_table()
    registration = yaml.safe_load(runner.REG.read_text())
    assert registration['case_table'] == rows
    assert len(rows) == len({runner.label(r) for r in rows}) == 68
    assert Counter((r['samples'], r['profile']) for r in rows) == {
        (n, p): 17 for n in (151000, 330000) for p in ('passive', 'active')}
    for n in (151000, 330000):
        for profile in ('passive', 'active'):
            selected = [r for r in rows if (r['samples'], r['profile']) == (n, profile)]
            assert {(r['bits'], r['factor']) for r in selected if r['engine'] == 'boost'} == {
                (bits, factor) for bits in (53, 160, 320) for factor in (4, 8, 16)}
            assert {(r['bits'], r['factor'], r['repeat']) for r in selected if r['engine'] == 'mpfr'} == {
                *((bits, f, 0) for bits in (160, 320) for f in (4, 8, 16)),
                (160, 16, 1), (320, 16, 1)}


@pytest.mark.parametrize('samples', [151000, 330000])
@pytest.mark.parametrize('profile,amplitude', [('passive', -50.), ('active', 2000.)])
def test_full_length_inputs_warmup_edges_and_release(samples, profile, amplitude):
    p, current, initial, command, epoch = runner.synthetic_input(samples, profile)
    assert command.shape == (samples,)
    assert current.shape == (samples+50000,)
    assert np.all(current[:75000] == 0)
    assert np.all(current[75000:samples+25000] == amplitude)
    assert np.all(current[samples+25000:] == 0)
    np.testing.assert_array_equal(current[50000:], command)
    np.testing.assert_array_equal(initial, [-65., .05, .6, .2, .05])
    assert epoch == {'step_start_index': 25000, 'step_stop_index': samples-25000,
                     'pre_window': [15000, 20000]}
    assert len(p) == 8
    if profile == 'passive':
        assert [p[k] for k in ('sodium_nS', 'potassium_nS', 'm_current_nS')] == [0., 0., 0.]


@pytest.mark.parametrize('samples,profile', [(501, 'active'), (151000, 'other'), (330001, 'passive')])
def test_no_unregistered_input(samples, profile):
    with pytest.raises(ValueError):
        runner.synthetic_input(samples, profile)


def base():
    return {'pins': {'source': 'fixed'}, 'table': runner.case_table(), 'runtime': {'numpy': 'fixed'},
            'cases': {}, 'status': 'running-synthetic-only'}


def test_completed_and_failed_case_resume_preserves_artifacts(tmp_path):
    origin = base()
    state = copy.deepcopy(origin)
    evidence = tmp_path/'evidence'
    evidence.write_text('unchanged')
    for i, status in enumerate(('retained', 'failure-retained-no-replacement')):
        state['cases'][runner.label(state['table'][i])] = {
            'status': status, 'artifacts': {str(evidence): runner.digest(evidence)}}
    runner.validate_resume(state, origin)
    evidence.write_text('changed')
    with pytest.raises(ValueError, match='provenance changed'):
        runner.validate_resume(state, origin)


@pytest.mark.parametrize('defect', ['running', 'extra', 'unknown', 'runtime', 'table', 'pins'])
def test_resume_fails_closed(defect):
    origin, state = base(), base()
    key = runner.label(state['table'][0])
    if defect in ('running', 'unknown'):
        state['cases'][key] = {'status': defect, 'artifacts': {}}
    elif defect == 'extra':
        state['cases']['unregistered'] = {'status': 'retained', 'artifacts': {}}
    elif defect == 'runtime':
        state['runtime']['numpy'] = 'different'
    elif defect == 'table':
        state['table'][0]['factor'] = 1
    else:
        state['pins'] = {}
    with pytest.raises(ValueError):
        runner.validate_resume(state, origin)


def test_atomic_checkpoint_valid_json_no_nan_and_no_leftover(tmp_path):
    path = tmp_path/'checkpoint.json'
    runner.atomic_checkpoint(path, {'a': [1, 2], 'status': 'retained'})
    assert json.loads(path.read_text()) == {'a': [1, 2], 'status': 'retained'}
    assert not path.with_suffix('.tmp').exists()
    with pytest.raises(ValueError):
        runner.atomic_checkpoint(path, {'a': float('nan')})
    assert json.loads(path.read_text())['status'] == 'retained'


def test_no_execution_without_separate_seal(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'SEAL', tmp_path/'missing-seal.yaml')
    with pytest.raises(FileNotFoundError):
        runner.run()


def test_resource_budget_reserves_full_case_without_marking_failure(monkeypatch):
    monkeypatch.setattr(runner.time, 'monotonic', lambda: 8000.)
    assert runner.budget_allows_case(7900., 7200, 3600)
    assert not runner.budget_allows_case(1000., 7200, 3600)


def test_no_execution_without_confirmed_budget(tmp_path, monkeypatch):
    seal = tmp_path/'seal.yaml'
    seal.write_text('status: sealed-before-long-synthetic-execution\nmaximum_invocation_seconds: 28800\n')
    monkeypatch.setattr(runner, 'SEAL', seal)
    with pytest.raises(ValueError, match='execution budget'):
        runner.run()


def test_wrong_execution_seal_cannot_build(tmp_path, monkeypatch):
    seal = tmp_path/'seal.yaml'
    seal.write_text('status: preparation-only\n')
    monkeypatch.setattr(runner, 'SEAL', seal)
    with pytest.raises(ValueError, match='prospective execution seal'):
        runner.run()


def test_independent_table_has_identical_support():
    assert verifier.expected_keys() == {runner.label(r) for r in runner.case_table()}


@pytest.mark.parametrize('length', [151000, 330000])
@pytest.mark.parametrize('mode', ['passive', 'active'])
def test_independent_input_and_loss_reconstruction(length, mode):
    from patchseq_fit_objective import score_sweep
    from patchseq_response_features import characterize

    _, current, _, command, epoch = runner.synthetic_input(length, mode)
    independent, independent_epoch = verifier.inputs(length, mode)
    np.testing.assert_array_equal(current, independent)
    assert independent_epoch == epoch
    voltage = verifier.passive_voltage(length)[50000:]
    primary = characterize(voltage, command, 50000., epoch, 0.)
    secondary = verifier.reconstruct(voltage, command, 50000., independent_epoch, 0.)
    assert verifier.agree(primary, secondary)
    target = {'baseline_mean_mV': -65., 'early_step_mean_mV': -60., 'steady_step_mean_mV': -55.,
              'detected_event_count': 3, 'first_event_latency_s': .01, 'last_to_first_interval_ratio': 1.}
    assert verifier.synthetic_loss(primary) == pytest.approx(score_sweep(target, primary)['loss'], abs=1e-12)


@pytest.mark.parametrize('length', [151000, 330000])
def test_passive_closed_form_edges_and_release(length):
    voltage = verifier.passive_voltage(length)
    assert len(voltage) == length+50000
    assert voltage[75000] == -65.
    assert voltage[75001] == pytest.approx(-70.+5.*np.exp(-.002), abs=1e-13)
    stop = length+25000
    assert voltage[stop] == pytest.approx(-70., abs=1e-12)
    assert voltage[stop+1] > voltage[stop]
    assert voltage[-1] == pytest.approx(-65., abs=1e-12)


@pytest.mark.parametrize('defect', ['count', 'type', 'null', 'voltage', 'loss'])
def test_comparison_keeps_original_gates(defect):
    features = {'baseline_mean_mV': -65., 'early_step_mean_mV': -60., 'steady_step_mean_mV': -55.,
                'detected_event_count': 3, 'first_event_latency_s': .01, 'last_to_first_interval_ratio': 1.}
    other = copy.deepcopy(features)
    left, right = np.zeros((3, 5)), np.zeros((3, 5))
    assert verifier.compare(left, right, features, other)['passed']
    if defect == 'count':
        other['detected_event_count'] = 4
    elif defect == 'type':
        other['detected_event_count'] = 3.
    elif defect == 'null':
        other['first_event_latency_s'] = None
    elif defect == 'voltage':
        right[1, 0] = 1.01e-5
    else:
        other['steady_step_mean_mV'] += .001
    assert not verifier.compare(left, right, features, other)['passed']


def test_independent_verifier_refuses_nonterminal_or_changed_sources(tmp_path, monkeypatch):
    reg = tmp_path/'registration.yaml'
    reg.write_text(yaml.safe_dump({'output': str(tmp_path)}))
    monkeypatch.setattr(verifier, 'REG', reg)
    checkpoint = {'pins': {}, 'cases': {}, 'status': 'running'}
    (tmp_path/'checkpoint.json').write_text(json.dumps(checkpoint))
    with pytest.raises(ValueError, match='source/build evidence'):
        verifier.main()


def test_worker_receipts_and_no_overwrite_without_real_long_integration(tmp_path, monkeypatch):
    class SyntheticStub:
        provenance: ClassVar[dict] = {'unit-test-only': True}

        def __init__(self, directory):
            pass

        def integrate(self, p, current, rate, initial, bits, factor):
            output = np.tile(initial, (len(current), 1))
            output[:, 0] = verifier.passive_voltage(151000)
            return output

    monkeypatch.setattr(runner, 'MPFRDriver', SyntheticStub)
    spec = {'case': {'samples': 151000, 'profile': 'passive', 'engine': 'mpfr', 'bits': 160, 'factor': 4},
            'pins': {}, 'build': str(tmp_path/'not-built'), 'provenance': SyntheticStub.provenance,
            'output': str(tmp_path/'arrays.npz'), 'receipt': str(tmp_path/'receipt.json')}
    path = tmp_path/'spec.json'
    path.write_text(json.dumps(spec))
    runner.worker(path)
    receipt = json.loads(Path(spec['receipt']).read_text())
    assert receipt['shape'] == [201000, 5]
    assert receipt['arrays_sha256'] == runner.digest(spec['output'])
    assert receipt['uncompressed_bytes'] == 201000*5*8
    assert receipt['features']['detected_event_count'] == 0
    assert receipt['integration_seconds'] >= 0
    original = runner.digest(spec['output'])
    with pytest.raises(FileExistsError):
        runner.worker(path)
    assert runner.digest(spec['output']) == original
    # A second controller's worker cannot enter while an existing worker owns the lock.
    with (tmp_path/'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            runner.worker(path)


def test_worker_failure_does_not_fabricate_arrays(tmp_path, monkeypatch):
    class FailedStub:
        provenance: ClassVar[dict] = {'unit-test-only': True}

        def __init__(self, directory):
            pass

        def integrate(self, *args):
            raise ValueError('deliberately retained failure')

    monkeypatch.setattr(runner, 'MPFRDriver', FailedStub)
    spec = {'case': {'samples': 151000, 'profile': 'active', 'engine': 'mpfr', 'bits': 160, 'factor': 4},
            'pins': {}, 'build': str(tmp_path/'not-built'), 'provenance': FailedStub.provenance,
            'output': str(tmp_path/'arrays.npz'), 'receipt': str(tmp_path/'receipt.json')}
    path = tmp_path/'spec.json'
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match='retained failure'):
        runner.worker(path)
    assert not Path(spec['output']).exists()
    assert not Path(spec['receipt']).exists()
