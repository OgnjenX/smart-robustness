"""Pilot orchestration only: fabricated arrays, no HH solver/build/experiment.

The fake engines below deliberately do not implement an ODE. Passing these
checks establishes software admission and retention behavior, not accuracy.
"""

from __future__ import annotations

import copy
import fcntl
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
import run_patchseq_gbs_pilot as runner


class FabricatedContext:
    def __init__(self, current, initial):
        self.current, self.initial = current, initial
        self.position = 0

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        pass

    def advance(self, stop):
        data = np.tile(self.initial, (stop-self.position+1, 1))
        # An index-dependent, tiny gate ramp makes concatenation/offset bugs
        # visible, while voltage is simply fabricated constant -65 mV.
        data[:, 1] += np.arange(self.position, stop+1)*1e-10
        self.position = stop
        return data

    @property
    def stats(self):
        return {'position': self.position, 'rhs': self.position*2,
                'accepted': self.position, 'rejected': 0, 'dense': 0}

    def snapshot(self):
        return {'fabricated_not_a_solver_checkpoint': True, 'position': self.position}

    def restore(self, document):
        assert document['fabricated_not_a_solver_checkpoint'] is True
        self.position = document['position']


class FabricatedGBS:
    def context(self, parameters, current, rate, initial, bits, policy):
        return FabricatedContext(current, initial)


class FabricatedReference:
    def integrate(self, parameters, current, rate, initial, bits, factor, **limits):
        ctx = FabricatedContext(current, initial)
        return SimpleNamespace(states=ctx.advance(len(current)-1),
            final_exact=('fabricated-not-a-reference',)*5,
            rhs_evaluations=13*factor*(len(current)-1), accepted_substeps=factor*(len(current)-1),
            maximum_embedded_estimate=np.zeros(5))


@pytest.fixture
def auth(tmp_path, monkeypatch):
    pin = tmp_path/'pinned-source'
    pin.write_text('fabricated controller test provenance, not a production seal')
    data = {'reg': yaml.safe_load(runner.REG.read_text()),
        'prep': {'builds': {'gbs': {'fake': 'primary'}, 'rkf78': {'fake': 'independent'}},
                 'runtime': {'fake': 'test-only'}},
        'pins': {str(pin): runner.file_sha(pin)}, 'registration_sha256': 'a'*64,
        'seal_sha256': 'b'*64, 'preparation_sha256': 'c'*64, 'output': tmp_path/'output'}
    monkeypatch.setattr(runner, 'load_authorization', lambda: data)
    monkeypatch.setattr(runner, 'open_engine', lambda auth, row:
                        FabricatedGBS() if row['engine'] == 'gbs' else FabricatedReference())
    # The child is invoked inline for these tests; production uses subprocess.
    monkeypatch.setattr(runner.os, 'getppid', os.getpid)
    for key, value in runner.WORKER_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(runner, 'run_child', lambda spec, logfile, timeout: runner.worker(spec))
    return data


def first_entry(state, status):
    return next((key, row) for key, row in state['cases'].items() if row['status'] == status)


@pytest.mark.parametrize('entry', ['controller', 'worker'])
def test_unsealed_entry_refuses_before_output_build_or_engine(tmp_path, monkeypatch, entry):
    monkeypatch.setattr(runner, 'SEAL', tmp_path/'absent-seal.yaml')
    monkeypatch.setattr(runner, 'PREPARATION', tmp_path/'no-builds')
    monkeypatch.setattr(runner, 'read_preparation', lambda: pytest.fail('build preparation reached'))
    with pytest.raises(FileNotFoundError):
        runner.run() if entry == 'controller' else runner.worker(tmp_path/'no-spec')
    assert not (tmp_path/'no-builds').exists()


def test_missing_array_auditor_blocks_build_preparation(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'PREPARATION', tmp_path/'no-builds')
    monkeypatch.setattr(runner, 'HHDriver', lambda *a: pytest.fail('build reached'))
    # A controlled source tree with no array auditor must fail even when a
    # future real auditor exists. This test never manufactures such an auditor.
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    for name in (runner.REGISTRATION, 'scripts/run_patchseq_gbs_pilot.py'):
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fabricated source pin')
    with pytest.raises(FileNotFoundError, match='verify_patchseq_gbs_pilot'):
        runner.prepare_builds()
    assert not (tmp_path/'no-builds').exists()


@pytest.fixture
def approval_files(tmp_path, monkeypatch):
    """Fabricated seals exist ONLY inside this isolated test directory."""
    original_reg = runner.REG
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, 'REG', original_reg)
    monkeypatch.setattr(runner, 'SEAL', tmp_path/'test-only-seal.yaml')
    monkeypatch.setattr(runner, 'PREPARATION', tmp_path/'preparation')
    runner.PREPARATION.mkdir()
    prep_path = runner.PREPARATION/'preparation.json'
    runner.write_json(prep_path, {'fabricated': 'not production preparation'})
    source = tmp_path/'source'
    source.write_text('test provenance')
    pins = {str(source): runner.file_sha(source)}
    prep = {'source_hashes': pins, 'pins': pins}
    reg = yaml.safe_load(original_reg.read_text())
    seal = {'schema_version': 1, 'status': 'sealed-before-short-synthetic-pilot',
        'human_execution_approved': True, 'human_approval_record': 'TEST ONLY, no human approval of pilot execution',
        'registration_sha256': runner.file_sha(original_reg),
        'preparation_path': str(prep_path.relative_to(tmp_path)), 'preparation_sha256': runner.file_sha(prep_path),
        'resources': copy.deepcopy(reg['resources']), 'allowed_operations': ['short-synthetic-pilot', 'independent-pilot-audit'],
        'source_hashes': pins}
    monkeypatch.setattr(runner, 'read_preparation', lambda: prep)
    return seal, prep


def test_exact_isolated_approval_scope_binds_preparation_and_every_source(approval_files):
    seal, prep = approval_files
    runner.SEAL.write_text(yaml.safe_dump(seal))
    approved = runner.load_authorization()
    assert approved['prep'] == prep
    assert approved['pins'] == runner.merge_pins(prep['pins'], {
        str(runner.SEAL): runner.file_sha(runner.SEAL),
        str(runner.PREPARATION/'preparation.json'): runner.file_sha(runner.PREPARATION/'preparation.json')})
    assert not approved['output'].exists()


@pytest.mark.parametrize('defect', ['extra', 'missing', 'schema_bool', 'human_false', 'human_int',
    'approval_empty', 'approval_not_string', 'status', 'registration_hash', 'preparation_hash',
    'preparation_path', 'source_support', 'budget', 'boolean_resource', 'operations'])
def test_invalid_execution_decision_cannot_reach_prepared_drivers(approval_files, defect):
    seal, _prep = approval_files
    if defect == 'extra':
        seal['extra'] = True
    elif defect == 'missing':
        seal.pop('human_approval_record')
    elif defect == 'schema_bool':
        seal['schema_version'] = True
    elif defect == 'human_false':
        seal['human_execution_approved'] = False
    elif defect == 'human_int':
        seal['human_execution_approved'] = 1
    elif defect == 'approval_empty':
        seal['human_approval_record'] = '  '
    elif defect == 'approval_not_string':
        seal['human_approval_record'] = ['approval']
    elif defect == 'status':
        seal['status'] = 'prepared-not-approved'
    elif defect == 'registration_hash':
        seal['registration_sha256'] = '0'*64
    elif defect == 'preparation_hash':
        seal['preparation_sha256'] = '0'*64
    elif defect == 'preparation_path':
        seal['preparation_path'] = 'other/preparation.json'
    elif defect == 'source_support':
        seal['source_hashes'] = {}
    elif defect == 'budget':
        seal['resources']['maximum_case_seconds'] += 1
    elif defect == 'boolean_resource':
        seal['resources']['concurrent_workers'] = True
    else:
        seal['allowed_operations'].append('biological-fitting')
    runner.SEAL.write_text(yaml.safe_dump(seal))
    with pytest.raises(ValueError):
        runner.load_authorization()
    assert not (runner.ROOT/'results').exists()


@pytest.fixture
def prepared_files(tmp_path, monkeypatch):
    """Build receipt validation with inert files; no compiler or library load."""
    monkeypatch.setattr(runner, 'PREPARATION', tmp_path/'preparation')
    runner.PREPARATION.mkdir()
    source = tmp_path/'source'
    source.write_text('fabricated source')
    sources = {str(source): runner.file_sha(source)}
    runtime = {'file_hashes': sources, 'fake': 'test-runtime'}
    monkeypatch.setattr(runner, 'repository_pins', lambda: sources)
    monkeypatch.setattr(runner, 'runtime_fingerprint', lambda: runtime)
    builds, pins = {}, dict(sources)
    for engine, name in (('gbs', 'hh.so'), ('rkf78', 'reference.so')):
        folder = runner.PREPARATION/(engine+'-build')
        folder.mkdir()
        binary = folder/name
        binary.write_bytes(b'not-an-executable-library')
        builds[engine] = {'source_hashes': sources, 'binary_path': str(binary.resolve()),
                          'binary_sha256': runner.file_sha(binary)}
        runner.write_json(folder/'build.json', builds[engine])
        pins.update({str(binary): runner.file_sha(binary), str(folder/'build.json'): runner.file_sha(folder/'build.json')})
    doc = {'status': 'built-and-pinned-no-pilot-trajectories', 'source_hashes': sources,
        'runtime': runtime, 'pins': pins, 'builds': builds,
        'arithmetic_probes': copy.deepcopy(runner.EXPECTED_ARITHMETIC_PROBES), 'preparation_seconds': 0.,
        'biological_access': False, 'pilot_executed': False, 'fitting_restart': False}
    runner.write_json(runner.PREPARATION/'preparation.json', doc)
    return doc


def test_build_preparation_requires_exact_current_receipts_and_product_pins(prepared_files):
    assert runner.read_preparation() == prepared_files


@pytest.mark.parametrize('entry', ['prepare', 'read'])
def test_linked_preparation_is_not_followed(tmp_path, monkeypatch, entry):
    destination = tmp_path/'external'
    destination.mkdir()
    link = tmp_path/'linked-preparation'
    link.symlink_to(destination, target_is_directory=True)
    monkeypatch.setattr(runner, 'PREPARATION', link)
    with pytest.raises(ValueError, match='linked pilot build preparation'):
        runner.prepare_builds() if entry == 'prepare' else runner.read_preparation()
    assert list(destination.iterdir()) == []


@pytest.mark.parametrize('defect', ['extra', 'missing_route', 'extra_route', 'runtime', 'sources',
    'pin_support', 'binary_path', 'binary_bytes', 'build_receipt', 'execution_flag', 'invalid_timing', 'arithmetic_probe'])
def test_changed_preparation_cannot_authorize_worker(prepared_files, defect):
    doc = prepared_files
    if defect == 'extra':
        doc['extra'] = True
    elif defect == 'missing_route':
        doc['builds'].pop('rkf78')
    elif defect == 'extra_route':
        doc['builds']['other'] = doc['builds']['gbs']
    elif defect == 'runtime':
        doc['runtime'] = {'file_hashes': {}, 'fake': 'different'}
    elif defect == 'sources':
        doc['source_hashes'] = {}
    elif defect == 'pin_support':
        doc['pins'].pop(doc['builds']['gbs']['binary_path'])
    elif defect == 'binary_path':
        doc['builds']['gbs']['binary_path'] += '.different'
    elif defect == 'binary_bytes':
        Path(doc['builds']['gbs']['binary_path']).write_bytes(b'changed binary')
    elif defect == 'build_receipt':
        (runner.PREPARATION/'gbs-build/build.json').write_text('{}')
    elif defect == 'execution_flag':
        doc['pilot_executed'] = True
    elif defect == 'arithmetic_probe':
        doc['arithmetic_probes']['gbs']['precision160'] = 159
    else:
        doc['preparation_seconds'] = True
    runner.write_json(runner.PREPARATION/'preparation.json', doc, atomic=True)
    with pytest.raises((ValueError, FileNotFoundError)):
        runner.read_preparation()


def test_exclusive_atomic_json_and_duplicate_nonfinite_link_guards(tmp_path):
    path = tmp_path/'checkpoint.json'
    runner.write_json(path, {'v': 1}, atomic=True)
    runner.write_json(path, {'v': 2}, atomic=True)
    assert runner.read_json(path) == {'v': 2}
    with pytest.raises(ValueError, match='orphan'):
        runner.write_json(path, {})
    path.with_suffix('.tmp').write_text('interrupted write')
    with pytest.raises(ValueError, match='orphan'):
        runner.write_json(path, {}, atomic=True)
    assert runner.read_json(path) == {'v': 2}
    for raw in ('{"v":1,"v":2}', '{"v":NaN}', '{"v":1e999}'):
        path.write_text(raw)
        with pytest.raises(ValueError):
            runner.read_json(path)
    with pytest.raises(ValueError):
        runner.write_json(tmp_path/'nonfinite.json', {'v': float('inf')})
    assert not (tmp_path/'nonfinite.json').exists()
    link = tmp_path/'linked.json'
    link.symlink_to(path)
    with pytest.raises(ValueError, match='linked'):
        runner.read_json(link)


def test_clean_boundary_resume_retains_full_ledger_and_cumulative_budget(auth):
    first = runner.run(stop_after_cases=1)
    assert len(first['cases']) == 104 and first['status'] == 'clean-boundary'
    assert sum(e['status'] == 'retained' for e in first['cases'].values()) == 1
    key, completed = first_entry(first, 'retained')
    retained = copy.deepcopy(completed)
    charged = first['controller']['cumulative_charged_seconds']
    assert charged >= 60.
    runner.validate_state(first, auth, resume=True)
    second = runner.run(stop_after_cases=1)
    assert second['cases'][key] == retained
    assert second['controller']['invocation'] == 2
    assert second['controller']['cumulative_charged_seconds'] >= charged+60.
    assert len(second['cases']) == 104
    assert sum(e['status'] == 'retained' for e in second['cases'].values()) == 2
    runner.validate_state(second, auth, resume=True)
    assert second['scientific_goal_complete'] is False


@pytest.mark.parametrize('defect', ['missing_case', 'extra_case', 'altered_case', 'bool_case',
    'config', 'spent_reset', 'controller_state', 'controller_pid', 'invocation', 'nonfinite_time',
    'false_terminal', 'false_failure', 'unknown_status', 'unknown_file', 'orphan_pending',
    'changed_array', 'missing_log', 'linked_lock', 'nested_output'])
def test_ledger_tampering_or_orphans_block_resume(auth, defect):
    state = runner.run(stop_after_cases=1)
    key, done = first_entry(state, 'retained')
    pending, _unused = first_entry(state, 'pending')
    files = runner.case_files(auth['output'], key)
    if defect == 'missing_case':
        state['cases'].pop(pending)
    elif defect == 'extra_case':
        state['cases']['unexpected'] = copy.deepcopy(done)
    elif defect == 'altered_case':
        state['cases'][pending]['case']['bits'] = 640
    elif defect == 'bool_case':
        state['cases'][pending]['case']['repeat'] = False
    elif defect == 'config':
        state['comparison_table_sha256'] = '0'*64
    elif defect == 'spent_reset':
        state['controller']['cumulative_charged_seconds'] = 0.
    elif defect == 'controller_state':
        state['controller']['state'] = 'running'
    elif defect == 'controller_pid':
        state['controller']['pid'] = True
    elif defect == 'invocation':
        state['controller']['invocation'] = 0
    elif defect == 'nonfinite_time':
        state['controller']['measured_invocation_seconds'] = float('nan')
    elif defect == 'false_terminal':
        state['status'] = runner.TERMINAL
    elif defect == 'false_failure':
        state['status'] = runner.STOPPED
    elif defect == 'unknown_status':
        state['status'] = 'success'
    elif defect == 'unknown_file':
        (auth['output']/'unlisted.json').write_text('{}')
    elif defect == 'orphan_pending':
        runner.case_files(auth['output'], pending)['array'].write_bytes(b'orphan')
    elif defect == 'changed_array':
        files['array'].write_bytes(b'changed')
    elif defect == 'missing_log':
        files['log'].unlink()
    elif defect == 'linked_lock':
        (auth['output']/'audit.lock').symlink_to(files['log'])
    else:
        (auth['output']/'nested').mkdir()
    with pytest.raises((ValueError, FileNotFoundError)):
        runner.validate_state(state, auth, resume=True)


@pytest.mark.parametrize('which', ['runner.lock', 'worker.lock'])
def test_existing_live_lock_blocks_duplicate_controller_without_checkpoint(auth, which):
    auth['output'].mkdir()
    with (auth['output']/which).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            runner.run()
    assert not (auth['output']/'checkpoint.json').exists()


def test_source_change_blocks_worker_and_retains_failure(auth, monkeypatch):
    def changed_source(spec, log, timeout):
        Path(next(iter(auth['pins']))).write_text('changed source')
        runner.worker(spec)
    monkeypatch.setattr(runner, 'run_child', changed_source)
    # Source mutation is an integrity interruption, not a clean failure/resume.
    with pytest.raises(ValueError, match='provenance changed'):
        runner.run()
    state = runner.read_json(auth['output']/'checkpoint.json')
    assert state['status'] == 'running'
    assert sum(e['status'] == 'running' for e in state['cases'].values()) == 1
    assert not list(auth['output'].glob('*.npz'))


@pytest.mark.parametrize('failure', ['timeout', 'child_exit', 'missing_receipt', 'malformed_receipt', 'empty_error'])
def test_first_execution_failure_is_terminal_retained_and_never_replayed(auth, monkeypatch, failure):
    calls = []
    def fail(spec, log, timeout):
        calls.append(str(spec))
        if failure == 'timeout':
            raise subprocess.TimeoutExpired('fabricated child', timeout)
        if failure == 'child_exit':
            raise subprocess.CalledProcessError(9, 'fabricated child')
        if failure == 'empty_error':
            raise ValueError('')
        if failure == 'malformed_receipt':
            files = runner.read_json(spec)['files']
            Path(files['array']).write_bytes(b'partial-array')
            Path(files['receipt']).write_text('{')
    monkeypatch.setattr(runner, 'run_child', fail)
    state = runner.run()
    assert state['status'] == runner.STOPPED and len(calls) == 1
    assert len(state['cases']) == 104
    assert sum(e['status'] == 'pending' for e in state['cases'].values()) == 103
    key, failed = first_entry(state, 'execution-failure-retained')
    assert failed['error'] and failed['case_seconds'] >= 0
    assert set(failed['artifacts']) >= {str(runner.case_files(auth['output'], key)[n]) for n in ('spec', 'log')}
    runner.validate_state(state, auth)
    assert runner.run() == state and len(calls) == 1
    with pytest.raises(ValueError, match='no interrupted or terminal replay'):
        runner.validate_state(state, auth, resume=True)


def test_timeout_with_still_locked_worker_remains_unresolved_not_terminal(auth, monkeypatch):
    held = []
    def orphan(spec, log, timeout):
        lock = (auth['output']/'worker.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        held.append(lock)
        raise subprocess.TimeoutExpired('fabricated still-live child', timeout)
    monkeypatch.setattr(runner, 'run_child', orphan)
    try:
        with pytest.raises(BlockingIOError):
            runner.run()
        state = runner.read_json(auth['output']/'checkpoint.json')
        assert state['status'] == state['controller']['state'] == 'running'
        assert sum(e['status'] == 'running' for e in state['cases'].values()) == 1
    finally:
        for lock in held:
            lock.close()
    with pytest.raises(ValueError, match='interrupted pilot worker'):
        runner.run()


@pytest.mark.parametrize('defect', ['parent', 'unheld_controller', 'environment', 'linked_worker'])
def test_direct_worker_cannot_bypass_controller_or_environment(auth, monkeypatch, defect):
    auth['output'].mkdir()
    state = runner.new_state(auth)
    row = runner.case_table()[0]
    key = runner.label(row)
    files = runner.case_files(auth['output'], key)
    runner.write_json(files['spec'], runner.spec_for(auth, row))
    files['log'].write_text('')
    state['cases'][key]['status'] = 'running'
    state['cases'][key]['artifacts'] = {str(files['spec']): runner.file_sha(files['spec'])}
    if defect == 'parent':
        state['controller']['pid'] += 10000
    if defect == 'environment':
        monkeypatch.setenv('OMP_NUM_THREADS', '2')
    if defect == 'linked_worker':
        (auth['output']/'worker.lock').symlink_to(files['log'])
    runner.write_json(auth['output']/'checkpoint.json', state)
    with (auth['output']/'runner.lock').open('a') as parent:
        if defect != 'unheld_controller':
            fcntl.flock(parent, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError):
            runner.worker(files['spec'])
    assert not files['array'].exists() and not files['receipt'].exists()


@pytest.mark.parametrize('prior,elapsed,allowed', [(0., 0., True), (24540., 0., True),
    (24540., .001, False), (0., 28140., False), (25200., 0., False)])
def test_budget_reserves_full_case_and_cleanup_with_global_bound(auth, prior, elapsed, allowed):
    assert runner.budget_allows(prior, elapsed, auth['reg']['resources']) is allowed


@pytest.mark.parametrize('value', [-1., True, float('nan'), float('inf')])
def test_invalid_budget_is_not_execution_permission(auth, value):
    with pytest.raises(ValueError):
        runner.budget_allows(value, 0., auth['reg']['resources'])


@pytest.mark.parametrize('bound', ['time', 'disk'])
def test_budget_stop_retains_all_pending_identities_without_qualification(auth, monkeypatch, bound):
    state = runner.run(stop_after_cases=1)
    retained = copy.deepcopy(state['cases'])
    if bound == 'time':
        state['controller']['cumulative_charged_seconds'] = 25199.
    else:
        # Test only the cap enforcement; this fake authorization is not a seal.
        auth['reg']['resources']['maximum_output_bytes'] = runner.output_bytes(auth['output'])+1
    runner.write_json(auth['output']/'checkpoint.json', state, atomic=True)
    monkeypatch.setattr(runner, 'run_child', lambda *a: pytest.fail('budget overrun'))
    stopped = runner.run()
    assert stopped['status'] == runner.EXHAUSTED
    assert stopped['cases'] == retained and len(stopped['cases']) == 104
    assert stopped['scientific_goal_complete'] is False
    assert runner.run() == stopped


@pytest.mark.parametrize('which', ['checkpoint', 'full', 'landing', 'reference'])
def test_worker_artifacts_use_real_input_support_but_fabricated_states(auth, which):
    row = next(r for r in runner.case_table() if
        (r['engine'] == 'rkf78' if which == 'reference' else r['engine'] == 'gbs' and r['role'] == which))
    files = runner.case_files(auth['output'], runner.label(row))
    auth['output'].mkdir()
    states, _command, _epoch, _final, stats, embedded, seconds, restart_seconds = runner.forward(
        FabricatedReference() if which == 'reference' else FabricatedGBS(), row, auth['reg'], files)
    assert list(states.shape) == runner.input_signature(row)['shape']
    assert stats['position'] == len(states)-1 and seconds >= restart_seconds >= 0
    assert files['restart'].exists() is (which == 'checkpoint')
    assert (embedded is not None) is (which == 'reference')
    assert np.all(states[:, 0] == -65.)
    np.testing.assert_array_equal(states[:, 1], .05+np.arange(len(states))*1e-10)


@pytest.mark.parametrize('defect', ['float32', 'short', 'nonfinite', 'initial', 'gate',
    'negative_counts', 'bool_counts', 'work_support', 'input_mutation'])
def test_invalid_forward_output_cannot_be_saved_as_success(auth, defect):
    row = next(r for r in runner.case_table() if r['engine'] == 'rkf78')
    class InvalidReference(FabricatedReference):
        def integrate(self, *args, **kwargs):
            result = super().integrate(*args, **kwargs)
            if defect == 'float32':
                result.states = result.states.astype(np.float32)
            elif defect == 'short':
                result.states = result.states[:-1]
            elif defect == 'nonfinite':
                result.states[-1, 0] = np.inf
            elif defect == 'initial':
                result.states[0, 0] += 1
            elif defect == 'gate':
                result.states[-1, 1] = 2
            elif defect == 'input_mutation':
                args[1][0] += 1
            else:
                # Reference stats are constructed by forward(), so malformed
                # engine counts test the resulting shared accounting guard.
                result.rhs_evaluations = (-1 if defect == 'negative_counts' else
                    True if defect == 'bool_counts' else 1)
            return result
    with pytest.raises(ValueError):
        runner.forward(InvalidReference(), row, auth['reg'], {})


@pytest.mark.parametrize('defect', ['missing', 'extra', 'position', 'dense'])
def test_malformed_primary_statistics_fail_before_array_write(auth, defect):
    row = next(r for r in runner.case_table() if r['engine'] == 'gbs' and r['role'] == 'full')
    class InvalidContext(FabricatedContext):
        @property
        def stats(self):
            result = super().stats
            if defect == 'missing':
                result.pop('rhs')
            elif defect == 'extra':
                result['ignored'] = 1
            elif defect == 'position':
                result['position'] -= 1
            else:
                result['dense'] = len(self.current)
            return result
    engine = SimpleNamespace(context=lambda p, current, rate, initial, bits, policy: InvalidContext(current, initial))
    with pytest.raises(ValueError, match='work accounting'):
        runner.forward(engine, row, auth['reg'], {})


def test_subprocess_has_one_pinned_worker_environment_and_explicit_timeout(tmp_path, monkeypatch):
    recorded = []
    monkeypatch.setattr(runner.subprocess, 'run', lambda *a, **kw: recorded.append((a, kw)))
    with (tmp_path/'worker.log').open('wb') as log:
        runner.run_child(Path('/test-only.spec.json'), log, 600)
    arguments, options = recorded[0]
    assert arguments[0] == [sys.executable, str(runner.SCRIPT), '--worker', '/test-only.spec.json']
    assert options['check'] is True and options['timeout'] == 600
    assert all(options['env'][key] == value for key, value in runner.WORKER_ENV.items())


@pytest.mark.parametrize('defect', ['missing', 'extra', 'claim', 'row', 'signature_type',
    'array_hash', 'array_size', 'memory', 'timing', 'feature_scope'])
def test_malformed_success_receipt_becomes_retained_failure(auth, monkeypatch, defect):
    def child(spec, log, timeout):
        runner.worker(spec)
        files = runner.read_json(spec)['files']
        receipt = runner.read_json(files['receipt'])
        if defect == 'missing':
            receipt.pop('arrays_sha256')
        elif defect == 'extra':
            receipt['unexpected'] = 1
        elif defect == 'claim':
            receipt['scientific_goal_complete'] = True
        elif defect == 'row':
            receipt['case']['bits'] = 640
        elif defect == 'signature_type':
            receipt['input_signature']['shape'][1] = 5.0
        elif defect == 'array_hash':
            receipt['arrays_sha256'] = '0'*64
        elif defect == 'array_size':
            receipt['compressed_bytes'] += 1
        elif defect == 'memory':
            receipt['peak_process_RSS_bytes'] = True
        elif defect == 'timing':
            receipt['forward_call_seconds'] = -1.
        else:
            receipt['features'] = receipt['score'] = None
        # Overwrite only an isolated fabricated worker fixture, never raw study
        # evidence. The controller must retain and classify this bad receipt.
        Path(files['receipt']).write_text(runner.encoded(receipt))
    monkeypatch.setattr(runner, 'run_child', child)
    state = runner.run()
    assert state['status'] == runner.STOPPED
    key, entry = first_entry(state, 'execution-failure-retained')
    assert entry['error'] and len(state['cases']) == 104
    assert str(runner.case_files(auth['output'], key)['receipt']) in entry['artifacts']
    runner.validate_state(state, auth)


def test_full_fabricated_ledger_finishes_without_scientific_qualification(auth, monkeypatch):
    # Running all 104 *fabricated* outputs exercises fixed support; no model
    # driver, compiled library or actual solver is loaded by this fixture.
    state = runner.run()
    assert len(state['cases']) == 104 and state['status'] == runner.TERMINAL
    assert all(e['status'] == 'retained' for e in state['cases'].values())
    runner.validate_state(state, auth)
    assert state['long_reference_qualified'] is state['scientific_goal_complete'] is False
    monkeypatch.setattr(runner, 'run_child', lambda *a: pytest.fail('terminal replay'))
    assert runner.run() == state
