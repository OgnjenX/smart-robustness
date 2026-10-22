"""Adverse audit admission/process fixtures; no solver, builds or pilot run."""

from __future__ import annotations

import ast
import copy
import fcntl
import inspect
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
import run_patchseq_gbs_pilot as producer
import verify_patchseq_gbs_pilot as audit
from patchseq_gbs_pilot_array_audit import ArrayAudit
from test_patchseq_gbs_pilot_array_audit import fixture as fabricated_array


def evidence(root):
    """Create fabricated arrays/specs/receipts in an isolated test directory."""
    root.mkdir()
    source = root.parent/'fabricated-source.py'
    source.write_text('TEST ONLY: no actual source/build execution approval')
    reg = yaml.safe_load(audit.REG.read_text())
    builds = {'gbs': {'fabricated': 'not a loaded primary library'},
              'rkf78': {'fabricated': 'not a loaded independent library'}}
    auth = {'reg': reg, 'prep': {'builds': builds, 'runtime': {'fabricated': 'not a runtime seal'}},
        'pins': {str(source): audit.digest(source)}, 'registration_sha256': 'a'*64,
        'seal_sha256': 'b'*64, 'preparation_sha256': 'c'*64, 'output': root}
    model = ArrayAudit(reg, builds)
    state = producer.new_state(auth)
    for key, row in model.rows.items():
        states, receipt, restart = fabricated_array(model, key)
        paths = audit.case_paths(root, key)
        audit.write_exclusive(paths['spec'], producer.spec_for(auth, row))
        paths['log'].write_text('fabricated worker fixture, not a model execution\n')
        with paths['array'].open('wb') as file:
            np.savez_compressed(file, states=states)
        receipt['arrays_sha256'] = audit.digest(paths['array'])
        receipt['compressed_bytes'] = paths['array'].stat().st_size
        audit.write_exclusive(paths['receipt'], receipt)
        if restart is not None:
            audit.write_exclusive(paths['restart'], restart)
        state['cases'][key].update(status='retained', case_seconds=1., error=None,
            artifacts={str(path): audit.digest(path) for path in paths.values() if path.exists()})
    state['status'] = audit.TERMINALS[0]
    state['controller'].update(state='closed-clean', cumulative_charged_seconds=164., measured_invocation_seconds=104.)
    audit.write_exclusive(root/'checkpoint.json', state)
    return auth, state


@pytest.fixture(scope='module')
def baseline(tmp_path_factory):
    return evidence(tmp_path_factory.mktemp('fabricated-independent-audit')/'output')


def test_auditor_does_not_import_producer_controller_solver_or_primary_features():
    imports = []
    for node in ast.walk(ast.parse(inspect.getsource(audit))):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module)
    assert not any(name.startswith(('run_patchseq', 'patchseq_gbs_hh', 'patchseq_rkf78',
                                  'patchseq_response', 'patchseq_fit_objective')) for name in imports)
    assert 'ctypes' not in imports and 'h5py' not in imports


@pytest.mark.parametrize('entry', ['parent', 'worker'])
def test_unsealed_audit_refuses_before_output_access(tmp_path, monkeypatch, entry):
    monkeypatch.setattr(audit, 'SEAL', tmp_path/'absent-decision.yaml')
    monkeypatch.setattr(audit, 'PREP', tmp_path/'no-preparation')
    with pytest.raises(ValueError, match='YAML'):
        audit.run_audit() if entry == 'parent' else audit.audit_worker(tmp_path/'absent-control')
    assert list(tmp_path.iterdir()) == []


def test_independent_source_closure_matches_controller_and_runtime_contract():
    assert audit.repository_sources() == producer.repository_pins()
    assert audit.ENV == producer.WORKER_ENV
    assert audit.REQUIRED_PROBES == producer.EXPECTED_ARITHMETIC_PROBES


def test_independently_discovered_actual_runtime_matches_controller_without_loading_a_solver():
    assert audit.runtime() == producer.runtime_fingerprint()


@pytest.fixture
def contract_files(tmp_path, monkeypatch):
    """Inert temporary binaries and TEST-only approval, not real execution authority."""
    monkeypatch.setattr(audit, 'ROOT', tmp_path)
    monkeypatch.setattr(audit, 'PREP', tmp_path/'preparation')
    monkeypatch.setattr(audit, 'SEAL', tmp_path/'TEST-only-decision.yaml')
    audit.PREP.mkdir()
    source = tmp_path/'TEST-only-source.py'
    source.write_text('No compiler, model, recording, or real execution approval used')
    sources = {str(source): audit.digest(source)}
    actual_runtime = {'file_hashes': sources, 'TEST-only': True}
    monkeypatch.setattr(audit, 'repository_sources', lambda: sources)
    monkeypatch.setattr(audit, 'runtime', lambda: actual_runtime)
    builds, pins = {}, dict(sources)
    for name, binary in (('gbs', 'hh.so'), ('rkf78', 'reference.so')):
        folder = audit.PREP/(name+'-build')
        folder.mkdir()
        path = folder/binary
        path.write_bytes(b'NOT an executable or loaded library')
        builds[name] = {'source_hashes': sources, 'binary_path': str(path), 'binary_sha256': audit.digest(path)}
        audit.write_exclusive(folder/'build.json', builds[name])
        pins.update({str(path): audit.digest(path), str(folder/'build.json'): audit.digest(folder/'build.json')})
    prep = {'status': 'built-and-pinned-no-pilot-trajectories', 'source_hashes': sources,
        'runtime': actual_runtime, 'pins': pins, 'builds': builds, 'preparation_seconds': .1,
        'arithmetic_probes': copy.deepcopy(audit.REQUIRED_PROBES),
        'biological_access': False, 'pilot_executed': False, 'fitting_restart': False}
    audit.write_exclusive(audit.PREP/'preparation.json', prep)
    reg = yaml.safe_load(audit.REG.read_text())
    seal = {'schema_version': 1, 'status': 'sealed-before-short-synthetic-pilot', 'human_execution_approved': True,
        'human_approval_record': 'TEST fixture only; not human approval of any actual pilot',
        'registration_sha256': audit.digest(audit.REG), 'preparation_path': str((audit.PREP/'preparation.json').relative_to(tmp_path)),
        'preparation_sha256': audit.digest(audit.PREP/'preparation.json'), 'resources': copy.deepcopy(reg['resources']),
        'allowed_operations': ['short-synthetic-pilot', 'independent-pilot-audit'], 'source_hashes': sources}
    audit.SEAL.write_text(yaml.safe_dump(seal))
    return seal, prep


def test_exact_test_contract_binds_every_build_source_runtime_product_and_receipt(contract_files):
    seal, prep = contract_files
    auth = audit.load_contract()
    assert auth['prep'] == prep
    assert auth['seal_sha256'] == audit.digest(audit.SEAL)
    assert auth['registration_sha256'] == seal['registration_sha256']
    assert not auth['output'].exists()


def test_readonly_preparation_check_does_not_create_or_require_execution_authority(contract_files):
    _seal, expected = contract_files
    audit.SEAL.unlink()  # Only this temporary TEST decision is removed.
    prep, pins = audit.checked_preparation()
    assert prep == expected and pins == expected['pins']
    assert not audit.SEAL.exists() and not (audit.ROOT/'results').exists()


@pytest.mark.parametrize('defect', ['extra_seal', 'schema_type', 'human_type', 'empty_approval', 'operation',
    'resources', 'resource_type', 'reg_hash', 'prep_path', 'prep_hash', 'sources', 'runtime',
    'source_closure', 'missing_build', 'extra_prep', 'prep_flag', 'prep_time', 'build_path',
    'build_receipt', 'product_hash', 'pin_support', 'actual_source_changed', 'precision_probe', 'reference_table_probe'])
def test_source_runtime_build_and_execution_admission_rejects_forged_or_incomplete_contracts(contract_files, defect):
    seal, prep = contract_files
    if defect == 'extra_seal':
        seal['extra'] = True
    elif defect == 'schema_type':
        seal['schema_version'] = True
    elif defect == 'human_type':
        seal['human_execution_approved'] = 1
    elif defect == 'empty_approval':
        seal['human_approval_record'] = ' '
    elif defect == 'operation':
        seal['allowed_operations'].append('biological-fitting')
    elif defect == 'resources':
        seal['resources']['maximum_case_seconds'] += 1
    elif defect == 'resource_type':
        seal['resources']['concurrent_workers'] = True
    elif defect == 'reg_hash':
        seal['registration_sha256'] = '0'*64
    elif defect == 'prep_path':
        seal['preparation_path'] = 'other/preparation.json'
    elif defect == 'prep_hash':
        seal['preparation_sha256'] = '0'*64
    elif defect == 'sources':
        seal['source_hashes'] = {}
    elif defect == 'runtime':
        prep['runtime'] = {}
    elif defect == 'source_closure':
        prep['source_hashes'] = {}
    elif defect == 'missing_build':
        prep['builds'].pop('rkf78')
    elif defect == 'extra_prep':
        prep['extra'] = True
    elif defect == 'prep_flag':
        prep['pilot_executed'] = True
    elif defect == 'prep_time':
        prep['preparation_seconds'] = 1
    elif defect == 'build_path':
        prep['builds']['gbs']['binary_path'] += '.other'
    elif defect == 'build_receipt':
        (audit.PREP/'gbs-build/build.json').write_text('{}')
    elif defect == 'product_hash':
        Path(prep['builds']['gbs']['binary_path']).write_bytes(b'changed actual binary')
    elif defect == 'pin_support':
        prep['pins'].pop(prep['builds']['gbs']['binary_path'])
    elif defect == 'precision_probe':
        prep['arithmetic_probes']['gbs']['precision320'] = 319
    elif defect == 'reference_table_probe':
        prep['arithmetic_probes']['rkf78_table_sha256'] = '0'*64
    else:
        Path(next(iter(prep['source_hashes']))).write_text('changed source')
    (audit.PREP/'preparation.json').write_text(audit.canonical(prep))
    if defect not in ('prep_hash',):
        seal['preparation_sha256'] = audit.digest(audit.PREP/'preparation.json')
    audit.SEAL.write_text(yaml.safe_dump(seal))
    with pytest.raises(ValueError):
        audit.load_contract()


def test_duplicate_nonfinite_and_symlink_metadata_are_not_read(tmp_path):
    path = tmp_path/'meta.json'
    for raw in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":1e999}'):
        path.write_text(raw)
        with pytest.raises(ValueError):
            audit.read_json(path)
    path.write_text('a: 1\na: 2\n')
    with pytest.raises(ValueError, match='duplicate'):
        audit.read_yaml(path)
    link = tmp_path/'link'
    link.symlink_to(path)
    with pytest.raises(ValueError):
        audit.read_json(link)
    with pytest.raises(ValueError):
        audit.write_exclusive(link, {})


def test_source_pin_guards_do_not_open_biological_or_unrelated_results(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, 'digest', lambda *a: pytest.fail('forbidden source body opened'))
    for name in (str(tmp_path/'body.nwb'), str(audit.ROOT/'results/unrelated-response/body.bin')):
        with pytest.raises(ValueError):
            audit.verify_pins({name: '0'*64})


def test_exact_complete_terminal_checkpoint_binds_all_artifacts_and_native_support(baseline):
    auth, _state = baseline
    state, model, hashes, size = audit.check_checkpoint(auth)
    assert len(model.rows) == len(state['cases']) == 104
    assert len(hashes) == 104*4+8+1
    assert size < 2**30
    assert hashes[str(auth['output']/'checkpoint.json')] == audit.digest(auth['output']/'checkpoint.json')


@pytest.mark.parametrize('defect', ['missing_case', 'extra_case', 'boolean_row', 'config_hash', 'source_support',
    'build', 'runtime', 'authority', 'status', 'controller_running', 'pid', 'invocation', 'time_type',
    'budget_reset', 'running_case', 'failure_status_without_failure', 'pending_orphan', 'missing_artifact_hash'])
def test_incomplete_altered_or_interrupted_ledgers_cannot_enter_audit(baseline, monkeypatch, defect):
    auth, original = baseline
    state = copy.deepcopy(original)
    key = next(iter(state['cases']))
    if defect == 'missing_case':
        state['cases'].pop(key)
    elif defect == 'extra_case':
        state['cases']['extra'] = state['cases'][key]
    elif defect == 'boolean_row':
        state['cases'][key]['case']['repeat'] = False
    elif defect == 'config_hash':
        state['input_manifest_sha256'] = '0'*64
    elif defect == 'source_support':
        state['pins'] = {}
    elif defect == 'build':
        state['builds'] = {}
    elif defect == 'runtime':
        state['runtime'] = {}
    elif defect == 'authority':
        state['scientific_goal_complete'] = True
    elif defect == 'status':
        state['status'] = 'running'
    elif defect == 'controller_running':
        state['controller']['state'] = 'running'
    elif defect == 'pid':
        state['controller']['pid'] = True
    elif defect == 'invocation':
        state['controller']['invocation'] = 0
    elif defect == 'time_type':
        state['controller']['measured_invocation_seconds'] = 104
    elif defect == 'budget_reset':
        state['controller']['cumulative_charged_seconds'] = 0.
    elif defect == 'running_case':
        state['cases'][key]['status'] = 'running'
    elif defect == 'failure_status_without_failure':
        state['status'] = audit.TERMINALS[1]
    elif defect == 'pending_orphan':
        state['cases'][key].update(status='pending', artifacts={}, case_seconds=None)
    else:
        state['cases'][key]['artifacts'].pop(next(iter(state['cases'][key]['artifacts'])))
    real_read = audit.read_json
    monkeypatch.setattr(audit, 'read_json', lambda path: state if Path(path) == auth['output']/'checkpoint.json' else real_read(path))
    with pytest.raises(ValueError):
        audit.check_checkpoint(auth)


@pytest.mark.parametrize('kind', ['object', 'shape', 'layout', 'extra_key', 'duplicate', 'stored', 'trailing'])
def test_numpy_archive_headers_are_bounded_before_array_allocation(tmp_path, kind):
    path = tmp_path/'array.npz'
    states = np.zeros((10, 5), dtype=np.float64)
    if kind == 'object':
        states = states.astype(object)
    elif kind == 'shape':
        states = states[:9]
    elif kind == 'layout':
        states = np.asfortranarray(states)
    if kind in ('duplicate', 'stored', 'trailing'):
        import io
        data = io.BytesIO()
        np.save(data, states)
        raw = data.getvalue()+(b'extra' if kind == 'trailing' else b'')
        with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_STORED if kind == 'stored' else zipfile.ZIP_DEFLATED) as file:
            file.writestr('states.npy', raw)
            if kind == 'duplicate':
                with pytest.warns(UserWarning, match='Duplicate name'):
                    file.writestr('states.npy', raw)
    else:
        with path.open('wb') as file:
            np.savez_compressed(file, states=states, **({'extra': states} if kind == 'extra_key' else {}))
    with pytest.raises(ValueError):
        audit.load_states(path, [10, 5])


def test_safe_exact_native_array_archive_roundtrip(tmp_path):
    states = np.arange(50, dtype=np.float64).reshape(10, 5)
    path = tmp_path/'array.npz'
    with path.open('wb') as file:
        np.savez_compressed(file, states=states)
    assert np.array_equal(audit.load_states(path, [10, 5]), states)


def test_array_symlink_is_refused_before_opening_archive(tmp_path, monkeypatch):
    path = tmp_path/'linked.npz'
    path.symlink_to(tmp_path/'not-evidence')
    monkeypatch.setattr(audit.zipfile, 'ZipFile', lambda *a, **kw: pytest.fail('linked archive opened'))
    with pytest.raises(ValueError, match='linked array'):
        audit.load_states(path, [10, 5])


def test_independent_analysis_reconstructs_all_fabricated_arrays_and_comparisons(baseline):
    auth, _state = baseline
    report = audit.analyze(auth)
    assert len(report['cases']) == 104 and len(report['comparisons']) == 120
    assert report['array_summary']['short_pilot_array_gates_passed'] is True
    assert report['resources']['native_array_uncompressed_bytes'] == 120805120
    assert report['scientific_goal_complete'] is report['long_reference_qualified'] is False
    assert not (auth['output']/'independent-assessment.json').exists()


@pytest.mark.parametrize('defect', ['case_omission', 'pair_omission', 'waived_pair', 'false_case_success',
    'new_authority', 'false_storage', 'false_timing', 'source_pin'])
def test_analysis_summary_cannot_replace_complete_independent_support(baseline, defect):
    auth, _state = baseline
    analysis = audit.analyze(auth)
    key = next(iter(analysis['cases']))
    if defect == 'case_omission':
        analysis['cases'].pop(key)
    elif defect == 'pair_omission':
        analysis['comparisons'].pop()
    elif defect == 'waived_pair':
        analysis['comparisons'][0]['required'] = False
    elif defect == 'false_case_success':
        analysis['cases'][key]['checks']['features_reconstructed'] = False
    elif defect == 'new_authority':
        analysis['fitting_restart'] = True
    elif defect == 'false_storage':
        analysis['resources']['native_array_uncompressed_bytes'] -= 1
    elif defect == 'false_timing':
        analysis['resources']['case_timings'][0]['total_case_seconds'] = 0.
    else:
        analysis['source_hashes'] = {}
    with pytest.raises(ValueError):
        audit.validate_analysis(auth, analysis, analysis['raw_artifact_hashes'])


@pytest.fixture
def process_fixture(tmp_path, monkeypatch):
    auth, state = evidence(tmp_path/'output')
    monkeypatch.setattr(audit, 'load_contract', lambda: auth)
    monkeypatch.setattr(audit.os, 'getppid', os.getpid)
    for key, value in audit.ENV.items():
        monkeypatch.setenv(key, value)
    def child(args, **options):
        assert args == [sys.executable, str(audit.SCRIPT), '--audit-worker', str(auth['output']/'audit-control.json')]
        assert 0. < options['timeout'] <= 3540.
        assert options['check'] is True
        assert all(options['env'][k] == v for k, v in audit.ENV.items())
        audit.audit_worker(args[-1])
    monkeypatch.setattr(audit.subprocess, 'run', child)
    return auth, state


def test_only_terminal_parent_admission_can_publish_exclusive_assessment(process_fixture):
    auth, _state = process_fixture
    result = audit.run_audit()
    assert result['state'] == 'terminal-independent-pilot-evidence-assessment'
    assert result['short_pilot_admissibility_passed'] is True
    assert result['scientific_goal_complete'] is False
    root = auth['output']
    assert audit.read_json(root/'independent-assessment.json') == result
    assert result['audit_outcome_sha256'] == audit.digest(root/'audit-outcome.json')
    with pytest.raises(ValueError, match='no automatic replay'):
        audit.run_audit()


@pytest.mark.parametrize('failure', ['timeout', 'exit', 'missing_analysis', 'changed_source'])
def test_timeout_failed_child_or_changed_evidence_never_publishes_or_replays(process_fixture, monkeypatch, failure):
    auth, _state = process_fixture
    def child(args, **options):
        if failure == 'timeout':
            raise subprocess.TimeoutExpired(args, options['timeout'])
        if failure == 'exit':
            raise subprocess.CalledProcessError(2, args)
        if failure == 'changed_source':
            audit.audit_worker(args[-1])
            Path(next(iter(auth['pins']))).write_text('changed provenance')
    monkeypatch.setattr(audit.subprocess, 'run', child)
    with pytest.raises(ValueError, match='failed; evidence retained'):
        audit.run_audit()
    assert not (auth['output']/'independent-assessment.json').exists()
    outcome = audit.read_json(auth['output']/'audit-outcome.json')
    assert outcome['status'] == 'audit-failure-retained-no-qualification-no-replay'
    with pytest.raises(ValueError, match='no automatic replay'):
        audit.run_audit()


@pytest.mark.parametrize('name', ['runner.lock', 'worker.lock', 'audit.lock'])
def test_live_controller_worker_or_other_audit_blocks_launch(process_fixture, name):
    auth, _state = process_fixture
    with (auth['output']/name).open('a') as file:
        fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            audit.run_audit()
    assert not (auth['output']/'audit-control.json').exists()


def test_audit_worker_cannot_use_stale_control_with_unheld_parent_locks(process_fixture):
    auth, _state = process_fixture
    control = {'schema_version': 1, 'launcher_pid': os.getpid(),
        'checkpoint_sha256': audit.digest(auth['output']/'checkpoint.json'), 'authorization_sha256': audit.sha(auth['pins']),
        'maximum_audit_seconds': 3600, 'prelaunch_seconds': .1, 'worker_timeout_seconds': 3539.9}
    for name in ('runner.lock', 'worker.lock', 'audit.lock'):
        (auth['output']/name).touch()
    path = auth['output']/'audit-control.json'
    audit.write_exclusive(path, control)
    with pytest.raises(ValueError, match='live parent holding all locks'):
        audit.audit_worker(path)
    assert not (auth['output']/'audit-analysis.json').exists()


def test_failed_and_pending_identities_are_preserved_in_a_nonqualifying_assessment(process_fixture):
    auth, state = process_fixture
    failed, pending = list(state['cases'])[:2]
    state['cases'][failed].update(status='execution-failure-retained', error='fixed fabricated failure, never replaced')
    for path in audit.case_paths(auth['output'], pending).values():
        if path.exists():
            path.unlink()  # Isolated fixture only, never original raw results.
    state['cases'][pending].update(status='pending', artifacts={}, case_seconds=None, error=None)
    state['status'] = audit.TERMINALS[1]
    (auth['output']/'checkpoint.json').write_text(audit.canonical(state))
    result = audit.run_audit()
    assert len(result['cases']) == 104 and len(result['comparisons']) == 120
    for key in (failed, pending):
        assert result['cases'][key]['state'] == 'unavailable-evidence-preserved'
        assert result['cases'][key]['passed'] is False
        assert result['cases'][key]['ledger'] == state['cases'][key]
    assert result['array_summary']['short_pilot_array_gates_passed'] is False
    assert result['short_pilot_admissibility_passed'] is False


def test_audit_write_storage_bound_is_checked_before_creating_an_output(baseline):
    auth, _state = baseline
    limited = {**auth, 'reg': copy.deepcopy(auth['reg'])}
    limited['reg']['resources']['maximum_output_bytes'] = audit.inventory(auth['output'], audit.allowed_outputs())+1
    with pytest.raises(ValueError, match='output budget'):
        audit.write_audit(limited, 'audit-analysis.json', {'cannot_fit': True})
    assert not (auth['output']/'audit-analysis.json').exists()


def test_expired_admission_budget_cannot_start_a_child(process_fixture, monkeypatch):
    auth, _state = process_fixture
    clock = iter((0., 3600.))
    monkeypatch.setattr(audit.time, 'monotonic', lambda: next(clock))
    monkeypatch.setattr(audit.subprocess, 'run', lambda *a, **kw: pytest.fail('expired budget launched a child'))
    with pytest.raises(ValueError, match='exhausted the fixed audit budget'):
        audit.run_audit()
    assert not (auth['output']/'audit-control.json').exists()


def test_remaining_combined_budget_reduces_timeout_without_expansion(process_fixture, monkeypatch):
    auth, state = process_fixture
    state['controller']['cumulative_charged_seconds'] = 28700.
    (auth['output']/'checkpoint.json').write_text(audit.canonical(state))
    timeouts = []
    def child(args, **options):
        timeouts.append(options['timeout'])
        raise subprocess.TimeoutExpired(args, options['timeout'])
    monkeypatch.setattr(audit.subprocess, 'run', child)
    with pytest.raises(ValueError, match='no automatic replay'):
        audit.run_audit()
    assert len(timeouts) == 1 and 0. < timeouts[0] < 40.
    assert not (auth['output']/'independent-assessment.json').exists()


@pytest.mark.parametrize('defect', ['control_changed', 'raw_case_changed', 'analysis_subset'])
def test_child_exit_success_cannot_mask_changed_control_raw_evidence_or_support(process_fixture, monkeypatch, defect):
    auth, state = process_fixture
    def child(args, **options):
        audit.audit_worker(args[-1])
        if defect == 'control_changed':
            path = auth['output']/'audit-control.json'
            data = audit.read_json(path)
            data['maximum_audit_seconds'] = 28800
            path.write_text(audit.canonical(data))
        elif defect == 'raw_case_changed':
            key = next(iter(state['cases']))
            audit.case_paths(auth['output'], key)['array'].write_bytes(b'changed after child analysis')
        else:
            path = auth['output']/'audit-analysis.json'
            data = audit.read_json(path)
            data['cases'].pop(next(iter(data['cases'])))
            path.write_text(audit.canonical(data))
    monkeypatch.setattr(audit.subprocess, 'run', child)
    with pytest.raises(ValueError, match='failed; evidence retained'):
        audit.run_audit()
    assert not (auth['output']/'independent-assessment.json').exists()
