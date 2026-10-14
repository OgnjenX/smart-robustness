"""Append transport retains interruptions, caps and locks unchanged."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

scripts = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(scripts))
try:
    spec = importlib.util.spec_from_file_location('append_acquisition', scripts/'acquire_patchseq_append_recordings.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
finally:
    sys.path.pop(0)


def fixture(tmp_path):
    source = tmp_path/'pinned-source.txt'
    source.write_text('immutable')
    pins = {source: module.digest(source)}
    slots = [{'slot': n, 'expected_recording_bytes': 4} for n in (1, 2)]
    reg = {'execution': {'deadline_seconds_from_first_start': 10},
           'transport': {'expected_sealed_total_recording_bytes': 8, 'maximum_charged_recording_bytes': 8}}
    return tmp_path/'output', reg, slots, pins


def worker(calls, interrupted=None):
    def run(directory, slot, remaining, charge):
        calls.append(slot['slot'])
        charge()
        (directory/'000-body.bin').write_bytes(b'abcd')
        if slot['slot'] == interrupted:
            raise KeyboardInterrupt('synthetic process interruption')
        return {'status': 'synthetic-terminal', 'HTTP_receipts': [], 'physiological_arrays_interpreted': False}
    return run


def execute(inputs, callback, **kwargs):
    return module.run(*inputs, callback, now=kwargs.pop('now', lambda: 100),
                      available=kwargs.pop('available', lambda path: 1 << 40), **kwargs)


def test_terminal_noop_validates_evidence_and_never_repeats_network(tmp_path):
    inputs, calls = fixture(tmp_path), []
    result = execute(inputs, worker(calls))
    assert result['status'].startswith('terminal-all-selected') and result['charged_recording_bytes'] == 8
    assert execute(inputs, worker(calls)) == result and calls == [1, 2]
    (inputs[0]/'slot-001/000-body.bin').write_bytes(b'bad!')
    with pytest.raises(ValueError, match='evidence changed'):
        execute(inputs, worker(calls))
    assert calls == [1, 2]


def test_interrupted_get_is_retained_and_only_unattempted_slot_runs(tmp_path):
    inputs, calls = fixture(tmp_path), []
    with pytest.raises(KeyboardInterrupt):
        execute(inputs, worker(calls, interrupted=1))
    result = execute(inputs, worker(calls))
    assert calls == [1, 2] and result['charged_recording_bytes'] == 8
    receipt = yaml.safe_load((inputs[0]/'slot-001/receipt.yaml').read_text())
    assert receipt['status'] == 'interrupted-attempt-retained-no-retry'


def test_single_process_lock_rejects_duplicate_runner(tmp_path):
    inputs = fixture(tmp_path)
    with module.lock(inputs[0]), pytest.raises(BlockingIOError):
        execute(inputs, worker([]))


def test_original_deadline_is_not_reset_after_interruption(tmp_path):
    inputs, calls = fixture(tmp_path), []
    with pytest.raises(KeyboardInterrupt):
        execute(inputs, worker(calls, interrupted=1))
    state = execute(inputs, worker(calls), now=lambda: 111)
    assert state['deadline_epoch'] == 110 and state['status'] == 'terminal-absolute-deadline-stop'
    assert state['unattempted_slots'] == [2] and calls == [1]


def test_disk_preflight_prevents_first_transfer_and_preserves_stop(tmp_path):
    inputs, calls = fixture(tmp_path), []
    state = execute(inputs, worker(calls), available=lambda path: 0)
    assert state['status'] == 'terminal-storage-preflight-stop' and calls == []
    assert execute(inputs, worker(calls))['status'] == state['status'] and calls == []


def test_get_budget_and_marker_reject_retry_before_network(tmp_path):
    directory = tmp_path/'slot-001'
    directory.mkdir()
    slot = {'slot': 1, 'expected_recording_bytes': 4}
    with pytest.raises(ValueError, match='budget exhausted'):
        module.charge(directory, slot, [directory], 3)
    assert not (directory/'GET-attempt.yaml').exists()
    module.charge(directory, slot, [directory], 4)
    with pytest.raises(ValueError, match='no retry'):
        module.charge(directory, slot, [directory], 4)


def test_directory_creation_interruption_is_not_retried(tmp_path):
    inputs, calls = fixture(tmp_path), []
    def fail(directory, slot, remaining, charge):
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        execute(inputs, fail)
    (inputs[0]/'slot-001/attempt.yaml').unlink()
    execute(inputs, worker(calls))
    assert calls == [2]
    assert yaml.safe_load((inputs[0]/'slot-001/receipt.yaml').read_text())['admission_marker_missing'] is True


def test_source_mutation_prevents_resume(tmp_path):
    inputs, calls = fixture(tmp_path), []
    with pytest.raises(KeyboardInterrupt):
        execute(inputs, worker(calls, interrupted=1))
    next(iter(inputs[3])).write_text('changed')
    with pytest.raises(ValueError, match='source changed'):
        execute(inputs, worker(calls))
    assert calls == [1]


@pytest.mark.parametrize('case', ['both_match', 'conflict', 'numeric_other', 'missing', 'whitespace'])
def test_identity_never_selects_favorable_session_field(case):
    def entry(value, raw_type='str'):
        return {'present': True, 'scalar': {'raw_type': raw_type, 'value': value}}
    datasets = {'/general/session_id': entry('34'), '/session_id': entry('34'),
                '/general/subject/subject_id': entry('12')}
    if case == 'conflict':
        datasets['/session_id'] = entry('35')
    elif case == 'numeric_other':
        datasets['/session_id'] = entry(34, 'int')
    elif case == 'missing':
        datasets['/session_id'] = datasets['/general/session_id'] = {'present': False}
    elif case == 'whitespace':
        datasets['/general/session_id'] = entry(' 34')
    flags, accepted = module.identity({'datasets': datasets}, {'file_path': 'sub-12/sub-12_ses-34_icephys.nwb'})
    assert accepted is (case == 'both_match')
    if case == 'numeric_other':
        assert flags['/session_id'] is None
