"""Terminal and integrity gates, using synthetic temporary files only."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

scripts = Path(__file__).parents[1]/'scripts'
for name in ('audit_patchseq_fit_checkpoint', 'verify_patchseq_fit_checkpoint_structure'):
    spec = importlib.util.spec_from_file_location(name, scripts/(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
mod = sys.modules['verify_patchseq_fit_checkpoint_structure']
STATUS = 'training-search-and-selected-numerical-controls-retained-independent-verification-pending'


def setup(tmp_path, status=STATUS):
    (tmp_path/'runner.lock').touch()
    (tmp_path/'checkpoint.json').write_text(json.dumps({'status': status}))


def test_no_checkpoint_is_not_success(tmp_path):
    with pytest.raises(ValueError), mod.terminal_checkpoint(tmp_path): pass


def test_live_runner_rejected(tmp_path):
    import fcntl
    setup(tmp_path)
    with (tmp_path/'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match='live'), mod.terminal_checkpoint(tmp_path): pass


def test_incomplete_state_rejected(tmp_path):
    setup(tmp_path, 'computing')
    with pytest.raises(ValueError, match='incomplete'), mod.terminal_checkpoint(tmp_path): pass


def test_completed_state_keeps_lock(tmp_path):
    import fcntl
    setup(tmp_path)
    with mod.terminal_checkpoint(tmp_path) as (state, sha):
        assert state['status'] == STATUS and len(sha) == 64
        with (tmp_path/'runner.lock').open('a') as lock, pytest.raises(BlockingIOError):
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_mutation_during_check_detected(tmp_path):
    setup(tmp_path)
    with pytest.raises(ValueError, match='changed'), mod.terminal_checkpoint(tmp_path):
        (tmp_path/'checkpoint.json').write_text('{}')


def test_hash_mutation_detected(tmp_path):
    p = tmp_path/'source'; p.write_text('before')
    pins = {str(p): mod.digest(p)}
    mod.verify_source_pins(pins)
    p.write_text('after')
    with pytest.raises(ValueError): mod.verify_source_pins(pins)


def test_held_out_digest_forbidden_before_receipt_read(monkeypatch):
    source = {'batches': [{'name': 'original'}]}
    monkeypatch.setattr(mod, 'read_yaml', lambda path: source)
    state = {'strata': {'Sst-L5': {'opened_training_files': [
        {'source_batch': 'original', 'slot': 1, 'partition': 'held-out'}]}}}
    catalog = {'records': [{'source_batch': 'original', 'slot': 1, 'partition': 'held-out'}]}
    with pytest.raises(ValueError, match='held-out'):
        mod.verify_training_bodies(state, {'strata': ['Sst-L5']}, catalog)
