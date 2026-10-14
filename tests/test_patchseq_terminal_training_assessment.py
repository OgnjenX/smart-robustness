"""Synthetic terminal assessment/lock controls; no real results are read."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

scripts = Path(__file__).parents[1]/'scripts'
for name in ('audit_patchseq_fit_checkpoint', 'patchseq_training_assessment_gate',
             'verify_patchseq_fit_checkpoint_structure', 'patchseq_verification_routes', 'assess_patchseq_training'):
    spec = importlib.util.spec_from_file_location(name, scripts/(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
mod = sys.modules['assess_patchseq_training']


def test_terminal_verifier_lock_and_snapshot(tmp_path):
    import fcntl
    p = tmp_path/'checkpoint.json'; p.write_text(json.dumps({'status': 'complete'}))
    (tmp_path/'verifier.lock').touch()
    with mod.completed_verification(p, 'complete') as (state, sha):
        assert state['status'] == 'complete' and len(sha) == 64
        with (tmp_path/'verifier.lock').open('a') as lock, pytest.raises(BlockingIOError):
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_incomplete_verifier_rejected(tmp_path):
    p = tmp_path/'checkpoint.json'; p.write_text('{"status":"computing"}')
    (tmp_path/'verifier.lock').touch()
    with pytest.raises(ValueError), mod.completed_verification(p, 'complete'): pass


def test_snapshot_mutation_rejected(tmp_path):
    p = tmp_path/'checkpoint.json'; p.write_text('{"status":"complete"}')
    (tmp_path/'verifier.lock').touch()
    with pytest.raises(ValueError, match='changed'), mod.completed_verification(p, 'complete'):
        p.write_text('{}')


def replay_fixture():
    protocol = {'strata': ['Sst-L5'], 'search': {'candidates_per_stratum': 2}}
    original = {'strata': {'Sst-L5': {'candidates': {'0': {}, '1': {}}}}}
    independent = {'all_reconstructions_agree': False,
                   'strata': {'Sst-L5': {'candidates': {'0': {'passed': True}, '1': {'passed': False}}}}}
    return original, independent, protocol


def test_disagreement_does_not_require_or_authorize_selected_heldout_data():
    report = mod.replay_disagreement(*replay_fixture())
    assert report['strata'][0]['disagreeing_candidate_ids'] == ['1']
    assert report['held_out_access_authorized'] is False
    assert report['scientific_goal_complete'] is False


@pytest.mark.parametrize('mutation', ['missing', 'aggregate', 'truthy'])
def test_disagreement_evidence_must_be_complete_and_consistent(mutation):
    original, independent, protocol = replay_fixture()
    if mutation == 'missing': independent['strata']['Sst-L5']['candidates'].pop('0')
    if mutation == 'aggregate': independent['all_reconstructions_agree'] = True
    if mutation == 'truthy': independent['strata']['Sst-L5']['candidates']['0']['passed'] = 1
    with pytest.raises(ValueError): mod.replay_disagreement(original, independent, protocol)
