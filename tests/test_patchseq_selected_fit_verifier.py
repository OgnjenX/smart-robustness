"""Synthetic selected-stage prerequisites and claim-boundary checks."""

import importlib.util
import sys
from pathlib import Path

import pytest

scripts = Path(__file__).parents[1]/'scripts'
names = ('verify_patchseq_whole_file_integrity', 'verify_patchseq_fitting_targets',
         'audit_patchseq_fit_checkpoint', 'patchseq_response_features', 'patchseq_isolated_hh',
         'patchseq_fit_search', 'patchseq_hh_independent_replay', 'patchseq_independent_fit_inputs',
         'patchseq_independent_fit_loss', 'verify_patchseq_fit_checkpoint_structure',
         'verify_patchseq_isolated_fit_replay', 'patchseq_selected_fit_agreement',
         'verify_patchseq_selected_fit')
for name in names:
    spec = importlib.util.spec_from_file_location(name, scripts/(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
mod = sys.modules['verify_patchseq_selected_fit']


def replay_fixture():
    independent = {'status': 'all-candidate-forward-feature-loss-reconstruction-retained',
                   'input_checkpoint_sha256': 'abc', 'all_reconstructions_agree': True,
                   'held_out_access_authorized': False,
                   'strata': {'Sst-L5': {'candidates': {'0': {'passed': True}}}}}
    original = {'strata': {'Sst-L5': {'candidates': {'0': {}}}}}
    return independent, original, 'abc', {'strata': ['Sst-L5']}


def test_complete_replay_prerequisite():
    mod.require_completed_replay(*replay_fixture())


@pytest.mark.parametrize('mutation', ['status', 'sha', 'agreement', 'missing', 'failure', 'access'])
def test_incomplete_or_disagreeing_replay_blocks_selected_stage(mutation):
    independent, original, sha, protocol = replay_fixture()
    if mutation == 'status': independent['status'] = 'computing'
    if mutation == 'sha': independent['input_checkpoint_sha256'] = 'different'
    if mutation == 'agreement': independent['all_reconstructions_agree'] = False
    if mutation == 'missing': independent['strata']['Sst-L5']['candidates'].clear()
    if mutation == 'failure': independent['strata']['Sst-L5']['candidates']['0']['passed'] = False
    if mutation == 'access': independent['held_out_access_authorized'] = True
    with pytest.raises(ValueError): mod.require_completed_replay(independent, original, sha, protocol)


def selected_fixture(training=True, numeric=True, agreement=True):
    original = {'selected_candidate': 0, 'candidates': {'0': {'all_training_observable_gates_pass': training}}}
    results = {'sweep': {'all_reconstruction_checks_pass': agreement,
                         'independent_numerical_controls': {'passed': numeric}}}
    return original, results, {'sweep'}


def test_verified_training_does_not_authorize_heldout_or_network():
    out = mod.selected_summary(*selected_fixture())
    assert out['training_and_numerical_eligible']
    assert out['held_out_access_authorized'] is False and out['network_promotion'] is False


@pytest.mark.parametrize('kwargs', [{'training': False}, {'numeric': False}, {'agreement': False}])
def test_any_failed_gate_preserves_ineligibility(kwargs):
    assert not mod.selected_summary(*selected_fixture(**kwargs))['training_and_numerical_eligible']


def test_missing_selected_sweep_rejected():
    original, _, expected = selected_fixture()
    with pytest.raises(ValueError): mod.selected_summary(original, {}, expected)
