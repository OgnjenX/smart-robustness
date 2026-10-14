"""Synthetic assessment classification controls; no actual scientific outcomes."""

import importlib.util
import sys
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('patchseq_training_assessment_gate', Path(__file__).parents[1]/'scripts/patchseq_training_assessment_gate.py')
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def fixture(training=True, numeric=True, agreement=True, table=True):
    protocol = {'strata': ['Sst-L5'], 'search': {'candidates_per_stratum': 2}}
    original = {'strata': {'Sst-L5': {'selected_candidate': 0,
        'candidates': {str(i): {'all_training_observable_gates_pass': training} for i in range(2)},
        'numerical_controls': {'sweep': {}}}}}
    independent = {'status': 'all-candidate-forward-feature-loss-reconstruction-retained',
        'input_checkpoint_sha256': 'abc', 'all_reconstructions_agree': table,
        'held_out_access_authorized': False, 'network_promotion': False,
        'strata': {'Sst-L5': {'candidates': {str(i): {'passed': table} for i in range(2)}}}}
    selected = {'status': 'selected-training-waveform-feature-and-numerical-checks-retained',
        'input_checkpoint_sha256': 'abc', 'held_out_access_authorized': False, 'network_promotion': False,
        'strata': {'Sst-L5': {'sweeps': {'sweep': {'all_reconstruction_checks_pass': agreement,
             'independent_numerical_controls': {'passed': numeric}}},
             'summary': {'selected_candidate': 0, 'all_selected_reconstructions_agree': agreement,
                         'all_selected_numerical_controls_pass': agreement and numeric,
                         'sealed_training_observable_gates_pass': training,
                         'training_and_numerical_eligible': agreement and numeric and training}}}}
    return original, independent, selected, protocol


@pytest.mark.parametrize('kwargs,category', [
    ({}, 'training-eligible-for-separately-sealed-held-out-design'),
    ({'table': False}, 'verification-disagreement-not-biological-failure'),
    ({'agreement': False}, 'selected-reconstruction-unresolved'),
    ({'numeric': False}, 'selected-numerical-gate-failed'),
    ({'training': False}, 'bounded-selected-observable-fit-failure')])
def test_classification_and_boundaries(kwargs, category):
    result = mod.assess_training(*fixture(**kwargs))
    assert result['strata'][0]['classification'] == category
    assert result['held_out_access_authorized'] is False
    assert result['network_promotion'] is False and result['scientific_goal_complete'] is False


@pytest.mark.parametrize('mutation', ['sha', 'scope', 'coverage', 'selection', 'summary', 'access', 'aggregate'])
def test_inconsistent_evidence_is_not_classified_as_success(mutation):
    original, independent, selected, protocol = fixture()
    if mutation == 'sha': selected['input_checkpoint_sha256'] = 'different'
    if mutation == 'scope': selected['strata'].clear()
    if mutation == 'coverage': independent['strata']['Sst-L5']['candidates'].pop('1')
    if mutation == 'selection': selected['strata']['Sst-L5']['summary']['selected_candidate'] = 1
    if mutation == 'summary': selected['strata']['Sst-L5']['summary']['all_selected_numerical_controls_pass'] = False
    if mutation == 'access': independent['held_out_access_authorized'] = True
    if mutation == 'aggregate': independent['all_reconstructions_agree'] = False
    with pytest.raises(ValueError): mod.assess_training(original, independent, selected, protocol)
