"""Mutation controls for structural audit; no real checkpoint or response access."""

import copy
import importlib.util
import sys
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('audit_patchseq_fit_checkpoint', Path(__file__).parents[1]/'scripts/audit_patchseq_fit_checkpoint.py')
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def fixture():
    protocol = {'strata': ['Sst-L5'], 'search': {'candidates_per_stratum': 2,
                                              'bounds': {f: [0., 2.] for f in mod.FIELDS}}}
    table = [{f: float(i) for f in mod.FIELDS} for i in range(2)]
    seal = {'source_sha256': {'runner': 'abc'}}
    catalog = {'records': [{'partition': 'fitting', 'source_group': 'Sst', 'layer_label': 'L5',
                           'source_batch': 'original', 'slot': 1, 'frozen_group_id': 'g',
                           'sweeps': [{'state': 'fitting-response-characterized', 'sweep_number': 4}]}]}
    candidates = {str(i): {'state': 'scored', 'parameters': table[i], 'loss': float(i),
                           'group_losses': {'g': float(i)}, 'simulation_failures': [],
                           'all_training_observable_gates_pass': True} for i in range(2)}
    control = {'passed': True, 'maximum_refinement_voltage_error_mV': 0.,
               'warmup_baseline_difference_mV': 0., 'gates': {name: True for name in
               ('exact_repeat', 'refinement_voltage', 'refinement_event_count',
                'warmup_baseline', 'warmup_event_count')}}
    state = {'status': mod.STATUS, 'source_hashes': {'runner': 'abc'}, 'candidate_table': table,
             'held_out_response_features_read': False, 'network_promotion': False,
             'frozen_baseline_modified': False, 'strata': {'Sst-L5': {'candidates': candidates,
             'opened_training_files': [{'source_batch': 'original', 'slot': 1, 'partition': 'fitting'}],
             'selected_candidate': 0, 'numerical_controls_complete': True,
             'all_numerical_controls_pass': True, 'numerical_controls': {'["original", 1, 4]': control}}}}
    return state, protocol, catalog, seal


def test_scope_is_explicitly_not_forward_verification():
    report = mod.audit(*fixture())
    assert report['status'] == 'structural-accounting-only-forward-replay-not-verified'
    assert report['held_out_access_authorized'] is False


@pytest.mark.parametrize('mutation', ['status', 'access', 'stratum', 'seal', 'table', 'missing',
                                    'parameters', 'groups', 'loss', 'nan', 'failure', 'selection',
                                    'control_missing', 'control_threshold', 'aggregate', 'held_file',
                                    'held_source', 'duplicate_case'])
def test_reject_corrupted_or_incomplete_evidence(mutation):
    state, protocol, catalog, seal = fixture()
    rows = state['strata']['Sst-L5']; row = rows['candidates']['0']
    if mutation == 'status': state['status'] = 'computing'
    if mutation == 'access': state['held_out_response_features_read'] = True
    if mutation == 'stratum': state['strata']['Vip-L5'] = copy.deepcopy(rows)
    if mutation == 'seal': state['source_hashes']['runner'] = 'different'
    if mutation == 'table': state['candidate_table'][1] = {}
    if mutation == 'missing': rows['candidates'].pop('1')
    if mutation == 'parameters': row['parameters'] = {}
    if mutation == 'groups': row['group_losses']['unobserved'] = 0
    if mutation == 'loss': row['loss'] = 1
    if mutation == 'nan': row['group_losses']['g'] = float('nan')
    if mutation == 'failure': row['simulation_failures'] = [{'source_key': ['original', 1, 99]}]
    if mutation == 'selection': rows['selected_candidate'] = 1
    if mutation == 'control_missing': rows['numerical_controls'].clear()
    if mutation == 'control_threshold': next(iter(rows['numerical_controls'].values()))['maximum_refinement_voltage_error_mV'] = .2
    if mutation == 'aggregate': rows['all_numerical_controls_pass'] = False
    if mutation == 'held_file': rows['opened_training_files'][0]['partition'] = 'held-out'
    if mutation == 'held_source': catalog['records'][0]['partition'] = 'held-out'
    if mutation == 'duplicate_case': catalog['records'].append(copy.deepcopy(catalog['records'][0]))
    with pytest.raises(ValueError): mod.audit(state, protocol, catalog, seal)


def test_numerical_failures_are_preserved_not_converted_to_passes():
    state, protocol, catalog, seal = fixture()
    rows = state['strata']['Sst-L5']
    rows['numerical_controls'] = {'["original", 1, 4]': {'passed': False, 'error': 'numerical failure'}}
    rows['all_numerical_controls_pass'] = False
    assert mod.audit(state, protocol, catalog, seal)['biological_validation_complete'] is False
