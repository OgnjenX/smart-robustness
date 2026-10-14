"""Transition accounting safeguards; synthetic checkpoint fixtures only."""

import copy
import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
import patchseq_fit_transition_audit as audit_mod
import run_patchseq_compiled_fit as runner


def fixture():
    config = runner.read_yaml(runner.REG)
    table = runner.candidates(config['search'])
    rows = {str(i): {'state': 'scored', 'parameters': runner.candidate_dict(p), 'loss': 0}
            for i, p in enumerate(table[:4])}
    return config, table, {'candidate_table': [runner.candidate_dict(p) for p in table],
                           'strata': {'Pvalb-L23': {'candidates': rows}}}


def test_complete_prefix_accepted_without_outcome_selection():
    config, table, old = fixture()
    runner.retained_scope(old, table, config['strata'])


@pytest.mark.parametrize('mutation', ['partial', 'hole', 'selected', 'terminal', 'parameter'])
def test_unsafe_checkpoint_transition_rejected(mutation):
    config, table, old = fixture()
    rows = old['strata']['Pvalb-L23']
    if mutation == 'partial': rows['candidates'].pop('3')
    if mutation == 'hole': rows['candidates']['4'] = rows['candidates'].pop('3')
    if mutation == 'selected': rows['selected_candidate'] = 0
    if mutation == 'terminal': old['status'] = 'complete'
    if mutation == 'parameter': rows['candidates']['0']['parameters']['leak_nS'] += 1
    with pytest.raises(ValueError):
        runner.retained_scope(old, table, config['strata'])


def test_preserved_prefix_and_exact_origin_audit(tmp_path):
    config, _, old = fixture()
    config['output'] = str(tmp_path)
    config['strata'] = ['Pvalb-L23']
    archive = tmp_path/'predecessor-checkpoint.json'
    archive.write_text(json.dumps(old))
    sha = audit_mod.digest(archive)
    state = copy.deepcopy(old)
    state.update(predecessor_checkpoint_sha256=sha,
        fitting_backend='compiled-RK4-with-retained-Brian2-prefix',
        source_hashes={str(archive): sha, 'synthetic-binary': 'abc'},
        compiled_build={'binary_path': 'synthetic-binary', 'binary_sha256': 'abc'},
        transition_checks={'Pvalb-L23': {k: {'passed': True} for k in old['strata']['Pvalb-L23']['candidates']}})
    state['strata']['Pvalb-L23']['candidate_backends'] = {str(i): 'retained-Brian2' for i in range(4)}
    assert audit_mod.audit_transition(state, config)['retained_original_rows_verified_and_unchanged']
    state['strata']['Pvalb-L23']['candidates']['0']['loss'] = 1
    with pytest.raises(ValueError, match='candidate changed'):
        audit_mod.audit_transition(state, config)


def test_distinct_independent_routes_and_original_protocol_preserved():
    config = runner.read_yaml(runner.REG)
    original = runner.read_yaml(config['original_protocol'])
    for name in ('search', 'replay', 'selected_numerical_controls', 'strata', 'runtime', 'held_out'):
        assert config[name] == original[name]
    assert config['output'] != original['output']
    for number in (1473, 1475):
        path = next(Path('docs/validation-results').glob(f'post2008-patchseq-amended*{number}.yaml'))
        reg = yaml.safe_load(path.read_text())
        assert reg['expected_independent_backend'] == 'independent-NumPy-RK4'
