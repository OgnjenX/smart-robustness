"""Prospective pilot metadata only: no solver, build, recording or experiment."""

from __future__ import annotations

import ast
import copy
import inspect
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
import patchseq_gbs_pilot_protocol as producer
import verify_patchseq_gbs_pilot_protocol as independent

ROOT = Path(__file__).parents[1]


@pytest.fixture(scope='module')
def protocol():
    return producer.document()


def test_complete_fixed_support_matches_separate_construction(protocol):
    rows, inputs, pairs = producer.case_table(), producer.input_manifest(), producer.comparison_table()
    summary = independent.validate_tables(rows, inputs, pairs)
    assert independent.validate_registration(protocol) == {
        **summary, 'execution_authorized': False, 'numerical_validation_performed': False}
    assert summary['total_cases'] == 104 and summary['comparison_count'] == 120
    assert Counter(row['role'] for row in rows) == {'full': 64, 'landing': 32, 'checkpoint': 8}
    assert Counter(row['engine'] for row in rows) == {'gbs': 72, 'rkf78': 32}
    assert len({producer.label(row) for row in rows}) == 104
    assert sum(pair['required'] for pair in pairs) == 112
    assert {producer.label(row) for row in rows} == {case for pair in pairs for case in (pair['left'], pair['right'])}
    assert protocol['resources']['total_native_array_uncompressed_bytes'] < 2**30
    assert (protocol['resources']['maximum_total_controller_seconds']+
            protocol['resources']['maximum_independent_audit_seconds'] ==
            protocol['resources']['maximum_controller_plus_audit_seconds'] == 28800)


def test_metadata_auditor_does_not_import_producer_or_numerical_routes():
    modules = []
    for node in ast.walk(ast.parse(inspect.getsource(independent))):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.append(node.module)
    assert all(not name.startswith('patchseq') for name in modules)
    assert all('gbs_hh' not in name and 'rkf78' not in name for name in modules)


@pytest.mark.parametrize('samples', [20001, 40001])
@pytest.mark.parametrize('profile', ['passive', 'active'])
def test_short_inputs_preserve_all_phases_and_fixed_physics(samples, profile):
    p, current, initial, command, epoch = producer.synthetic_input(samples, profile)
    assert command.shape == (samples,) and current.shape == (samples+5000,)
    assert command.dtype == current.dtype == initial.dtype == np.float64
    np.testing.assert_array_equal(current[:5000], 0.)
    np.testing.assert_array_equal(current[5000:], command)
    np.testing.assert_array_equal(command[:2500], 0.)
    np.testing.assert_array_equal(command[samples-2500:], 0.)
    np.testing.assert_array_equal(command[2500:samples-2500], -50. if profile == 'passive' else 2000.)
    assert p['capacitance_pF'] == 100. and p['leak_nS'] == 10. and p['threshold_shift_mV'] == -60.
    assert p['sodium_nS'] == (0. if profile == 'passive' else 2000.)
    np.testing.assert_array_equal(initial.view(np.uint64), np.array([-65., .05, .6, .2, .05]).view(np.uint64))
    assert epoch == {'step_start_index': 2500, 'step_stop_index': samples-2500, 'pre_window': [1500, 2000]}
    assert epoch['step_stop_index']-epoch['step_start_index'] >= 5000
    assert all(0 < index < len(current)-1 for index in producer.probes(samples))


def test_declared_window_contract_is_accepted_by_both_existing_extractors_without_a_simulation():
    from patchseq_response_features import characterize
    from verify_patchseq_fitting_targets import reconstruct

    for samples in (20001, 40001):
        for profile in ('passive', 'active'):
            _, _, _, command, epoch = producer.synthetic_input(samples, profile)
            # Constant fabricated voltage tests source-window support only, not
            # HH behavior or the numerical comparison's eventual outcome.
            voltage = np.full(samples, -65.)
            a = characterize(voltage, command, 50000., epoch, 0.)
            b = reconstruct(voltage, command, 50000., epoch, 0.)
            assert a == b and a['detected_event_count'] == 0


@pytest.mark.parametrize('defect', ['missing', 'duplicate', 'extra', 'reorder', 'boolean_factor',
                                  'floating_factor', 'changed_policy', 'changed_endpoint'])
def test_missing_altered_duplicate_or_reordered_cases_fail_independent_support(defect):
    rows = producer.case_table()
    if defect == 'missing':
        rows.pop()
    elif defect == 'duplicate':
        rows[-1] = copy.deepcopy(rows[0])
    elif defect == 'extra':
        rows.append(copy.deepcopy(rows[0]))
    elif defect == 'reorder':
        rows.reverse()
    elif defect in ('boolean_factor', 'floating_factor'):
        row = next(r for r in rows if r['factor'] == 1)
        row['factor'] = True if defect == 'boolean_factor' else 1.0
    elif defect == 'changed_policy':
        next(r for r in rows if r['policy'] == 'strict')['policy'] = 'tuned'
    else:
        next(r for r in rows if r['role'] == 'landing')['stop_index'] += 1
    with pytest.raises(ValueError):
        independent.validate_tables(rows, producer.input_manifest(), producer.comparison_table())


@pytest.mark.parametrize('which', ['initial', 'current_hash', 'shape', 'epoch', 'feature_scope'])
def test_input_support_is_not_trusted_from_worker_style_metadata(which):
    manifest = producer.input_manifest()
    item = manifest[next(iter(manifest))]
    if which == 'initial':
        item['initial_binary64'][0] = (-64.).hex()
    elif which == 'current_hash':
        item['integrated_current_sha256'] = '0'*64
    elif which == 'shape':
        item['shape'][0] -= 1
    elif which == 'epoch':
        item['epoch']['step_stop_index'] -= 1
    else:
        item['feature_scoring'] = not item['feature_scoring']
    with pytest.raises(ValueError):
        independent.validate_tables(producer.case_table(), manifest, producer.comparison_table())


@pytest.mark.parametrize('kind', ['missing', 'duplicate', 'waive', 'endpoint', 'wrong_pair'])
def test_comparison_controls_cannot_be_dropped_waived_or_repaired(kind):
    pairs = producer.comparison_table()
    if kind == 'missing':
        pairs.pop()
    elif kind == 'duplicate':
        pairs[-1] = copy.deepcopy(pairs[0])
    elif kind == 'waive':
        next(pair for pair in pairs if pair['required'])['required'] = False
    elif kind == 'endpoint':
        next(pair for pair in pairs if pair['kind'] == 'dense-landing')['scope'] = 'full'
    else:
        pairs[0]['right'] = pairs[0]['left']
    with pytest.raises(ValueError):
        independent.validate_tables(producer.case_table(), producer.input_manifest(), pairs)


@pytest.mark.parametrize('path,value', [
    (('boundaries', 'execution_authorized'), True), (('boundaries', 'long_reference_qualification'), True),
    (('boundaries', 'fitting_restart'), True), (('boundaries', 'original_score_replacement'), True),
    (('boundaries', 'biological_or_held_out_access'), True), (('inputs', 'warmup_samples'), 50000),
    (('inputs', 'active_step_pA'), 1999.), (('inputs', 'initial'), [-64., .05, .6, .2, .05]),
    (('inputs', 'parent_full_command_lengths'), [151000]), (('resources', 'maximum_case_seconds'), 601),
    (('resources', 'maximum_total_controller_seconds'), 28801), (('resources', 'concurrent_workers'), 2),
    (('comparisons', 'whole_voltage_mV'), 1.1e-5), (('comparisons', 'feature_float_relative'), 1e-8),
    (('comparisons', 'loss_absolute'), 1e-5), (('failure_policy', 'no_replacement_or_automatic_rerun'), False),
    (('primary_policies', 'strict', 'relative'), '1e-20'),
    (('primary_policies', 'strict', 'maximum_rhs'), 15000001),
    (('primary_policies', 'half_cap', 'maximum_step_ms'), '0.2'),
])
def test_resource_input_gate_or_authority_changes_are_rejected(protocol, path, value):
    modified = copy.deepcopy(protocol)
    target = modified
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        independent.validate_registration(modified)


@pytest.mark.parametrize('field', ['definition_source_sha256', 'metadata_auditor_sha256',
                                 'case_table_sha256', 'input_manifest_sha256', 'comparison_table_sha256'])
def test_source_or_table_pin_mismatch_not_accepted(protocol, field):
    modified = copy.deepcopy(protocol)
    modified[field] = '0'*64
    with pytest.raises(ValueError):
        independent.validate_registration(modified)


def test_typed_case_admission_labels_and_no_builder_side_effects(tmp_path):
    row = next(r for r in producer.case_table() if r['factor'] == 1)
    for update in ({'factor': True}, {'factor': 1.0}, {'profile': '../bad'}, {'extra': True}):
        with pytest.raises(ValueError):
            producer.label({**row, **update})
    with pytest.raises(ValueError):
        producer.synthetic_input(151000, 'active')  # Short pilot cannot silently become the long study.
    result = subprocess.run([sys.executable, str(ROOT/'scripts/patchseq_gbs_pilot_protocol.py'), '--plan'],
                            cwd=tmp_path, check=True, capture_output=True, text=True, timeout=30)
    metadata = json.loads(result.stdout)
    assert len(metadata['cases']) == 104 and len(metadata['comparisons']) == 120
    assert list(tmp_path.iterdir()) == []
    result = subprocess.run([sys.executable, str(ROOT/'scripts/patchseq_gbs_pilot_protocol.py'), '--execute'],
                            cwd=tmp_path, check=False, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and list(tmp_path.iterdir()) == []


def test_saved_prospective_registration_matches_source_and_independent_audit(protocol):
    saved = yaml.safe_load((ROOT/producer.REGISTRATION).read_text())
    assert saved == protocol
    assert independent.validate_registration(saved)['execution_authorized'] is False
