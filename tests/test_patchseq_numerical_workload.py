"""Synthetic metadata only: no source recordings or neuron simulations."""

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
import build_patchseq_numerical_workload as builder
import verify_patchseq_numerical_workload as auditor


def inputs():
    protocol = {'strata': ['Pvalb-L23', 'Sst-L5'], 'search': {'candidates_per_stratum': 2}}
    table = [{'C': 20., 'gL': 1.}, {'C': 40., 'gL': 2.}]
    catalog, paired = {'records': []}, {'synthetic': {'records': []}}
    for slot, (group, layer) in enumerate((('Pvalb', 'L23'), ('Sst', 'L5'))):
        record = {'partition': 'fitting', 'source_group': group, 'layer_label': layer,
                  'source_batch': 'synthetic', 'slot': slot, 'frozen_group_id': f'group-{slot}',
                  'source_cell_specimen_id': 100+slot,
                  'sweeps': [{'state': 'fitting-response-characterized', 'sweep_number': 1,
                              'features': {'detected_event_count': slot}}]}
        catalog['records'].append(record)
        paired['synthetic']['records'].append({'slot': slot, 'series': [
            {'path': 'stimulus', 'role': 'stimulus', 'data_shape': [101]},
            {'path': 'response', 'role': 'response', 'rate': 50000.}],
            'sweeps': [{'sweep_number': 1, 'series_paths': ['stimulus', 'response']}]})
    # It is intentionally impossible to resolve this held-out row's body/metadata.
    catalog['records'].append({'partition': 'held-out', 'source_batch': 'FORBIDDEN'})
    original = {'strata': {'Pvalb-L23': {'candidates': {'0': {'parameters': table[0]}}}}}
    benchmark = {'cases': {f'{b}/{f}': {'native_steps': 500, 'elapsed_s': .5*f}
                            for b in (53, 160, 320) for f in (4, 8, 16)}}
    return protocol, catalog, paired, table, original, benchmark


def test_complete_scope_and_independent_reconstruction():
    args = inputs()
    report = builder.assemble(*args)
    assert report['totals'] == {'parameter_vectors': 2, 'candidate_stratum_entries': 4,
                                'supported_groups': 2, 'training_sweeps': 2,
                                'candidate_sweep_cases': 4, 'warmup_inclusive_native_intervals': 200400}
    assert auditor.verify(report, *args) == report['totals']


@pytest.mark.parametrize('defect', ['missing_case', 'duplicate_case', 'new_case', 'parameter',
    'stratum_entry', 'group', 'target_hash', 'grid', 'storage', 'cost', 'total', 'prefix',
    'boolean_index', 'body_access', 'simulation', 'fitting', 'held_out', 'promotion', 'complete'])
def test_independent_checker_rejects_corruption(defect):
    args = inputs()
    report = builder.assemble(*args)
    if defect == 'missing_case':
        report['candidate_sweep_cases'].pop()
    elif defect == 'duplicate_case':
        report['candidate_sweep_cases'].append(report['candidate_sweep_cases'][0])
    elif defect == 'new_case':
        report['candidate_sweep_cases'][0]['source_key'] = ['unsupported', 10, 99]
    elif defect == 'parameter':
        report['parameters'] = copy.deepcopy(report['parameters'])
        report['parameters'][0]['C'] = 21.
        report['candidate_table_sha256'] = builder.identity(report['parameters'])
    elif defect == 'stratum_entry':
        report['candidate_stratum_entries'].pop()
    elif defect == 'group':
        report['sweeps'][0]['frozen_group_id'] = 'changed'
    elif defect == 'target_hash':
        report['sweeps'][0]['target_metadata_sha256'] = 'changed'
    elif defect == 'grid':
        report['sweeps'][0]['command_samples'] += 1
    elif defect == 'storage':
        report['nine_arm_five_state_uncompressed_bytes'] += 1
    elif defect == 'cost':
        report['rough_nine_arm_serial_seconds'] *= 2
    elif defect == 'total':
        report['totals']['supported_groups'] = 1
    elif defect == 'prefix':
        report['original_retained_rows'] = {}
    elif defect == 'boolean_index':
        report['candidate_stratum_entries'][1]['candidate_index'] = True
    else:
        field = {'body_access': 'recording_arrays_opened', 'simulation': 'simulations_run',
                 'fitting': 'fitting_restart_authorized', 'held_out': 'held_out_access_authorized',
                 'promotion': 'network_promotion', 'complete': 'scientific_goal_complete'}[defect]
        report[field] = True
    with pytest.raises(ValueError):
        auditor.verify(report, *args)


@pytest.mark.parametrize('defect', ['duplicate_record', 'duplicate_sweep', 'no_stratum',
                                  'roles', 'rate', 'historical_parameter'])
def test_builder_rejects_bad_metadata(defect):
    args = list(inputs())
    protocol, catalog, paired, table, original, benchmark = args
    if defect == 'duplicate_record':
        catalog['records'].insert(0, copy.deepcopy(catalog['records'][0]))
    elif defect == 'duplicate_sweep':
        catalog['records'][0]['sweeps'] *= 2
    elif defect == 'no_stratum':
        catalog['records'][0]['sweeps'] = []
    elif defect == 'roles':
        paired['synthetic']['records'][0]['series'][0]['role'] = 'response'
    elif defect == 'rate':
        paired['synthetic']['records'][0]['series'][1]['rate'] = 50000.5
    else:
        original['strata']['Pvalb-L23']['candidates']['0']['parameters'] = {'C': 21., 'gL': 1.}
    with pytest.raises(ValueError):
        builder.assemble(protocol, catalog, paired, table, original, benchmark)


def test_recording_body_suffix_rejected_before_open(tmp_path):
    with pytest.raises(ValueError, match='forbids recording'):
        builder.read_metadata(tmp_path/'do-not-open.nwb', {})


def test_metadata_digest_change_rejected(tmp_path):
    path = tmp_path/'metadata.json'
    path.write_text('{"record": 1}')
    pins = {}
    assert builder.read_metadata(path, pins) == {'record': 1}
    path.write_text('{"record": 2}')
    with pytest.raises(ValueError, match='hash differs'):
        builder.read_metadata(path, pins)
