"""Independent pilot metadata/support audit. Never imports its producer or solvers.

This validates prospective identities, input bytes and unchanged gate contracts;
it does not audit simulated arrays or authorize their execution.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = ROOT/'docs/validation-results/post2008-patchseq-gbs-synthetic-pilot-registration-1515.yaml'


def require(condition):
    if not condition:
        raise ValueError('independent prospective pilot support or contract differs')


def canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError('invalid typed pilot metadata') from exc


def sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def case_name(row):
    mode = row['policy'] if row['engine'] == 'gbs' else 'factor'+str(row['factor'])
    stop = str(row['stop_index']) if row['stop_index'] is not None else 'all'
    return f"{row['samples']}-{row['profile']}-{row['engine']}-{row['bits']}-{mode}-{row['role']}-{row['repeat']}-{stop}"


def expected_rows():
    result = []
    for n, kind in ((20001, 'passive'), (20001, 'active'), (40001, 'passive'), (40001, 'active')):
        endpoints = (2500, 7501, 5000+n//2, 5000+n-2500+1)
        for width in (160, 320):
            def append(engine, policy=None, factor=None, role='full', repeat=0, stop=None, n=n, kind=kind, width=width):
                result.append({'samples': n, 'profile': kind, 'engine': engine, 'bits': width, 'policy': policy,
                               'factor': factor, 'role': role, 'repeat': repeat, 'stop_index': stop})
            for mode in ('standard', 'strict', 'half_cap'):
                append('gbs', policy=mode)
            append('gbs', policy='strict', repeat=1)
            append('gbs', policy='strict', role='checkpoint')
            for index in endpoints:
                append('gbs', policy='strict', role='landing', stop=index)
            for divisor in (1, 2, 4):
                append('rkf78', factor=divisor)
            append('rkf78', factor=4, repeat=1)
    return sorted(result, key=case_name)


def expected_inputs(rows):
    fields = ['capacitance_pF', 'leak_nS', 'leak_reversal_mV', 'sodium_nS', 'potassium_nS',
              'm_current_nS', 'threshold_shift_mV', 'm_tau_max_ms']
    result = {}
    source = {}
    for row in rows:
        n, profile = row['samples'], row['profile']
        key = n, profile
        if key not in source:
            values = [100., 10., -65., 2000., 500., 10., -60., 1000.]
            if profile == 'passive':
                for index in (3, 4, 5):
                    values[index] = 0.
            # Independently generate the declared mathematical command, not a
            # saved worker input or a call to the producer's helper.
            amplitude = {'passive': -50., 'active': 2000.}[profile]
            command = np.array([amplitude if 2500 <= j < n-2500 else 0. for j in range(n)], dtype=np.float64)
            full = np.array([0.]*5000+command.tolist(), dtype=np.float64)
            source[key] = values, command, full
        values, command, full = source[key]
        stop = row['stop_index']
        executed = full if stop is None else full[:stop+1]
        result[case_name(row)] = {
            'parameters_binary64': {field: value.hex() for field, value in zip(fields, values, strict=True)},
            'initial_binary64': [value.hex() for value in [-65., .05, .6, .2, .05]],
            'rate_binary64': (50000.).hex(),
            'full_current_sha256': hashlib.sha256(full.tobytes()).hexdigest(),
            'integrated_current_sha256': hashlib.sha256(executed.tobytes()).hexdigest(),
            'command_sha256': hashlib.sha256(command.tobytes()).hexdigest(),
            'shape': [len(executed), 5], 'epoch': {'step_start_index': 2500, 'step_stop_index': n-2500,
                                                'pre_window': [1500, 2000]},
            'warmup_samples': 5000, 'feature_scoring': row['role'] != 'landing',
            'uncompressed_array_bytes': len(executed)*40}
    return result


def expected_comparisons(rows):
    result = []
    def add(kind, a, b, required=True, scope='full'):
        result.append({'kind': kind, 'left': case_name(a), 'right': case_name(b), 'required': required, 'scope': scope})
    for n in (20001, 40001):
        for profile in ('passive', 'active'):
            group = [r for r in rows if r['samples'] == n and r['profile'] == profile]
            def choose(_group=group, **parts):
                selected = [r for r in _group if all(r[key] == value for key, value in parts.items())]
                require(len(selected) == 1)
                return selected[0]
            for width in (160, 320):
                gbs = {policy: choose(engine='gbs', bits=width, policy=policy, role='full', repeat=0)
                       for policy in ('standard', 'strict', 'half_cap')}
                ref = {factor: choose(engine='rkf78', bits=width, factor=factor, repeat=0) for factor in (1, 2, 4)}
                add('primary-accuracy', gbs['standard'], gbs['strict'])
                add('primary-step-cap', gbs['strict'], gbs['half_cap'])
                add('reference-coarse', ref[1], ref[2], required=False)
                add('reference-finest', ref[2], ref[4])
                add('independent-forward', gbs['strict'], ref[4])
                add('exact-repeat', gbs['strict'], choose(engine='gbs', bits=width, repeat=1))
                add('exact-repeat', ref[4], choose(engine='rkf78', bits=width, repeat=1))
                add('exact-restart', gbs['strict'], choose(engine='gbs', bits=width, role='checkpoint'))
                for landing in [r for r in group if r['engine'] == 'gbs' and r['bits'] == width and r['role'] == 'landing']:
                    add('dense-landing', gbs['strict'], landing, scope='endpoint')
            for mode in ('standard', 'strict', 'half_cap'):
                add('primary-precision', choose(engine='gbs', bits=160, policy=mode, role='full', repeat=0),
                    choose(engine='gbs', bits=320, policy=mode, role='full', repeat=0))
            for divisor in (1, 2, 4):
                add('reference-precision', choose(engine='rkf78', bits=160, factor=divisor, repeat=0),
                    choose(engine='rkf78', bits=320, factor=divisor, repeat=0))
    return sorted(result, key=lambda row: (row['kind'], row['left'], row['right']))


def validate_tables(rows, inputs, comparisons):
    expected = expected_rows()
    require(canonical(rows) == canonical(expected))
    require(canonical(inputs) == canonical(expected_inputs(expected)))
    require(canonical(comparisons) == canonical(expected_comparisons(expected)))
    return {'case_table_sha256': sha(rows), 'input_manifest_sha256': sha(inputs),
            'comparison_table_sha256': sha(comparisons), 'total_cases': len(rows),
            'comparison_count': len(comparisons)}


def validate_registration(data):
    require(type(data) is dict)
    expected_keys = {'schema_version', 'artifact_id', 'date', 'status', 'parent_plan', 'preparation',
        'definition_source', 'definition_source_sha256', 'metadata_auditor', 'metadata_auditor_sha256',
        'required_runner', 'required_array_auditor', 'required_execution_seal', 'output', 'total_cases',
        'case_table_sha256', 'input_manifest_sha256', 'comparison_count', 'comparison_table_sha256',
        'inputs', 'primary_policies', 'independent_factors', 'synthetic_target', 'restart_control_native_index',
        'landing_probes', 'comparisons', 'resources', 'failure_policy', 'boundaries', 'next_gate'}
    require(set(data) == expected_keys)
    require(canonical([data['schema_version'], data['artifact_id'], data['date'], data['status']]) ==
            canonical([1, 1515, '2026-10-22', 'prospective-short-pilot-fixed-not-execution-authorized']))
    require(data['parent_plan'] == 'docs/validation-results/post2008-patchseq-segment-adaptive-plan-1511.yaml')
    require(data['preparation'] == 'docs/validation-results/post2008-patchseq-gbs-hh-reference-preparation-1514.yaml')
    paths = {'definition_source': 'scripts/patchseq_gbs_pilot_protocol.py',
             'metadata_auditor': 'scripts/verify_patchseq_gbs_pilot_protocol.py'}
    for key, expected in paths.items():
        require(data[key] == expected)
        require(data[key+'_sha256'] == hashlib.sha256((ROOT/expected).read_bytes()).hexdigest())
    for key, expected in {'required_runner': 'scripts/run_patchseq_gbs_pilot.py',
        'required_array_auditor': 'scripts/verify_patchseq_gbs_pilot.py',
        'required_execution_seal': 'docs/validation-results/post2008-patchseq-gbs-synthetic-pilot-execution-seal-1516.yaml',
        'output': 'results/patchseq-gbs-synthetic-pilot-1517'}.items():
        require(data[key] == expected)
    rows = expected_rows()
    inputs = expected_inputs(rows)
    pairs = expected_comparisons(rows)
    expected_inputs_contract = {'command_lengths': [20001, 40001], 'rate_hz': 50000., 'warmup_samples': 5000,
        'baseline_samples': 2500, 'release_samples': 2500, 'pre_window': [1500, 2000],
        'field_order': ['capacitance_pF', 'leak_nS', 'leak_reversal_mV', 'sodium_nS', 'potassium_nS',
                        'm_current_nS', 'threshold_shift_mV', 'm_tau_max_ms'],
        'active_parameters': [100., 10., -65., 2000., 500., 10., -60., 1000.],
        'initial': [-65., .05, .6, .2, .05], 'active_step_pA': 2000., 'passive_step_pA': -50., 'holding_pA': 0.,
        'passive_parameters': 'zero-sodium-potassium-and-M-conductances',
        'parent_full_command_lengths': [151000, 330000],
        'scope': 'shortened-analogues-not-replacements-for-all-four-full1510profiles'}
    require(canonical(data['inputs']) == canonical(expected_inputs_contract))
    expected_policy = {}
    for name, tolerance, cap in [('standard', '1e-18', '0.2'), ('strict', '1e-22', '0.2'), ('half_cap', '1e-22', '0.1')]:
        expected_policy[name] = {'absolute': tolerance, 'relative': tolerance, 'maximum_step_ms': cap,
            'initial_step_ms': '0.02', 'minimum_step_ms': '1e-30', 'maximum_rhs': 15000000,
            'maximum_steps': 1000000, 'maximum_rejections': 100000}
    require(canonical(data['primary_policies']) == canonical(expected_policy))
    require(canonical(data['independent_factors']) == '[1,2,4]')
    require(canonical(data['synthetic_target']) == canonical({'baseline_mean_mV': -65., 'early_step_mean_mV': -60.,
        'steady_step_mean_mV': -55., 'detected_event_count': 3, 'first_event_latency_s': .01,
        'last_to_first_interval_ratio': 1.}))
    require(canonical(data['restart_control_native_index']) == '4096')
    require(canonical(data['landing_probes']) == canonical({'20001': [2500, 7501, 15000, 22502],
                                                          '40001': [2500, 7501, 25000, 42502]}))
    require(canonical(data['comparisons']) == canonical({'whole_voltage_mV': 1e-5,
        'feature_float_absolute': 1e-5, 'feature_float_relative': 1e-9,
        'feature_discrete_and_structure': 'exact-types-values-and-support', 'loss_absolute': 1e-6, 'loss_relative': 1e-8,
        'repeat_and_restart': 'bitwise-all-five-states-on-the-entire-native-grid',
        'dense_landing': 'full-run-observation-at-declared-index-versus-separate-accepted-prefix-endpoint',
        'reference_coarse': 'report-all-failures-not-the-finest-reference-gate',
        'all_other_comparisons': 'mandatory-no-favorable-subset',
        'synthetic_fit_target_gates': 'report-only-not-a-biological-fit-or-numerical-promotion-gate'}))
    require(canonical(data['resources']) == canonical({'maximum_invocation_seconds': 28800,
        'maximum_total_controller_seconds': 25200, 'maximum_controller_plus_audit_seconds': 28800,
        'maximum_case_seconds': 600, 'cleanup_reserve_seconds': 60,
        'maximum_independent_audit_seconds': 3600, 'maximum_output_bytes': 1073741824,
        'maximum_independent_rhs': 15000000, 'maximum_independent_substeps': 1000000,
        'total_native_array_uncompressed_bytes': sum(item['uncompressed_array_bytes'] for item in inputs.values()),
        'concurrent_workers': 1, 'external_or_paid_compute': False,
        'incomplete_or_failed_pilot': 'retain-all-identities-and-artifacts-no-qualification-or-automatic-budget-expansion'}))
    require(canonical(data['failure_policy']) == canonical({'stop_on_execution_failure': True,
        'no_replacement_or_automatic_rerun': True,
        'interrupted_controller_or_worker': 'block-automatic-resume-preserve-evidence-and-inspect-real-process-state',
        'completed_case_resume': 'only-after-clean-controller-boundary-with-identical-pins-builds-runtime-and-artifacts'}))
    require(set(data['boundaries']) == {'execution_authorized', 'scientific_goal_complete', 'biological_or_held_out_access',
        'fitting_restart', 'network_promotion', 'original_score_replacement', 'frozen_baseline_change',
        'long_reference_qualification', 'whole_biological_study_feasibility'})
    require(all(value is False for value in data['boundaries'].values()))
    require(data['next_gate'] == 'independent-array-audit-and-explicit-execution-decision-before-seal-no-simulation-from-this-document')
    summary = validate_tables(rows, inputs, pairs)
    for key, expected in summary.items():
        require(canonical(data[key]) == canonical(expected))
    return {**summary, 'execution_authorized': False, 'numerical_validation_performed': False}


if __name__ == '__main__':
    print(json.dumps(validate_registration(yaml.safe_load(REGISTRATION.read_text())), sort_keys=True))
