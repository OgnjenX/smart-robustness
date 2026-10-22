"""Prospective short synthetic pilot definition. No solver/build/data reader.

The original long inputs remain mandatory for later qualification. These shorter
analogues measure the entire primary-plus-independent workflow before that cost
is authorized. A registration is not an execution seal or human approval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = 'docs/validation-results/post2008-patchseq-gbs-synthetic-pilot-registration-1515.yaml'
EXECUTION_SEAL = 'docs/validation-results/post2008-patchseq-gbs-synthetic-pilot-execution-seal-1516.yaml'
OUTPUT = 'results/patchseq-gbs-synthetic-pilot-1517'
FIELDS = ('capacitance_pF', 'leak_nS', 'leak_reversal_mV', 'sodium_nS', 'potassium_nS',
          'm_current_nS', 'threshold_shift_mV', 'm_tau_max_ms')
BASE_PARAMETERS = (100., 10., -65., 2000., 500., 10., -60., 1000.)
INITIAL = (-65., .05, .6, .2, .05)
RATE = 50000.
WARM = 5000
BASELINE = RELEASE = 2500
PRE_WINDOW = (1500, 2000)
LENGTHS = (20001, 40001)
FULL_LENGTHS = (151000, 330000)
TARGET = {'baseline_mean_mV': -65., 'early_step_mean_mV': -60., 'steady_step_mean_mV': -55.,
          'detected_event_count': 3, 'first_event_latency_s': .01, 'last_to_first_interval_ratio': 1.}


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def sha(value):
    return hashlib.sha256(encoded(value).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def policies():
    common = {'initial_step_ms': '0.02', 'minimum_step_ms': '1e-30',
              'maximum_rhs': 15_000_000, 'maximum_steps': 1_000_000, 'maximum_rejections': 100_000}
    return {'standard': {**common, 'absolute': '1e-18', 'relative': '1e-18', 'maximum_step_ms': '0.2'},
            'strict': {**common, 'absolute': '1e-22', 'relative': '1e-22', 'maximum_step_ms': '0.2'},
            'half_cap': {**common, 'absolute': '1e-22', 'relative': '1e-22', 'maximum_step_ms': '0.1'}}


def probes(samples):
    if type(samples) is not int or samples not in LENGTHS:
        raise ValueError('outside fixed pilot lengths')
    stop = samples-RELEASE
    return (WARM//2, WARM+BASELINE+1, WARM+(BASELINE+stop)//2, WARM+stop+1)


def _key(row):
    control = row['policy'] if row['engine'] == 'gbs' else 'factor'+str(row['factor'])
    endpoint = 'all' if row['stop_index'] is None else str(row['stop_index'])
    return '-'.join(map(str, (row['samples'], row['profile'], row['engine'], row['bits'],
                              control, row['role'], row['repeat'], endpoint)))


def case_table():
    rows = []
    for samples in LENGTHS:
        for profile in ('passive', 'active'):
            for bits in (160, 320):
                for policy in ('standard', 'strict', 'half_cap'):
                    rows.append({'samples': samples, 'profile': profile, 'engine': 'gbs', 'bits': bits,
                        'policy': policy, 'factor': None, 'role': 'full', 'repeat': 0, 'stop_index': None})
                for role, repeat in (('full', 1), ('checkpoint', 0)):
                    rows.append({'samples': samples, 'profile': profile, 'engine': 'gbs', 'bits': bits,
                        'policy': 'strict', 'factor': None, 'role': role, 'repeat': repeat, 'stop_index': None})
                for probe in probes(samples):
                    rows.append({'samples': samples, 'profile': profile, 'engine': 'gbs', 'bits': bits,
                        'policy': 'strict', 'factor': None, 'role': 'landing', 'repeat': 0, 'stop_index': probe})
                for factor in (1, 2, 4):
                    rows.append({'samples': samples, 'profile': profile, 'engine': 'rkf78', 'bits': bits,
                        'policy': None, 'factor': factor, 'role': 'full', 'repeat': 0, 'stop_index': None})
                rows.append({'samples': samples, 'profile': profile, 'engine': 'rkf78', 'bits': bits,
                    'policy': None, 'factor': 4, 'role': 'full', 'repeat': 1, 'stop_index': None})
    return sorted(rows, key=_key)


def validate_case(row):
    try:
        canonical = encoded(row)
    except (TypeError, ValueError) as exc:
        raise ValueError('invalid typed pilot case') from exc
    if type(row) is not dict or not any(encoded(item) == canonical for item in case_table()):
        raise ValueError('pilot case outside complete frozen support')
    return row


def label(row):
    return _key(validate_case(row))


def synthetic_input(samples, profile):
    if type(samples) is not int or samples not in LENGTHS or profile not in ('passive', 'active'):
        raise ValueError('outside fixed short-pilot input table')
    values = list(BASE_PARAMETERS)
    if profile == 'passive':
        values[3:6] = [0., 0., 0.]
    command = np.zeros(samples, dtype=np.float64)
    command[BASELINE:samples-RELEASE] = -50. if profile == 'passive' else 2000.
    current = np.concatenate((np.zeros(WARM, dtype=np.float64), command))
    return (dict(zip(FIELDS, values, strict=True)), current, np.array(INITIAL, dtype=np.float64),
            command, {'step_start_index': BASELINE, 'step_stop_index': samples-RELEASE,
                      'pre_window': list(PRE_WINDOW)})


def input_signature(row):
    validate_case(row)
    p, full, initial, command, epoch = synthetic_input(row['samples'], row['profile'])
    current = full if row['stop_index'] is None else full[:row['stop_index']+1]
    return {'parameters_binary64': {key: float(p[key]).hex() for key in FIELDS},
        'initial_binary64': [float(x).hex() for x in initial], 'rate_binary64': RATE.hex(),
        'full_current_sha256': hashlib.sha256(full.tobytes()).hexdigest(),
        'integrated_current_sha256': hashlib.sha256(current.tobytes()).hexdigest(),
        'command_sha256': hashlib.sha256(command.tobytes()).hexdigest(),
        'shape': [len(current), 5], 'epoch': epoch, 'warmup_samples': WARM,
        'feature_scoring': row['role'] != 'landing', 'uncompressed_array_bytes': len(current)*5*8}


def input_manifest():
    return {_key(row): input_signature(row) for row in case_table()}


def comparison_table():
    rows = case_table()
    comparisons = []
    def find(samples, profile, engine, bits, policy=None, factor=None, role='full', repeat=0, stop_index=None):
        chosen = [r for r in rows if r == {'samples': samples, 'profile': profile, 'engine': engine,
            'bits': bits, 'policy': policy, 'factor': factor, 'role': role, 'repeat': repeat, 'stop_index': stop_index}]
        if len(chosen) != 1:
            raise ValueError('pilot comparison support differs')
        return _key(chosen[0])
    def pair(kind, a, b, required=True, scope='full'):
        comparisons.append({'kind': kind, 'left': a, 'right': b, 'required': required, 'scope': scope})
    for n in LENGTHS:
        for profile in ('passive', 'active'):
            for bits in (160, 320):
                standard = find(n, profile, 'gbs', bits, 'standard')
                strict = find(n, profile, 'gbs', bits, 'strict')
                half = find(n, profile, 'gbs', bits, 'half_cap')
                ref = {f: find(n, profile, 'rkf78', bits, factor=f) for f in (1, 2, 4)}
                pair('primary-accuracy', standard, strict)
                pair('primary-step-cap', strict, half)
                pair('reference-coarse', ref[1], ref[2], required=False)
                pair('reference-finest', ref[2], ref[4])
                pair('independent-forward', strict, ref[4])
                pair('exact-repeat', strict, find(n, profile, 'gbs', bits, 'strict', repeat=1))
                pair('exact-repeat', ref[4], find(n, profile, 'rkf78', bits, factor=4, repeat=1))
                pair('exact-restart', strict, find(n, profile, 'gbs', bits, 'strict', role='checkpoint'))
                for probe in probes(n):
                    pair('dense-landing', strict,
                         find(n, profile, 'gbs', bits, 'strict', role='landing', stop_index=probe), scope='endpoint')
            for policy in ('standard', 'strict', 'half_cap'):
                pair('primary-precision', find(n, profile, 'gbs', 160, policy), find(n, profile, 'gbs', 320, policy))
            for factor in (1, 2, 4):
                pair('reference-precision', find(n, profile, 'rkf78', 160, factor=factor),
                     find(n, profile, 'rkf78', 320, factor=factor))
    return sorted(comparisons, key=lambda item: (item['kind'], item['left'], item['right']))


def document():
    manifest = input_manifest()
    return {'schema_version': 1, 'artifact_id': 1515, 'date': '2026-10-22',
        'status': 'prospective-short-pilot-fixed-not-execution-authorized',
        'parent_plan': 'docs/validation-results/post2008-patchseq-segment-adaptive-plan-1511.yaml',
        'preparation': 'docs/validation-results/post2008-patchseq-gbs-hh-reference-preparation-1514.yaml',
        'definition_source': 'scripts/patchseq_gbs_pilot_protocol.py',
        'definition_source_sha256': file_sha(__file__),
        'metadata_auditor': 'scripts/verify_patchseq_gbs_pilot_protocol.py',
        'metadata_auditor_sha256': file_sha(ROOT/'scripts/verify_patchseq_gbs_pilot_protocol.py'),
        'required_runner': 'scripts/run_patchseq_gbs_pilot.py',
        'required_array_auditor': 'scripts/verify_patchseq_gbs_pilot.py',
        'required_execution_seal': EXECUTION_SEAL, 'output': OUTPUT,
        'total_cases': len(case_table()), 'case_table_sha256': sha(case_table()),
        'input_manifest_sha256': sha(manifest), 'comparison_count': len(comparison_table()),
        'comparison_table_sha256': sha(comparison_table()),
        'inputs': {'command_lengths': list(LENGTHS), 'rate_hz': RATE, 'warmup_samples': WARM,
            'baseline_samples': BASELINE, 'release_samples': RELEASE, 'pre_window': list(PRE_WINDOW),
            'field_order': list(FIELDS), 'active_parameters': list(BASE_PARAMETERS), 'initial': list(INITIAL),
            'active_step_pA': 2000., 'passive_step_pA': -50., 'holding_pA': 0.,
            'passive_parameters': 'zero-sodium-potassium-and-M-conductances',
            'parent_full_command_lengths': list(FULL_LENGTHS),
            'scope': 'shortened-analogues-not-replacements-for-all-four-full1510profiles'},
        'primary_policies': policies(), 'independent_factors': [1, 2, 4], 'synthetic_target': dict(TARGET),
        'restart_control_native_index': 4096,
        'landing_probes': {str(n): list(probes(n)) for n in LENGTHS},
        'comparisons': {'whole_voltage_mV': 1e-5, 'feature_float_absolute': 1e-5,
            'feature_float_relative': 1e-9, 'feature_discrete_and_structure': 'exact-types-values-and-support',
            'loss_absolute': 1e-6, 'loss_relative': 1e-8,
            'repeat_and_restart': 'bitwise-all-five-states-on-the-entire-native-grid',
            'dense_landing': 'full-run-observation-at-declared-index-versus-separate-accepted-prefix-endpoint',
            'reference_coarse': 'report-all-failures-not-the-finest-reference-gate',
            'all_other_comparisons': 'mandatory-no-favorable-subset',
            'synthetic_fit_target_gates': 'report-only-not-a-biological-fit-or-numerical-promotion-gate'},
        'resources': {'maximum_invocation_seconds': 28800, 'maximum_total_controller_seconds': 25200,
            'maximum_controller_plus_audit_seconds': 28800,
            'maximum_case_seconds': 600, 'cleanup_reserve_seconds': 60,
            'maximum_independent_audit_seconds': 3600, 'maximum_output_bytes': 1073741824,
            'maximum_independent_rhs': 15000000, 'maximum_independent_substeps': 1000000,
            'total_native_array_uncompressed_bytes': sum(x['uncompressed_array_bytes'] for x in manifest.values()),
            'concurrent_workers': 1, 'external_or_paid_compute': False,
            'incomplete_or_failed_pilot': 'retain-all-identities-and-artifacts-no-qualification-or-automatic-budget-expansion'},
        'failure_policy': {'stop_on_execution_failure': True, 'no_replacement_or_automatic_rerun': True,
            'interrupted_controller_or_worker': 'block-automatic-resume-preserve-evidence-and-inspect-real-process-state',
            'completed_case_resume': 'only-after-clean-controller-boundary-with-identical-pins-builds-runtime-and-artifacts'},
        'boundaries': {'execution_authorized': False, 'scientific_goal_complete': False,
            'biological_or_held_out_access': False, 'fitting_restart': False, 'network_promotion': False,
            'original_score_replacement': False, 'frozen_baseline_change': False,
            'long_reference_qualification': False, 'whole_biological_study_feasibility': False},
        'next_gate': 'independent-array-audit-and-explicit-execution-decision-before-seal-no-simulation-from-this-document'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', action='store_true', help='Print all case/input/comparison metadata, no execution')
    args = parser.parse_args()
    payload = {'cases': case_table(), 'inputs': input_manifest(), 'comparisons': comparison_table()} if args.plan else document()
    print(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False))
