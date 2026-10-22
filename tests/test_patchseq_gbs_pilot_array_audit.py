"""Adverse/fabricated numerical-evidence fixtures; no pilot or solver execution."""

from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
import patchseq_gbs_pilot_array_audit as independent
from patchseq_fit_objective import score_sweep
from patchseq_response_features import characterize
from verify_patchseq_gbs_pilot_protocol import REGISTRATION


def checksum(document):
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def exact(value, width):
    negative = type(value) is float and value == 0. and math.copysign(1., value) < 0
    number = Fraction(value)
    if not number:
        return '-0p0' if negative else '0p0'
    n, d = number.numerator, number.denominator
    assert d & (d-1) == 0
    trailing = (abs(n) & -abs(n)).bit_length()-1
    n >>= trailing
    padding = width-abs(n).bit_length()
    assert padding >= 0
    return ('-' if n < 0 else '')+format(abs(n) << padding, 'x')+'p'+str(trailing-(d.bit_length()-1)-padding)


@pytest.fixture(scope='module')
def audit():
    return independent.ArrayAudit(yaml.safe_load(REGISTRATION.read_text()),
        {'gbs': {'fabricated': 'not a loaded primary library'},
         'rkf78': {'fabricated': 'not a loaded independent library'}})


def envelope(audit, key, states, index, stats):
    row, signature = audit.rows[key], audit.manifest[key]
    configuration = {'kind': 'gbs-hh-primary-5-v1', 'bits': row['bits'],
        'policy': audit.reg['primary_policies'][row['policy']], 'rate_binary64': float(50000).hex(),
        'samples': len(states), 'current_sha256': signature['integrated_current_sha256'],
        'initial': [v.hex() for v in [-65., .05, .6, .2, .05]],
        'parameters': [signature['parameters_binary64'][name] for name in audit.reg['inputs']['field_order']],
        'build_sha256': checksum(audit.builds['gbs']), 'barrier_intervals': 4096,
        'environment': {'mpfr': '4.2.2', 'header': '4.2.2', 'gmp': '6.3.0', 'tls': 1,
            'emin': -1073741823, 'emax': 1073741823, 'precision160': 160, 'precision320': 320,
            'cancellation160': 1, 'cancellation320': 1}}
    fields = ['gbs-guarded-core-v1', str(row['bits']), '5', checksum(configuration), str(index),
        exact(index*Fraction.from_float(.02), row['bits']),
        *[exact(float(v), row['bits']) for v in states[index]],
        *[str(stats[name]) for name in ('rhs', 'accepted', 'rejected', 'dense')]]
    record = {'schema': 'gbs-hh-checkpoint-v1', 'configuration': configuration, 'payload': '\n'.join(fields)+'\n'}
    return {**record, 'sha256': checksum(record)}


def fixture(audit, key):
    """Closed-form voltage or constant active voltage, never an HH trajectory."""
    row, signature = audit.rows[key], audit.manifest[key]
    size = signature['shape'][0]
    states = np.tile(np.array([-65., .05, .6, .2, .05]), (size, 1))
    if row['profile'] == 'passive':
        # Independently constructed scalar RC fixture; no gates are integrated.
        onset, release = 7500, row['samples']+2500
        for index in range(onset, size):
            if index < release:
                states[index, 0] = -70.+5.*math.exp(-(index-onset)*.002)
            else:
                states[index, 0] = -65.+5.*math.expm1(-(release-onset)*.002)*math.exp(-(index-release)*.002)
    index = size-1
    multiplier = 2 if row['engine'] == 'gbs' else 13*row['factor']
    stats = {'position': index, 'rhs': multiplier*index,
        'accepted': index if row['engine'] == 'gbs' else row['factor']*index, 'rejected': 0, 'dense': 0}
    if row['engine'] == 'gbs':
        final = envelope(audit, key, states, index, stats)
        embedded = None
    else:
        final = {'kind': 'rkf78-exact-final', 'values': [exact(float(v), row['bits']) for v in states[-1]]}
        embedded = [0.]*5
    receipt = {'schema_version': 1, 'case_key': key, 'case': copy.deepcopy(row),
        'input_signature': copy.deepcopy(signature), 'shape': [size, 5], 'arrays_sha256': '0'*64,
        'uncompressed_bytes': states.nbytes, 'compressed_bytes': 123,
        'features': None, 'score': None, 'final_state': final, 'solver_statistics': stats,
        'maximum_embedded_estimate': embedded, 'driver_open_seconds': .1, 'forward_call_seconds': .2,
        'checkpoint_serialization_seconds': 0., 'postprocess_seconds': .1, 'compression_seconds': .1,
        'peak_process_RSS_bytes': 1000000, 'scientific_goal_complete': False,
        'biological_access': False, 'fitting_restart': False}
    if row['role'] != 'landing':
        n = row['samples']
        command = np.zeros(n)
        command[2500:n-2500] = -50. if row['profile'] == 'passive' else 2000.
        features = characterize(states[5000:, 0], command, 50000.,
            {'step_start_index': 2500, 'step_stop_index': n-2500, 'pre_window': [1500, 2000]}, 0.)
        receipt['features'], receipt['score'] = features, score_sweep(audit.reg['synthetic_target'], features)
    restart = None
    if row['role'] == 'checkpoint':
        restart = envelope(audit, key, states, 4096, {'rhs': 8192, 'accepted': 4096, 'rejected': 0, 'dense': 0})
        receipt['checkpoint_serialization_seconds'] = .01
    return states, receipt, restart


def choose(audit, **parts):
    return next(key for key, row in audit.rows.items() if all(row[name] == value for name, value in parts.items()))


def test_module_has_no_forward_producer_controller_build_or_recording_calls():
    modules = []
    for node in ast.walk(ast.parse(inspect.getsource(independent))):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.append(node.module)
    assert not any(name.startswith(('patchseq_', 'run_patchseq')) for name in modules)
    assert not any(name in modules for name in ('ctypes', 'subprocess', 'h5py'))


@pytest.mark.parametrize('width', [160, 320])
@pytest.mark.parametrize('value', [-65., .02, -1e-8, 1.+1e-8, 0., -0., 1e-300])
def test_independent_exact_decoder_preserves_value_bits(width, value):
    decoded = independent.Dyadic.parse(exact(value, width), width)
    assert decoded.compare(independent.Dyadic.from_float(value)) == 0
    assert np.float64(decoded.export()).view(np.uint64) == np.float64(value).view(np.uint64)


@pytest.mark.parametrize('bad', ['+0p0', '00p0', '0p-0', '0p+0', '0p1', 'NaN', '1p0',
    '8'+'0'*39+'p+0', '8'+'0'*39+'p-0', '8'+'0'*39+'p01', 'F'*40+'p0',
    '8'+'0'*39+'p1073741824', '8'+'0'*39+'p1073741823', '8'+'0'*39+'p-1073741984'])
def test_malformed_precision_width_exponent_and_spelling_are_rejected(bad):
    with pytest.raises(ValueError):
        independent.Dyadic.parse(bad, 160)


def test_extreme_exponents_compare_without_enormous_integer_shifts():
    tiny = independent.Dyadic.parse('8'+'0'*39+'p-1073741983', 160)
    huge = independent.Dyadic.parse('8'+'0'*39+'p1073741663', 160)
    assert tiny.compare(independent.Dyadic.from_float(1.)) < 0 and tiny.export() == 0.
    assert huge.compare(tiny) > 0
    assert independent.Dyadic(-1, huge.exponent).compare(independent.Dyadic.from_float(-1.)) < 0
    with pytest.raises(ValueError, match='overflows'):
        huge.export()


@pytest.mark.parametrize('width', [160, 320])
def test_dyadic_sign_scale_and_cancellation_comparisons_against_rational_arithmetic(width):
    rng = np.random.default_rng(7123)
    for _ in range(200):
        a, b = map(int, rng.integers(-(2**50), 2**50, size=2))
        ea, eb = map(int, rng.integers(-2000, 2000, size=2))
        left, right = independent.Dyadic(a, ea), independent.Dyadic(b, eb)
        rational_left = Fraction(a)*Fraction(2)**ea
        rational_right = Fraction(b)*Fraction(2)**eb
        expected = (rational_left > rational_right)-(rational_left < rational_right)
        assert left.compare(right) == expected
        # Canonical fixed-width spelling is checked independently when the
        # exact rational has a significand small enough for the target width.
        assert independent.Dyadic.parse(exact(rational_left, width), width).compare(left) == 0


@pytest.mark.parametrize('lower', [False, True])
def test_gate_outside_full_precision_bound_is_not_hidden_by_binary64_rounding(lower):
    width = 160
    gate = -1e-8 if lower else 1.+1e-8
    outside = Fraction.from_float(gate)+(-1 if lower else 1)*Fraction(1, 2**130)
    observed = np.array([-65., gate, .6, .2, .05])
    encoded = [exact(float(v), width) for v in observed]
    encoded[1] = exact(outside, width)
    assert independent.Dyadic.parse(encoded[1], width).export() == gate
    with pytest.raises(ValueError, match='full-precision accepted gate'):
        independent.exact_states(encoded, width, observed)


@pytest.mark.parametrize('samples', [20001, 40001])
@pytest.mark.parametrize('engine', ['gbs', 'rkf78'])
@pytest.mark.parametrize('profile', ['active', 'passive'])
@pytest.mark.parametrize('bits', [160, 320])
def test_fabricated_complete_trace_features_losses_and_exact_final_support(audit, samples, engine, profile, bits):
    key = choose(audit, samples=samples, engine=engine, profile=profile, bits=bits, role='full')
    states, receipt, restart = fixture(audit, key)
    report = audit.inspect(key, states, receipt, restart=restart)
    assert report['passed'] and report['checks']['features_reconstructed']
    assert report['checks']['six_term_score_reconstructed']
    if profile == 'passive':
        assert report['passive_maximum_voltage_error_mV'] < 1e-10
    assert report['score']['all_applicable_gates_pass'] is False  # Arbitrary target not a qualification gate.


@pytest.mark.parametrize('role', ['landing', 'checkpoint'])
@pytest.mark.parametrize('bits', [160, 320])
def test_fabricated_landing_or_disk_restart_has_full_index_and_payload_support(audit, role, bits):
    key = choose(audit, role=role, bits=bits, profile='passive')
    states, receipt, restart = fixture(audit, key)
    report = audit.inspect(key, states, receipt, restart=restart)
    assert report['passed']
    assert (report['features'] is None) is (role == 'landing')


@pytest.mark.parametrize('defect', ['dtype', 'shape', 'nonfinite', 'initial_bits', 'gate', 'strided'])
def test_whole_arrays_cannot_be_replaced_by_shape_correct_summaries(audit, defect):
    key = choose(audit, role='full', engine='gbs')
    states, receipt, _restart = fixture(audit, key)
    if defect == 'dtype':
        states = states.astype(np.float32)
    elif defect == 'shape':
        states = states[:-1]
    elif defect == 'nonfinite':
        states[10, 0] = np.nan
    elif defect == 'initial_bits':
        states[0, 1] = -.05
    elif defect == 'gate':
        states[10, 2] = -1e-7
    else:
        states = np.asfortranarray(states)
    with pytest.raises(ValueError):
        audit.inspect(key, states, receipt)


@pytest.mark.parametrize('defect', ['checksum', 'configuration', 'payload_identity', 'index', 'time',
    'state', 'counts', 'noncanonical_integer', 'truncated', 'zero_rhs'])
def test_checkpoint_cannot_claim_another_input_build_policy_state_time_or_work(audit, defect):
    key = choose(audit, role='full', engine='gbs', bits=160)
    states, receipt, _restart = fixture(audit, key)
    final = receipt['final_state']
    fields = final['payload'].splitlines()
    if defect == 'checksum':
        final['sha256'] = '0'*64
    elif defect == 'configuration':
        final['configuration']['build_sha256'] = '0'*64
    elif defect == 'payload_identity':
        fields[3] = '0'*64
    elif defect == 'index':
        fields[4] = str(len(states)-2)
    elif defect == 'time':
        fields[5] = exact((len(states)-1)*.02, 160)  # Double-rounded product, not the exact native time.
        if independent.Dyadic.parse(fields[5], 160).compare(
                independent.Dyadic.from_float((len(states)-1)*.02)) != 0:
            pytest.fail('invalid fixture')
    elif defect == 'state':
        fields[6] = exact(-64., 160)
    elif defect == 'counts':
        fields[11] = str(receipt['solver_statistics']['rhs']+1)
    elif defect == 'noncanonical_integer':
        fields[11] = '00'
    elif defect == 'truncated':
        fields.pop()
    else:
        fields[11] = '0'
    if defect not in ('checksum', 'configuration'):
        final['payload'] = '\n'.join(fields)+'\n'
        final['sha256'] = checksum({k: v for k, v in final.items() if k != 'sha256'})
    with pytest.raises(ValueError):
        audit.inspect(key, states, receipt)


@pytest.mark.parametrize('defect', ['missing', 'wrong_index', 'reset_counts', 'extra_for_full'])
def test_restart_control_cannot_be_dropped_or_reset_work_counters(audit, defect):
    key = choose(audit, role='full' if defect == 'extra_for_full' else 'checkpoint', engine='gbs')
    states, receipt, restart = fixture(audit, key)
    if defect == 'missing':
        restart = None
    elif defect == 'wrong_index':
        fields = restart['payload'].splitlines()
        fields[4] = '4095'
        restart['payload'] = '\n'.join(fields)+'\n'
        restart['sha256'] = checksum({k: v for k, v in restart.items() if k != 'sha256'})
    elif defect == 'reset_counts':
        final = receipt['final_state']
        fields = final['payload'].splitlines()
        fields[11:13] = ['200', '100']
        final['payload'] = '\n'.join(fields)+'\n'
        final['sha256'] = checksum({k: v for k, v in final.items() if k != 'sha256'})
        receipt['solver_statistics'].update(rhs=200, accepted=100)
    else:
        restart = copy.deepcopy(receipt['final_state'])
    with pytest.raises(ValueError):
        audit.inspect(key, states, receipt, restart=restart)


@pytest.mark.parametrize('field', ['baseline_mean_mV', 'detected_event_count', 'first_event_latency_s',
    'event_candidate_indices', 'resistance_proxy_state'])
def test_untrusted_feature_summary_disagreement_is_retained_as_a_failed_check(audit, field):
    key = choose(audit, role='full', engine='gbs')
    states, receipt, _restart = fixture(audit, key)
    receipt['features'][field] = 'changed-worker-summary'
    report = audit.inspect(key, states, receipt)
    assert report['checks']['features_reconstructed'] is False and report['passed'] is False


@pytest.mark.parametrize('field', ['loss', 'terms', 'gates', 'all_applicable_gates_pass'])
def test_all_six_terms_and_gate_support_are_independently_reconstructed(audit, field):
    key = choose(audit, role='full', engine='rkf78')
    states, receipt, _restart = fixture(audit, key)
    receipt['score'][field] = 'changed-score-summary'
    report = audit.inspect(key, states, receipt)
    assert report['checks']['six_term_score_reconstructed'] is False and report['passed'] is False


@pytest.mark.parametrize('defect', ['extra', 'missing', 'case_type', 'shape_type', 'signature',
    'authority', 'timing', 'timing_type', 'memory', 'storage', 'hash', 'unexpected_restart_time'])
def test_receipt_claims_identity_storage_and_types_are_not_trusted(audit, defect):
    key = choose(audit, role='full', engine='rkf78')
    states, receipt, _restart = fixture(audit, key)
    if defect == 'extra':
        receipt['extra'] = True
    elif defect == 'missing':
        receipt.pop('arrays_sha256')
    elif defect == 'case_type':
        receipt['case']['repeat'] = False
    elif defect == 'shape_type':
        receipt['shape'][1] = 5.
    elif defect == 'signature':
        receipt['input_signature']['integrated_current_sha256'] = '0'*64
    elif defect == 'authority':
        receipt['fitting_restart'] = 0
    elif defect == 'timing':
        receipt['compression_seconds'] = -1.
    elif defect == 'timing_type':
        receipt['compression_seconds'] = 0
    elif defect == 'memory':
        receipt['peak_process_RSS_bytes'] = True
    elif defect == 'storage':
        receipt['compressed_bytes'] = 2**30
    elif defect == 'hash':
        receipt['arrays_sha256'] = 'invalid'
    else:
        receipt['checkpoint_serialization_seconds'] = 1.
    with pytest.raises(ValueError):
        audit.inspect(key, states, receipt)


@pytest.mark.parametrize('defect', ['factor_count', 'missing_count', 'bool_count', 'negative', 'extra',
    'embedded_missing', 'embedded_type', 'embedded_negative', 'exact_state', 'exact_width'])
def test_reference_work_diagnostics_and_final_precision_are_independent_checks(audit, defect):
    key = choose(audit, role='full', engine='rkf78', bits=320)
    states, receipt, _restart = fixture(audit, key)
    stats = receipt['solver_statistics']
    if defect == 'factor_count':
        stats['accepted'] -= 1
    elif defect == 'missing_count':
        stats.pop('dense')
    elif defect == 'bool_count':
        stats['rejected'] = False
    elif defect == 'negative':
        stats['rhs'] = -1
    elif defect == 'extra':
        stats['ignored_work'] = 100
    elif defect == 'embedded_missing':
        receipt['maximum_embedded_estimate'] = None
    elif defect == 'embedded_type':
        receipt['maximum_embedded_estimate'][0] = 0
    elif defect == 'embedded_negative':
        receipt['maximum_embedded_estimate'][0] = -1.
    elif defect == 'exact_state':
        receipt['final_state']['values'][0] = exact(-64., 320)
    else:
        receipt['final_state']['values'][0] = exact(-65., 160)
    with pytest.raises(ValueError):
        audit.inspect(key, states, receipt)


def test_passive_whole_waveform_failure_remains_visible_even_when_features_agree(audit):
    key = choose(audit, role='full', engine='gbs', profile='passive')
    states, receipt, _restart = fixture(audit, key)
    states[100, 0] += 1e-4
    report = audit.inspect(key, states, receipt)
    assert report['checks']['features_reconstructed'] is True
    assert report['checks']['six_term_score_reconstructed'] is True
    assert report['checks']['passive_analytic_voltage_agreement'] is False
    assert report['passed'] is False


def test_landing_prefix_is_not_given_fabricated_complete_response_features(audit):
    key = choose(audit, role='landing')
    states, receipt, _restart = fixture(audit, key)
    receipt['features'] = {'detected_event_count': 0}
    with pytest.raises(ValueError, match='prefix was scored'):
        audit.inspect(key, states, receipt)


@pytest.mark.parametrize('left,right,expected', [({'n': 0}, {'n': False}, False),
    ([1.], [1], False), ({'a': None}, {'a': 0.}, False), ({'a': 0.}, {'a': 0., 'b': 1}, False),
    (1., 1.+1e-6, True), (1., 1.+1e-4, False)])
def test_feature_agreement_keeps_exact_types_structure_and_fixed_float_bounds(left, right, expected):
    assert independent.typed_agree(left, right) is expected


@pytest.mark.parametrize('pair_kind', ['primary-accuracy', 'primary-step-cap', 'reference-coarse',
    'reference-finest', 'independent-forward', 'exact-repeat', 'exact-restart',
    'dense-landing', 'primary-precision', 'reference-precision'])
def test_all_registered_comparison_kinds_use_the_complete_declared_scope(audit, pair_kind):
    pair = next(p for p in audit.pairs if p['kind'] == pair_kind)
    values = []
    for key in (pair['left'], pair['right']):
        states, receipt, restart = fixture(audit, key)
        values.append((states, audit.inspect(key, states, receipt, restart=restart)))
    (left, lr), (right, rr) = values
    result = audit.compare(pair, left, right, lr, rr)
    assert result['passed'] and result['required'] is (pair_kind != 'reference-coarse')
    if pair_kind in ('exact-repeat', 'exact-restart'):
        assert result['bitwise_five_state_agreement'] is True
    if pair_kind == 'dense-landing':
        assert result['typed_feature_agreement'] is result['loss_agreement'] is None


@pytest.mark.parametrize('defect', ['waveform', 'feature', 'loss', 'repeat_gate_bit'])
def test_full_waveform_and_discrete_feature_loss_or_repeat_disagreements_fail(audit, defect):
    kind = 'exact-repeat' if defect == 'repeat_gate_bit' else 'independent-forward'
    pair = next(p for p in audit.pairs if p['kind'] == kind)
    left, receipt, restart = fixture(audit, pair['left'])
    lr = audit.inspect(pair['left'], left, receipt, restart=restart)
    right, receipt, restart = fixture(audit, pair['right'])
    rr = audit.inspect(pair['right'], right, receipt, restart=restart)
    if defect == 'waveform':
        right[11, 0] += 1e-4  # Early warmup, not merely the command/feature window.
    elif defect == 'feature':
        rr['features']['detected_event_count'] += 1
    elif defect == 'loss':
        rr['score']['loss'] += 1.
    else:
        right[11, 1] = np.nextafter(right[11, 1], 1.)
    result = audit.compare(pair, left, right, lr, rr)
    assert result['passed'] is False


def test_dense_landing_must_match_the_declared_endpoint_not_a_favorable_neighbor(audit):
    pair = next(p for p in audit.pairs if p['kind'] == 'dense-landing')
    left, receipt, restart = fixture(audit, pair['left'])
    lr = audit.inspect(pair['left'], left, receipt, restart=restart)
    right, receipt, restart = fixture(audit, pair['right'])
    rr = audit.inspect(pair['right'], right, receipt, restart=restart)
    left[len(right)-1, 0] += 2e-5
    result = audit.compare(pair, left, right, lr, rr)
    assert result['passed'] is False and result['maximum_state_discrepancies'][0] > 1e-5
    changed_pair = {**pair, 'scope': 'full'}
    with pytest.raises(ValueError, match='complete fixed support'):
        audit.compare(changed_pair, left, right, lr, rr)


def test_repeat_rejects_signed_zero_gate_differences_with_zero_numeric_error(audit):
    pair = next(p for p in audit.pairs if p['kind'] == 'exact-repeat')
    left, receipt, restart = fixture(audit, pair['left'])
    lr = audit.inspect(pair['left'], left, receipt, restart=restart)
    right, receipt, restart = fixture(audit, pair['right'])
    rr = audit.inspect(pair['right'], right, receipt, restart=restart)
    left[10, 1], right[10, 1] = 0., -0.
    result = audit.compare(pair, left, right, lr, rr)
    assert result['maximum_state_discrepancies'] == [0.]*5
    assert result['bitwise_five_state_agreement'] is result['passed'] is False


def test_unregistered_extra_gate_difference_tolerance_is_not_invented(audit):
    pair = next(p for p in audit.pairs if p['kind'] == 'independent-forward')
    left, receipt, restart = fixture(audit, pair['left'])
    lr = audit.inspect(pair['left'], left, receipt, restart=restart)
    right, receipt, restart = fixture(audit, pair['right'])
    rr = audit.inspect(pair['right'], right, receipt, restart=restart)
    right[10, 1] += .01
    result = audit.compare(pair, left, right, lr, rr)
    assert result['maximum_state_discrepancies'][1] > 0.
    assert result['passed'] is True  # All states retained/reported; pair gate was voltage/features/loss only.


def test_complete_fabricated_support_cannot_claim_long_reference_or_scientific_completion(audit):
    cache, reports = {}, {}
    for key in audit.rows:
        states, receipt, restart = fixture(audit, key)
        cache[key] = states
        reports[key] = audit.inspect(key, states, receipt, restart=restart)
    pairs = [audit.compare(pair, cache[pair['left']], cache[pair['right']],
                          reports[pair['left']], reports[pair['right']]) for pair in audit.pairs]
    summary = audit.summarize(reports, pairs)
    assert summary['case_count'] == 104 and summary['comparison_count'] == 120
    assert summary['mandatory_comparison_count'] == 112 and summary['short_pilot_array_gates_passed'] is True
    assert summary['long_reference_qualified'] is summary['scientific_goal_complete'] is False
    assert summary['provenance_and_resource_admission_performed_by_this_module'] is False
    # A failed mandatory comparison is not rescued by all other cases passing.
    pairs[0]['passed'] = False
    assert audit.summarize(reports, pairs)['short_pilot_array_gates_passed'] is False
    # Coarse-reference failures remain reported but do not become a finer gate.
    pairs[0]['passed'] = True
    next(pair for pair in pairs if pair['kind'] == 'reference-coarse')['passed'] = False
    summary = audit.summarize(reports, pairs)
    assert summary['short_pilot_array_gates_passed'] and summary['report_only_coarse_failures'] == 1
    with pytest.raises(ValueError, match='no favorable subset'):
        audit.summarize(dict(list(reports.items())[:-1]), pairs)
    with pytest.raises(ValueError):
        audit.summarize(reports, pairs[:-1])
    pairs[0]['required'] = False
    with pytest.raises(ValueError):
        audit.summarize(reports, pairs)
