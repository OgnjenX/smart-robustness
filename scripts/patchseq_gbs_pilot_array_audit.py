"""Independent pure saved-array checks for pilot1515; no execution authority.

This module neither loads a solver nor imports the producer/controller. Its
caller still needs a separately implemented, sealed admission/resource/artifact
audit before interpreting these computations as an assessment of real evidence.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass

import numpy as np
from verify_patchseq_fitting_targets import reconstruct
from verify_patchseq_gbs_pilot_protocol import (
    canonical,
    case_name,
    expected_comparisons,
    expected_inputs,
    expected_rows,
    sha,
    validate_registration,
)

INITIAL = np.array([-65., .05, .6, .2, .05], dtype=np.float64)
ENVIRONMENT = {'mpfr': '4.2.2', 'header': '4.2.2', 'gmp': '6.3.0', 'tls': 1,
    'emin': -1073741823, 'emax': 1073741823, 'precision160': 160, 'precision320': 320,
    'cancellation160': 1, 'cancellation320': 1}
RECEIPT_FIELDS = {'schema_version', 'case_key', 'case', 'input_signature', 'shape', 'arrays_sha256',
    'uncompressed_bytes', 'compressed_bytes', 'features', 'score', 'final_state', 'solver_statistics',
    'maximum_embedded_estimate', 'driver_open_seconds', 'forward_call_seconds',
    'checkpoint_serialization_seconds', 'postprocess_seconds', 'compression_seconds',
    'peak_process_RSS_bytes', 'scientific_goal_complete', 'biological_access', 'fitting_restart'}


def need(condition, message):
    if not condition:
        raise ValueError('independent pilot array audit: '+message)


def bits_equal(left, right):
    return bool(left.shape == right.shape and left.dtype == right.dtype == np.float64
                and np.array_equal(left.view(np.uint64), right.view(np.uint64)))


def typed_agree(left, right, absolute=1e-5, relative=1e-9):
    if type(left) is not type(right):
        return False
    if type(right) is dict:
        return left.keys() == right.keys() and all(typed_agree(left[k], right[k], absolute, relative) for k in right)
    if type(right) is list:
        return len(left) == len(right) and all(typed_agree(a, b, absolute, relative) for a, b in zip(left, right, strict=True))
    if type(right) is float:
        return math.isfinite(left) and math.isfinite(right) and abs(left-right) <= absolute+relative*abs(right)
    return left == right


@dataclass(frozen=True)
class Dyadic:
    """Exact signed integer times a power of two, without any forward library.

Magnitude comparisons do not allocate shifts proportional to an untrusted MPFR
exponent. Even an exponent near the MPFR limit needs at most precision-sized
integer work; signed zero is preserved separately for binary64 readback.
"""

    significand: int
    exponent: int
    negative_zero: bool = False

    @classmethod
    def parse(cls, text, width):
        need(type(width) is int and width in (160, 320) and type(text) is str
             and len(text) <= width//4+32, 'invalid exact-number width or text')
        if text in ('0p0', '-0p0'):
            return cls(0, 0, text.startswith('-'))
        match = re.fullmatch(r'(-?)([0-9a-f]+)p(-?(?:0|[1-9][0-9]*))', text)
        need(match is not None, 'noncanonical exact-number spelling')
        sign, digits, exponent_text = match.groups()
        exponent = int(exponent_text)
        number = int(digits, 16)
        need(len(digits) == width//4 and number.bit_length() == width
            and str(exponent) == exponent_text
            and -1073741823 <= exponent+width <= 1073741823, 'exact-number support differs')
        return cls(-number if sign else number, exponent)

    @classmethod
    def from_float(cls, number):
        numerator, denominator = number.as_integer_ratio()
        return cls(numerator, -(denominator.bit_length()-1), number == 0. and math.copysign(1., number) < 0)

    def compare(self, other):
        a, b = self.significand, other.significand
        sign_a, sign_b = (a > 0)-(a < 0), (b > 0)-(b < 0)
        if sign_a != sign_b:
            return (sign_a > sign_b)-(sign_a < sign_b)
        if not a:
            return 0
        top_a, top_b = abs(a).bit_length()+self.exponent, abs(b).bit_length()+other.exponent
        if top_a != top_b:
            return sign_a*((top_a > top_b)-(top_a < top_b))
        # Equal top bits bound this shift by the significand-width difference.
        shift = min(self.exponent, other.exponent)
        mag_a, mag_b = abs(a) << (self.exponent-shift), abs(b) << (other.exponent-shift)
        return sign_a*((mag_a > mag_b)-(mag_a < mag_b))

    def export(self):
        if not self.significand:
            return -0. if self.negative_zero else 0.
        try:
            result = float.fromhex(('-' if self.significand < 0 else '')+
                '0x'+format(abs(self.significand), 'x')+'p'+str(self.exponent))
        except OverflowError as exc:
            raise ValueError('independent exact state overflows binary64') from exc
        need(math.isfinite(result), 'nonfinite independent exact-state export')
        return result


def exact_states(texts, width, observed):
    need(type(texts) is list and len(texts) == 5, 'full exact five-state support differs')
    values = [Dyadic.parse(text, width) for text in texts]
    lower, upper = Dyadic.from_float(-1e-8), Dyadic.from_float(1.+1e-8)
    need(all(value.compare(lower) >= 0 and value.compare(upper) <= 0 for value in values[1:]),
         'full-precision accepted gate exceeds the original binary64-lifted bound')
    exported = np.array([value.export() for value in values], dtype=np.float64)
    need(bits_equal(exported, observed), 'full exact state does not export to the retained native sample')
    return texts


def unsigned(text):
    need(type(text) is str and bool(re.fullmatch(r'0|[1-9][0-9]{0,19}', text)), 'noncanonical checkpoint integer')
    number = int(text)
    need(number <= 2**64-1, 'checkpoint integer exceeds its recorded width')
    return number


def inputs(row):
    count = row['samples']
    native = np.arange(count+5000)
    drive = np.where((native >= 7500) & (native < 5000+count-2500),
                     -50. if row['profile'] == 'passive' else 2000., 0.).astype(np.float64)
    return drive, {'step_start_index': 2500, 'step_stop_index': count-2500, 'pre_window': [1500, 2000]}


def passive_voltage(row):
    native = np.arange(row['samples']+5000)
    onset, release = 7500, row['samples']+2500
    result = np.full(native.size, -65.)
    delta = 1000./50000.  # The stipulated original binary64 native interval.
    during = (native >= onset) & (native < release)
    result[during] = -70.+5.*np.exp(-(native[during]-onset)*delta/10.)
    at_release = -70.+5.*math.exp(-(release-onset)*delta/10.)
    after = native >= release
    result[after] = -65.+(at_release+65.)*np.exp(-(native[after]-release)*delta/10.)
    return result if row['stop_index'] is None else result[:row['stop_index']+1]


def six_term_score(features):
    names = ('baseline_mean_mV', 'early_step_mean_mV', 'steady_step_mean_mV',
             'detected_event_count', 'first_event_latency_s', 'last_to_first_interval_ratio')
    targets, scales = (-65., -60., -55., 3, .01, 1.), (2., 2., 2., 1., .01, .25)
    terms, gates = {}, {}
    for name, target, scale in zip(names, targets, scales, strict=True):
        value = features.get(name)
        valid = type(value) in (int, float) and math.isfinite(value)
        if name == 'detected_event_count' and valid:
            valid = value >= 0 and int(value) == value
        terms[name] = ((value-target)/scale)**2 if valid else 1000.
        gates[name] = bool(valid and abs(value-target) <= scale)
    return {'loss': math.fsum(terms.values())/6, 'terms': terms, 'gates': gates,
            'all_applicable_gates_pass': all(gates.values())}


class ArrayAudit:
    def __init__(self, registration, builds):
        validate_registration(registration)
        need(type(builds) is dict and set(builds) == {'gbs', 'rkf78'}, 'both forward build identities are required')
        self.reg, self.builds = json.loads(canonical(registration)), json.loads(canonical(builds))
        self.rows = {case_name(row): row for row in expected_rows()}
        self.manifest = expected_inputs(list(self.rows.values()))
        self.pairs = expected_comparisons(list(self.rows.values()))

    def states(self, key, states):
        need(key in self.rows, 'case is not in the complete prospective support')
        need(type(states) is np.ndarray and states.dtype == np.float64 and states.flags.c_contiguous
            and list(states.shape) == self.manifest[key]['shape'] and np.isfinite(states).all(),
            'whole native five-state array support differs')
        need(bits_equal(states[0], INITIAL), 'explicit initial bits differ')
        need(bool(np.all(states[:, 1:] >= -1e-8) and np.all(states[:, 1:] <= 1.+1e-8)),
            'native observed gate exceeds the original bounds')

    def statistics(self, row, stats, index):
        need(type(stats) is dict and set(stats) == {'position', 'rhs', 'accepted', 'rejected', 'dense'}
            and all(type(value) is int and 0 <= value <= 2**64-1 for value in stats.values()), 'invalid work accounting')
        need(stats['position'] == index and stats['dense'] <= index, 'work/native support differs')
        if row['engine'] == 'gbs':
            policy = self.reg['primary_policies'][row['policy']]
            need(0 < stats['accepted'] <= policy['maximum_steps'] and stats['accepted'] <= stats['rhs']
                and stats['rhs'] <= policy['maximum_rhs'] and stats['rejected'] <= min(stats['rhs'], policy['maximum_rejections']),
                'primary work budgets or counts differ')
        else:
            intervals = row['factor']*index
            need(stats == {'position': index, 'rhs': intervals*13, 'accepted': intervals, 'rejected': 0, 'dense': 0}
                and stats['rhs'] <= self.reg['resources']['maximum_independent_rhs']
                and intervals <= self.reg['resources']['maximum_independent_substeps'], 'independent fixed work support differs')

    def checkpoint(self, key, envelope, states, index, final_stats=None):
        row, signature = self.rows[key], self.manifest[key]
        need(type(envelope) is dict and set(envelope) == {'schema', 'configuration', 'payload', 'sha256'}
            and envelope['schema'] == 'gbs-hh-checkpoint-v1', 'checkpoint envelope differs')
        configuration = {'kind': 'gbs-hh-primary-5-v1', 'bits': row['bits'],
            'policy': self.reg['primary_policies'][row['policy']], 'rate_binary64': (50000.).hex(),
            'samples': signature['shape'][0], 'current_sha256': signature['integrated_current_sha256'],
            'initial': signature['initial_binary64'],
            'parameters': [signature['parameters_binary64'][name] for name in self.reg['inputs']['field_order']],
            'build_sha256': sha(self.builds['gbs']), 'barrier_intervals': 4096, 'environment': ENVIRONMENT}
        need(canonical(envelope['configuration']) == canonical(configuration)
            and envelope['sha256'] == sha({k: v for k, v in envelope.items() if k != 'sha256'}),
            'checkpoint input/build/policy identity or envelope checksum differs')
        text = envelope['payload']
        need(type(text) is str and len(text) <= 16384 and text.endswith('\n') and '\x00' not in text,
             'invalid checkpoint payload text')
        fields = text.splitlines()
        need(len(fields) == 15 and fields[:4] == ['gbs-guarded-core-v1', str(row['bits']), '5', sha(configuration)]
            and unsigned(fields[4]) == index, 'checkpoint header/native index differs')
        drive, _epoch = inputs(row)
        barriers = {0, signature['shape'][0]-1, *range(4096, signature['shape'][0]-1, 4096)}
        barriers.update(int(j) for j in np.flatnonzero(drive[1:signature['shape'][0]-1] != drive[:signature['shape'][0]-2])+1)
        need(index in barriers, 'checkpoint is not at a prescribed accepted endpoint')
        delta = Dyadic.from_float(1000./50000.)
        need(Dyadic.parse(fields[5], row['bits']).compare(Dyadic(index*delta.significand, delta.exponent)) == 0,
             'checkpoint time differs from native-index-times-original-binary64-interval')
        exact = exact_states(fields[6:11], row['bits'], states[index])
        counts = dict(zip(('rhs', 'accepted', 'rejected', 'dense'), map(unsigned, fields[11:]), strict=True))
        stats = {'position': index, **counts}
        self.statistics(row, stats, index)
        if final_stats is not None:
            need(canonical(stats) == canonical(final_stats), 'final checkpoint work counts differ from receipt')
        return exact, stats

    def inspect(self, key, states, receipt, *, restart=None):
        self.states(key, states)
        row, signature = self.rows[key], self.manifest[key]
        need(type(receipt) is dict and set(receipt) == RECEIPT_FIELDS, 'receipt field support differs')
        canonical(receipt)
        need(type(receipt['schema_version']) is int and receipt['schema_version'] == 1
            and receipt['case_key'] == key and canonical(receipt['case']) == canonical(row)
            and canonical(receipt['input_signature']) == canonical(signature)
            and canonical(receipt['shape']) == canonical(signature['shape'])
            and type(receipt['uncompressed_bytes']) is int and receipt['uncompressed_bytes'] == states.nbytes,
            'receipt identity/native support differs')
        for name in ('biological_access', 'fitting_restart', 'scientific_goal_complete'):
            need(receipt[name] is False, 'receipt makes an unauthorized claim')
        for name in ('driver_open_seconds', 'forward_call_seconds', 'checkpoint_serialization_seconds',
                     'postprocess_seconds', 'compression_seconds'):
            need(type(receipt[name]) is float and math.isfinite(receipt[name]) and receipt[name] >= 0., 'receipt timing differs')
        need(type(receipt['peak_process_RSS_bytes']) is int and receipt['peak_process_RSS_bytes'] > 0
            and type(receipt['compressed_bytes']) is int and 0 < receipt['compressed_bytes'] <= states.nbytes+65536
            and type(receipt['arrays_sha256']) is str and re.fullmatch('[0-9a-f]{64}', receipt['arrays_sha256']) is not None,
            'receipt storage or process memory differs')
        index = len(states)-1
        stats = receipt['solver_statistics']
        self.statistics(row, stats, index)
        final = receipt['final_state']
        if row['engine'] == 'gbs':
            exact, _counts = self.checkpoint(key, final, states, index, stats)
            need(receipt['maximum_embedded_estimate'] is None, 'primary has unregistered reference diagnostics')
        else:
            need(type(final) is dict and set(final) == {'kind', 'values'} and final['kind'] == 'rkf78-exact-final',
                 'reference full-precision final support differs')
            exact = exact_states(final['values'], row['bits'], states[-1])
            embedded = receipt['maximum_embedded_estimate']
            need(type(embedded) is list and len(embedded) == 5
                and all(type(v) is float and math.isfinite(v) and v >= 0. for v in embedded),
                'independent embedded diagnostic support differs')
        if row['role'] == 'checkpoint':
            _exact_restart, restart_stats = self.checkpoint(key, restart, states, 4096)
            need(all(restart_stats[k] <= stats[k] for k in stats), 'restart reset a cumulative work count')
        else:
            need(restart is None and receipt['checkpoint_serialization_seconds'] == 0., 'unexpected restart artifact or timing')
        if row['role'] == 'landing':
            need(receipt['features'] is None and receipt['score'] is None, 'prefix was scored as a complete response epoch')
            features = score = None
            feature_ok = score_ok = None
        else:
            drive, epoch = inputs(row)
            features = reconstruct(states[5000:, 0], drive[5000:], 50000., epoch, 0.)
            score = six_term_score(features)
            feature_ok = typed_agree(receipt['features'], features)
            score_ok = typed_agree(receipt['score'], score, 1e-6, 1e-8)
        error = float(np.max(np.abs(states[:, 0]-passive_voltage(row)))) if row['profile'] == 'passive' else None
        checks = {'features_reconstructed': feature_ok, 'six_term_score_reconstructed': score_ok,
                  'passive_analytic_voltage_agreement': error <= 1e-5 if error is not None else None}
        return {'case_key': key, 'state': 'independently-remeasured-not-forward-replayed',
            'checks': checks, 'passed': all(v for v in checks.values() if v is not None),
            'features': features, 'score': score, 'exact_final_values': exact,
            'passive_maximum_voltage_error_mV': error, 'solver_statistics': stats,
            'timing': {name: receipt[name] for name in receipt if name.endswith('_seconds')},
            'peak_process_RSS_bytes': receipt['peak_process_RSS_bytes'],
            'compressed_bytes': receipt['compressed_bytes'], 'uncompressed_bytes': states.nbytes}

    def compare(self, pair, left, right, left_report, right_report):
        need(canonical(pair) in {canonical(item) for item in self.pairs}, 'comparison is not in complete fixed support')
        for key, states, report in ((pair['left'], left, left_report), (pair['right'], right, right_report)):
            self.states(key, states)
            need(report['case_key'] == key, 'comparison summary belongs to another case')
        endpoint = pair['scope'] == 'endpoint'
        if endpoint:
            need(self.rows[pair['right']]['stop_index'] == len(right)-1, 'landing support differs')
            a, b = left[len(right)-1], right[-1]
        else:
            need(left.shape == right.shape, 'whole comparison array support differs')
            a, b = left, right
        with np.errstate(over='ignore', invalid='ignore'):
            differences = np.abs(a-b) if endpoint else np.max(np.abs(a-b), axis=0)
        voltage_ok = bool(math.isfinite(differences[0]) and differences[0] <= 1e-5)
        exact = pair['kind'] in ('exact-repeat', 'exact-restart')
        bitwise_ok = bits_equal(a, b) if exact else None
        feature_ok = loss_ok = None
        loss_difference = None
        if not endpoint:
            feature_ok = typed_agree(left_report['features'], right_report['features'])
            lo, hi = left_report['score']['loss'], right_report['score']['loss']
            loss_difference = abs(lo-hi)
            loss_ok = math.isfinite(loss_difference) and loss_difference <= 1e-6+1e-8*abs(hi)
            if not math.isfinite(loss_difference):
                loss_difference = None
        # Non-voltage gate differences remain fully reported, but do not invent
        # an extra scientific pairwise tolerance absent from registration1515.
        return {**pair, 'maximum_state_discrepancies': [float(v) if math.isfinite(v) else None for v in differences],
            'difference_overflow': bool(not np.isfinite(differences).all()),
            'whole_voltage_agreement': voltage_ok, 'typed_feature_agreement': feature_ok,
            'loss_agreement': loss_ok, 'absolute_loss_difference': loss_difference,
            'bitwise_five_state_agreement': bitwise_ok,
            'full_precision_final_values_identical': (left_report['exact_final_values'] == right_report['exact_final_values']) if exact else None,
            'passed': all(v for v in (voltage_ok, feature_ok, loss_ok, bitwise_ok) if v is not None)}

    def summarize(self, cases, comparisons):
        need(type(cases) is dict and set(cases) == set(self.rows), 'incomplete case report: no favorable subset')
        need(type(comparisons) is list and len(comparisons) == len(self.pairs), 'incomplete comparison report')
        for expected, actual in zip(self.pairs, comparisons, strict=True):
            need(type(actual) is dict and set(expected) | {'passed'} <= set(actual)
                and canonical({k: actual[k] for k in expected}) == canonical(expected)
                and type(actual['passed']) is bool, 'comparison identity or result support differs')
        need(all(type(item) is dict and {'case_key', 'passed'} <= set(item)
                 and item['case_key'] == key and type(item['passed']) is bool for key, item in cases.items()),
             'case summary identity or result support differs')
        return {'case_count': len(cases), 'comparison_count': len(comparisons),
            'mandatory_comparison_count': sum(pair['required'] for pair in self.pairs),
            'case_array_checks_passed': all(item['passed'] for item in cases.values()),
            'mandatory_comparisons_passed': all(item['passed'] for item in comparisons if item['required']),
            'report_only_coarse_failures': sum(not item['passed'] for item in comparisons if not item['required']),
            'short_pilot_array_gates_passed': (all(item['passed'] for item in cases.values())
                and all(item['passed'] for item in comparisons if item['required'])),
            'provenance_and_resource_admission_performed_by_this_module': False,
            'long_reference_qualified': False, 'whole_biological_workflow_feasibility_established': False,
            'biological_access': False, 'fitting_restart': False, 'scientific_goal_complete': False}
