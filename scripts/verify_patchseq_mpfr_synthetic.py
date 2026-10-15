"""Independent saved-array/support assessment; never runs a neuron or opens NWB.

Uses the separately implemented frozen response extractor, independently
reconstructs the synthetic inputs and loss, and checks a closed-form RC oracle.
The forward implementations under comparison are MPFR and sealed Boost; this
audit does not constitute a third independently executed active trajectory.
"""

from __future__ import annotations

import hashlib
import json
import math
from importlib.metadata import version
from pathlib import Path

import numpy as np
import yaml
from verify_patchseq_fitting_targets import reconstruct

REG = Path('docs/validation-results/post2008-patchseq-mpfr-long-synthetic-registration-1506.yaml')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expected_table():
    rows = []
    for length in [151000, 330000]:
        for mode in ['passive', 'active']:
            for backend, precisions in [('boost', [53, 160, 320]), ('mpfr', [160, 320])]:
                for precision in precisions:
                    for refinement in [4, 8, 16]:
                        rows.append({'samples': length, 'profile': mode, 'engine': backend,
                                     'bits': precision, 'factor': refinement, 'repeat': 0})
            for precision in [160, 320]:
                rows.append({'samples': length, 'profile': mode, 'engine': 'mpfr',
                             'bits': precision, 'factor': 16, 'repeat': 1})
    return rows


def expected_keys():
    return {'-'.join(str(r[k]) for k in ('samples', 'profile', 'engine', 'bits', 'factor', 'repeat'))
            for r in expected_table()}


def validate_support(data):
    if (data['status'] != 'synthetic-matrix-retained-awaiting-independent-assessment'
        or set(data['cases']) != expected_keys() or data['table'] != expected_table()
        or any(data.get(k) is not False for k in
               ('biological_data_access', 'fitting_restart_authorized', 'scientific_goal_complete'))):
        raise ValueError('matrix incomplete or unsupported; cannot assess a favorable subset')
    if any(r.get('status') not in ('retained', 'failure-retained-no-replacement')
           for r in data['cases'].values()):
        raise ValueError('nonterminal or unsupported case state')


def check_case_evidence(root, key, row, pins, builds):
    paths = {s: root/(key+s) for s in ('.json', '.log', '.npz', '.receipt.json')}
    required = {str(paths[s]) for s in ('.json', '.log')}
    if row['status'] == 'retained':
        required.update(str(paths[s]) for s in ('.npz', '.receipt.json'))
    actual = {str(p) for p in paths.values() if p.is_file()}
    if (set(row['artifacts']) != actual or not required <= actual
        or any(sha(p) != h for p, h in row['artifacts'].items())):
        raise ValueError('case artifact support or hash changed')
    spec = json.loads(paths['.json'].read_text())
    n, mode, engine, bits, factor, repeat = key.split('-')
    expected = {'samples': int(n), 'profile': mode, 'engine': engine,
                'bits': int(bits), 'factor': int(factor), 'repeat': int(repeat)}
    if (spec.get('case') != expected or spec.get('pins') != pins
        or spec.get('build') != str(root/(engine+'-build')) or spec.get('provenance') != builds[engine]
        or spec.get('output') != str(paths['.npz']) or spec.get('receipt') != str(paths['.receipt.json'])):
        raise ValueError('case identity, source/build or output binding differs')
    if not math.isfinite(row['total_case_seconds']) or row['total_case_seconds'] < 0:
        raise ValueError('invalid end-to-end timing evidence')


def inputs(length, mode):
    if length not in (151000, 330000) or mode not in ('passive', 'active'):
        raise ValueError('unregistered synthetic input')
    drive = np.zeros(length+50000, dtype=np.float64)
    drive[75000:length+25000] = {'passive': -50., 'active': 2000.}[mode]
    epoch = {'step_start_index': 25000, 'step_stop_index': length-25000, 'pre_window': [15000, 20000]}
    return drive, epoch


def passive_voltage(length):
    """Exact continuous RC voltage at native samples, including current edges."""
    count = length+50000
    result = np.full(count, -65.)
    left, right = 75000, length+25000
    # C=100pF, g=10nS -> tau=10ms; current=-50pA -> equilibrium=-70mV.
    result[left:right] = -70.+5.*np.exp(-np.arange(right-left)*.02/10.)
    at_release = -70.+5.*math.exp(-(right-left)*.02/10.)
    result[right:] = -65.+(at_release+65.)*np.exp(-np.arange(count-right)*.02/10.)
    return result


def agree(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(agree(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(agree(x, y) for x, y in zip(a, b, strict=True))
    if type(a) is float:
        return math.isfinite(a) and math.isfinite(b) and abs(a-b) <= 1e-5+1e-9*abs(b)
    return a == b


def synthetic_loss(features):
    names = ['baseline_mean_mV', 'early_step_mean_mV', 'steady_step_mean_mV',
             'detected_event_count', 'first_event_latency_s', 'last_to_first_interval_ratio']
    targets, scales = [-65., -60., -55., 3, .01, 1.], [2., 2., 2., 1., .01, .25]
    terms = []
    for name, target, scale in zip(names, targets, scales, strict=True):
        prediction = features.get(name)
        terms.append(1000. if prediction is None or not math.isfinite(prediction)
                     else ((prediction-target)/scale)**2)
    return math.fsum(terms)/6


def compare(left, right, lf, rf):
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] != 5:
        raise ValueError('state array shape mismatch')
    differences = np.max(np.abs(left-right), axis=0)
    loss_a, loss_b = synthetic_loss(lf), synthetic_loss(rf)
    loss_ok = abs(loss_a-loss_b) <= 1e-6+1e-8*abs(loss_b)
    features_ok = agree(lf, rf)
    return {'maximum_state_discrepancies': differences.tolist(),
            'whole_voltage_agreement': bool(differences[0] <= 1e-5),
            'typed_feature_agreement': features_ok, 'loss_agreement': loss_ok,
            'passed': bool(differences[0] <= 1e-5 and features_ok and loss_ok)}


def main():
    reg = yaml.safe_load(REG.read_text())
    root = Path(reg['output'])
    checkpoint = root/'checkpoint.json'
    data = json.loads(checkpoint.read_text())
    pins = data['pins']
    if (pins.get(str(REG)) != sha(REG) or pins.get(str(Path(__file__).relative_to(Path.cwd()))) != sha(__file__)
        or any(sha(p) != h for p, h in pins.items())):
        raise ValueError('source/build evidence changed or verifier not sealed')
    validate_support(data)
    if data['runtime'] != {k: version(k) for k in ('numpy', 'scipy', 'PyYAML')}:
        raise ValueError('independent verification runtime differs')
    report = {'checkpoint_sha256': sha(checkpoint), 'source_hashes': pins,
              'cases': {}, 'comparisons': {}, 'repeats': {}, 'biological_data_access': False,
              'fitting_restart_authorized': False, 'scientific_goal_complete': False}
    audit = root/'independent-assessment.json'
    if audit.exists():
        raise ValueError('existing independent assessment retained; refusing overwrite')
    cache = {}
    for key, row in data['cases'].items():
        check_case_evidence(root, key, row, pins, data['builds'])
        if row['status'] != 'retained':
            report['cases'][key] = {'state': 'failure-preserved-not-qualified', 'evidence': row}
            continue
        n, mode, _engine, _bits, _factor, _repeat = key.split('-')
        n = int(n)
        with np.load(root/(key+'.npz'), allow_pickle=False) as file:
            if file.files != ['states']:
                raise ValueError('unexpected array keys')
            states = file['states']
        if states.shape != (n+50000, 5) or states.dtype != np.float64 or not np.isfinite(states).all():
            raise ValueError('invalid retained all-state array')
        if not np.array_equal(states[0], [-65., .05, .6, .2, .05]):
            raise ValueError('explicit initial state differs')
        drive, epoch = inputs(n, mode)
        receipt = json.loads((root/(key+'.receipt.json')).read_text())
        if receipt != row['receipt'] or receipt['current_sha256'] != hashlib.sha256(drive.tobytes()).hexdigest():
            raise ValueError('receipt/input mismatch')
        if (receipt['arrays_sha256'] != sha(root/(key+'.npz')) or receipt['shape'] != list(states.shape)
            or receipt['uncompressed_bytes'] != states.nbytes
            or receipt['compressed_bytes'] != (root/(key+'.npz')).stat().st_size
            or any(not math.isfinite(receipt[k]) or receipt[k] < 0
                   for k in ('integration_seconds', 'compression_seconds'))):
            raise ValueError('array or timing receipt differs')
        fs = reconstruct(states[50000:, 0], drive[50000:], 50000., epoch, 0.)
        loss = synthetic_loss(fs)
        loss_ok = abs(loss-receipt['score']['loss']) <= 1e-6+1e-8*abs(loss)
        flags = {'features_reconstructed': agree(fs, receipt['features']), 'loss_reconstructed': loss_ok}
        if mode == 'passive':
            error = float(np.max(np.abs(states[:, 0]-passive_voltage(n))))
            flags['passive_oracle_agreement'] = error <= 1e-5
        report['cases'][key] = {'state': 'independently-remeasured', 'checks': flags,
                                 'passed': all(flags.values()), 'features': fs, 'loss': loss,
                                 'timing': {k: receipt[k] for k in ('integration_seconds', 'compression_seconds')},
                                 'total_case_seconds': row['total_case_seconds'],
                                 'compressed_bytes': receipt['compressed_bytes'],
                                 'uncompressed_bytes': receipt['uncompressed_bytes'],
                                 'passive_maximum_voltage_error_mV': error if mode == 'passive' else None}
        # Bound memory to one profile, loading only compared pairs below.
        cache[key] = fs
    def pair(a, b, repeat=False):
        if a not in cache or b not in cache:
            return {'passed': False, 'state': 'failure-or-incomplete-support'}
        with np.load(root/(a+'.npz'), allow_pickle=False) as af, np.load(root/(b+'.npz'), allow_pickle=False) as bf:
            left, right = af['states'], bf['states']
        if repeat:
            return {'passed': bool(np.array_equal(left.view(np.uint64), right.view(np.uint64)))}
        return compare(left, right, cache[a], cache[b])
    for n in (151000, 330000):
        for mode in ('passive', 'active'):
            prefix = f'{n}-{mode}-'
            for bits in (160, 320):
                for factor in (4, 8, 16):
                    a, b = prefix+f'mpfr-{bits}-{factor}-0', prefix+f'boost-{bits}-{factor}-0'
                    report['comparisons'][a+'/'+b] = {'role': 'required-cross-backend', **pair(a, b)}
                a, b = prefix+f'mpfr-{bits}-16-0', prefix+f'mpfr-{bits}-16-1'
                report['repeats'][a+'/'+b] = pair(a, b, repeat=True)
            for engine in ('boost', 'mpfr'):
                for factor in (4, 8, 16):
                    a, b = prefix+f'{engine}-160-{factor}-0', prefix+f'{engine}-320-{factor}-0'
                    report['comparisons'][a+'/'+b] = {'role': 'precision', **pair(a, b)}
                for bits in ((53, 160, 320) if engine == 'boost' else (160, 320)):
                    for lo, hi in ((4, 8), (8, 16)):
                        a, b = prefix+f'{engine}-{bits}-{lo}-0', prefix+f'{engine}-{bits}-{hi}-0'
                        report['comparisons'][a+'/'+b] = {'role': 'refinement', **pair(a, b)}
            for factor in (4, 8, 16):
                for bits in (160, 320):
                    a, b = prefix+f'boost-53-{factor}-0', prefix+f'boost-{bits}-{factor}-0'
                    report['comparisons'][a+'/'+b] = {'role': 'native-control-not-promotion-gate', **pair(a, b)}
    report['backend_equivalence_passed'] = (
        all(r.get('passed') is True for r in report['cases'].values())
        and all(r['passed'] for r in report['repeats'].values())
        and all(r['passed'] for r in report['comparisons'].values() if r['role'] == 'required-cross-backend'))
    report['finest_reference_qualification'] = {}
    for n in (151000, 330000):
        for mode in ('passive', 'active'):
            for engine in ('boost', 'mpfr'):
                prefix = f'{n}-{mode}-{engine}-'
                required = [(prefix+'160-16-0', prefix+'320-16-0')]
                required += [(prefix+f'{bits}-8-0', prefix+f'{bits}-16-0') for bits in (160, 320)]
                report['finest_reference_qualification'][prefix.rstrip('-')] = all(
                    report['comparisons'][a+'/'+b]['passed'] for a, b in required)
    report['production_feasibility_established'] = False
    report['status'] = 'synthetic-assessed-not-production-authorization'
    if sha(checkpoint) != report['checkpoint_sha256'] or any(sha(p) != h for p, h in pins.items()):
        raise ValueError('source/checkpoint changed during verification')
    for key, row in data['cases'].items():
        check_case_evidence(root, key, row, pins, data['builds'])
    with audit.open('x') as file:
        json.dump(report, file, sort_keys=True, allow_nan=False)
    print('Independent synthetic assessment retained; equivalence:', report['backend_equivalence_passed'])


if __name__ == '__main__':
    main()
