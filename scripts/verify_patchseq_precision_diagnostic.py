"""Independent terminal saved-array measurements, source and case accounting."""

from __future__ import annotations

import fcntl
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import yaml
from patchseq_hh_independent_replay import kinetics
from patchseq_independent_fit_loss import observable_loss
from run_patchseq_isolated_fit import fitting_records, prepare_inputs, read_yaml
from verify_patchseq_fitting_targets import reconstruct
from verify_patchseq_reference_convergence import agreement, close, compare
from verify_patchseq_whole_file_integrity import digest

REG = Path('docs/validation-results/post2008-patchseq-precision-registration-1490.yaml')
TERMINAL = 'fixed-two-input-precision-diagnostic-complete-not-production-authorization'


@contextmanager
def terminal(directory):
    directory = Path(directory)
    with (directory/'runner.lock').open('r') as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        path = directory/'checkpoint.json'
        sha = digest(path)
        state = json.loads(path.read_text())
        if state.get('status') != TERMINAL:
            raise ValueError('precision diagnostic is not terminal')
        yield state, sha
        if digest(path) != sha:
            raise ValueError('precision terminal snapshot changed during audit')


def reference_qualified(data, comparisons):
    return (len(data) == 9 and sum(v is not None for v in data.values()) == 9
            and comparisons.get('160-16/320-16', {}).get('passed') is True
            and comparisons.get('160-8/160-16', {}).get('passed') is True
            and comparisons.get('320-8/320-16', {}).get('passed') is True)


def valid_states(states, current, initial):
    return (states.dtype == np.float64 and states.shape == (len(current), 5)
            and np.isfinite(states).all() and np.array_equal(states[0], initial)
            and np.all(states[:, 1:] >= -1e-8) and np.all(states[:, 1:] <= 1+1e-8))


def main():
    reg = read_yaml(REG)
    output = Path(reg['output'])
    with terminal(output) as (state, checkpoint_sha):
        pins = dict(state['source_hashes'])
        pins[str(Path(__file__))] = digest(__file__)
        if any(digest(p) != h for p, h in pins.items()):
            raise ValueError('precision source/build/dependency hash differs')
        if any(digest(p) != h for p, h in reg['prerequisite_hashes'].items()):
            raise ValueError('precision prerequisite hash differs')
        build = read_yaml(output/'build'/'build.yaml')
        if build != state['build'] or digest(build['binary_path']) != build['binary_sha256']:
            raise ValueError('precision build receipt differs')
        if (digest(build['compiler']) != build['compiler_sha256']
            or any(digest(p) != h for p, h in build['dependencies'].items())):
            raise ValueError('precision compiler or transitive headers changed')
        for bits in (53, 160, 320):
            if state['precision_probes'][str(bits)] != {
                'digits': bits, 'cancellation': 0 if bits == 53 else 2.**-100}:
                raise ValueError('precision probe receipt differs')
        original = json.loads(Path(reg['original']).read_text())
        parameter = original['strata']['Pvalb-L23']['candidates']['19']['parameters']
        if state['parameters'] != parameter:
            raise ValueError('candidate parameters differ')
        protocol = read_yaml('docs/validation-results/post2008-patchseq-isolated-fit-registration-1436.yaml')
        if digest(protocol['training_catalog']) != protocol['training_catalog_sha256']:
            raise ValueError('training catalog changed')
        keys = {tuple(k) for k in reg['source_keys']}
        records = []
        for original_record in fitting_records(read_yaml(protocol['training_catalog']), 'Pvalb-L23'):
            record = dict(original_record)
            record['sweeps'] = [s for s in original_record['sweeps'] if
                (record['source_batch'], record['slot'], s['sweep_number']) in keys]
            if record['sweeps']:
                records.append(record)
        targets = {(r['source_batch'], r['slot'], s['sweep_number']): s['features']
                   for r in records for s in r['sweeps']}
        if set(targets) != keys:
            raise ValueError('independent source support differs')
        source = read_yaml('docs/validation-results/post2008-patchseq-fitting-target-registration-1429.yaml')
        inputs, ledger, _ = prepare_inputs(records, source, read_yaml(source['nominal']))
        if ledger != state['opened_training_command_files'] or set(inputs) != keys:
            raise ValueError('independent command accounting differs')
        expected = {f'{batch}-{slot}-{number}/{bits}-{factor}' for batch, slot, number in keys
                    for bits in (53, 160, 320) for factor in (4, 8, 16)}
        if len(expected) != 18 or set(state['cases']) != expected:
            raise ValueError('precision case support incomplete or expanded')
        reports, summaries, artifacts = {}, {}, {}
        all_agree = True
        for key in map(tuple, reg['source_keys']):
            label = '-'.join(map(str, key))
            command, rate, holding, epoch = inputs[key]
            nominal = command+holding
            current = np.r_[np.full(round(rate), nominal[0]), nominal]
            receipt = state['inputs'][label]
            input_path = output/(label+'--input.npz')
            if (Path(receipt['path']) != input_path or digest(input_path) != receipt['sha256']
                or receipt['current_sha256'] != hashlib.sha256(current.tobytes()).hexdigest()
                or receipt['warmup_samples'] != round(rate) or receipt['rate_hz'] != rate
                or receipt['holding_pA'] != holding):
                raise ValueError('input receipt differs from independent command reconstruction')
            with np.load(input_path, allow_pickle=False) as inp:
                initial = inp['initial'].copy()
                if (set(inp.files) != {'initial', 'current'}
                    or not np.array_equal(inp['current'], current)
                    or initial.shape != (5,) or not np.array_equal(initial, receipt['initial'])):
                    raise ValueError('explicit initial state or full current differs')
            a, b, slow, _ = kinetics(np.array([parameter['leak_reversal_mV']]),
                                    np.array([parameter['threshold_shift_mV']]),
                                    np.array([parameter['m_tau_max_ms']]))
            equilibrium = np.r_[parameter['leak_reversal_mV'], (a/(a+b)).ravel(), slow.ravel()]
            if not np.allclose(initial, equilibrium, atol=1e-14, rtol=0):
                raise ValueError('initial state does not follow original equilibrium rule')
            artifacts[str(input_path)] = digest(input_path)
            data = {}
            for bits in (53, 160, 320):
                for factor in (4, 8, 16):
                    name, case_id = f'{bits}-{factor}', f'{label}/{bits}-{factor}'
                    row = state['cases'][case_id]
                    array = output/f'{label}--{bits}-{factor}.npz'
                    spec_path = output/f'{label}--{bits}-{factor}.json'
                    log = output/f'{label}--{bits}-{factor}.log'
                    if digest(spec_path) != row['spec_sha256'] or digest(log) != row['log_sha256']:
                        raise ValueError('case spec or log hash differs')
                    spec = json.loads(spec_path.read_text())
                    wanted = {'source_hashes': state['source_hashes'], 'input': str(input_path),
                              'input_sha256': digest(input_path), 'build_directory': str(output/'build'),
                              'build': build, 'parameters': parameter, 'rate_hz': rate,
                              'bits': bits, 'factor': factor, 'output': str(array)}
                    if spec != wanted or row['bits'] != bits or row['factor'] != factor:
                        raise ValueError('case specification differs from fixed table')
                    artifacts.update({str(spec_path): digest(spec_path), str(log): digest(log)})
                    data[name] = None
                    if row['state'] == 'failed-retained-no-replacement':
                        if not isinstance(row.get('error'), str) or not row['error']:
                            raise ValueError('missing failure reason')
                        if (array.exists() != row['orphan_output_retained']
                            or row['orphan_array_sha256'] != (digest(array) if array.exists() else None)):
                            raise ValueError('failure artifact changed')
                        if array.exists():
                            artifacts[str(array)] = digest(array)
                        reports[case_id] = {'state': row['state'], 'error': row['error']}
                        continue
                    if row['state'] != 'retained' or Path(row['arrays']) != array or digest(array) != row['array_sha256']:
                        raise ValueError('saved state array scope or hash differs')
                    with np.load(array, allow_pickle=False) as archive:
                        if archive.files != ['states']:
                            raise ValueError('unexpected state archive members')
                        states = archive['states'].copy()
                    if not valid_states(states, current, initial):
                        raise ValueError('state grid, initial state, finiteness or gate bounds differ')
                    features = reconstruct(states[round(rate):, 0], command, rate, epoch, holding)
                    score = observable_loss(targets[key], features)
                    checks = {'features': agreement(features, row['features']),
                              'loss': close(score['loss'], row['score']['loss']),
                              'gates': score['gates'] == row['score']['gates']}
                    all_agree = all_agree and all(checks.values())
                    reports[case_id] = {'state': 'retained', 'checks': checks,
                                        'independent_features': features, 'independent_score': score}
                    data[name] = {'voltage': states[:, 0], 'features': features, 'loss': score['loss']}
                    artifacts[str(array)] = digest(array)
            pairs = [(f'160-{f}', f'320-{f}') for f in (4, 8, 16)]
            for precision in (53, 160, 320):
                pairs.extend([(f'{precision}-4', f'{precision}-8'), (f'{precision}-8', f'{precision}-16')])
            pairs.extend((f'53-{f}', f'{precision}-{f}') for f in (4, 8, 16) for precision in (160, 320))
            comparisons = {a+'/'+b: compare(data[a], data[b]) for a, b in pairs}
            summary = {'comparisons': comparisons, 'reference_qualified': reference_qualified(data, comparisons)}
            same = agreement(summary, state['summaries'][label])
            all_agree = all_agree and same
            summaries[label] = {**summary, 'stored_summary_agrees': same}
        if (set(state['summaries']) != set(summaries) or set(state['inputs']) != set(summaries)
            or any(digest(p) != h for p, h in pins.items())):
            raise ValueError('terminal scope or source chain changed')
        report = {'status': 'independent-precision-array-feature-loss-and-case-audit-retained',
                  'checkpoint_sha256': checkpoint_sha, 'source_hashes': pins, 'artifacts': artifacts,
                  'cases': reports, 'summaries': summaries, 'all_independent_checks_agree': all_agree,
                  'all_precision_trajectories_independently_resimulated': False,
                  'fitting_resumed': False, 'held_out_response_features_read': False,
                  'production_transition_authorized': False, 'scientific_goal_complete': False}
    destination = Path(reg['independent_audit']['output'])
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x') as file:
        yaml.safe_dump(report, file, sort_keys=False)
    print('Independent precision audit retained; agreement:', all_agree, flush=True)


if __name__ == '__main__':
    main()
