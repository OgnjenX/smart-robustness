"""Fixed two-input arithmetic/refinement diagnostic, never a fitting runner."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import subprocess
import sys
from contextlib import ExitStack
from importlib.metadata import version
from pathlib import Path

import numpy as np
from patchseq_fit_objective import score_sweep
from patchseq_independent_fit_inputs import inputs_for_training
from patchseq_precision_driver import PrecisionDriver, digest, shared_brian_initial_state
from patchseq_response_features import characterize
from patchseq_selected_fit_agreement import feature_agreement
from run_patchseq_isolated_fit import checkpoint, fitting_records, read_yaml

REG = Path('docs/validation-results/post2008-patchseq-precision-registration-1490.yaml')
TERMINAL = 'fixed-two-input-precision-diagnostic-complete-not-production-authorization'


def restricted_records(records, keys):
    """Do not decode other training commands, much less held-out responses."""
    kept = []
    for record in records:
        item = copy.deepcopy(record)
        item['sweeps'] = [s for s in item['sweeps'] if
                          (item['source_batch'], item['slot'], s['sweep_number']) in keys]
        if item['sweeps']:
            kept.append(item)
    actual = {(r['source_batch'], r['slot'], s['sweep_number']) for r in kept for s in r['sweeps']}
    if actual != keys:
        raise ValueError('fixed two-input source support differs')
    return kept


def comparison(left, right):
    if left is None or right is None:
        return {'passed': False, 'replays_complete': False}
    error = float(np.max(np.abs(left['voltage']-right['voltage'])))
    features = feature_agreement(left['features'], right['features'])
    loss = abs(left['loss']-right['loss']) <= 1e-6+1e-8*abs(right['loss'])
    return {'passed': error <= 1e-5 and features and loss, 'replays_complete': True,
            'maximum_waveform_difference_mV': error, 'feature_agreement': features,
            'loss_agreement': loss}


def pairs():
    result = [(f'160-{f}', f'320-{f}') for f in (4, 8, 16)]
    result += [(f'{b}-{lo}', f'{b}-{hi}') for b in (53, 160, 320) for lo, hi in ((4, 8), (8, 16))]
    result += [(f'53-{f}', f'{b}-{f}') for f in (4, 8, 16) for b in (160, 320)]
    return result


def qualified(rows, comparisons):
    required = ('160-16/320-16', '160-8/160-16', '320-8/320-16')
    return (len(rows) == 9 and all(r is not None for r in rows.values())
            and all(comparisons.get(k, {}).get('passed') is True for k in required))


def worker(spec_path):
    spec = json.loads(Path(spec_path).read_text())
    if any(digest(p) != h for p, h in spec['source_hashes'].items()):
        raise ValueError('worker provenance differs')
    if digest(spec['input']) != spec['input_sha256']:
        raise ValueError('worker command/initial state changed')
    driver = PrecisionDriver(spec['build_directory'])
    if driver.provenance != spec['build']:
        raise ValueError('worker retained build differs')
    with np.load(spec['input'], allow_pickle=False) as inp:
        values = driver.integrate(spec['parameters'], inp['current'], spec['rate_hz'],
                                  inp['initial'], spec['bits'], spec['factor'])
    if any(digest(p) != h for p, h in spec['source_hashes'].items()):
        raise ValueError('worker provenance changed during integration')
    with Path(spec['output']).open('xb') as file:
        np.savez_compressed(file, states=values)


def main():
    reg = read_yaml(REG)
    if reg['status'] != 'registered-before-biological-execution':
        raise ValueError('diagnostic execution registration is not sealed')
    for p, h in reg['prerequisite_hashes'].items():
        if digest(p) != h:
            raise ValueError('precision diagnostic prerequisite changed')
    previous = json.loads(Path(reg['predecessor']).read_text())
    pins = dict(previous['source_hashes'])
    for p, h in pins.items():
        if digest(p) != h:
            raise ValueError('predecessor source chain changed')
    pins.update(reg['prerequisite_hashes'])
    for name in reg['source_files']:
        pins[name] = digest(name)
    pins[str(REG)] = digest(REG)
    original = json.loads(Path(reg['original']).read_text())
    parameters = original['strata']['Pvalb-L23']['candidates']['19']['parameters']
    protocol = read_yaml('docs/validation-results/post2008-patchseq-isolated-fit-registration-1436.yaml')
    if any(version(k) != v for k, v in protocol['runtime'].items()):
        raise ValueError('precision diagnostic runtime differs from frozen protocol')
    if digest(protocol['training_catalog']) != protocol['training_catalog_sha256']:
        raise ValueError('training catalog changed')
    keys = {tuple(k) for k in reg['source_keys']}
    records = restricted_records(fitting_records(read_yaml(protocol['training_catalog']), 'Pvalb-L23'), keys)
    targets = {(r['source_batch'], r['slot'], s['sweep_number']): s['features']
               for r in records for s in r['sweeps']}
    source = read_yaml('docs/validation-results/post2008-patchseq-fitting-target-registration-1429.yaml')
    output = Path(reg['output'])
    output.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        for path, mode in ((output/'runner.lock', fcntl.LOCK_EX),
                           (Path(reg['original']).parent/'runner.lock', fcntl.LOCK_SH),
                           (Path('results/patchseq-compiled-fit-1467/runner.lock'), fcntl.LOCK_SH)):
            handle = stack.enter_context(path.open('a'))
            fcntl.flock(handle, mode | fcntl.LOCK_NB)
        driver = PrecisionDriver(output/'build')
        pins.update({driver.provenance['binary_path']: driver.provenance['binary_sha256'],
                     str(output/'build'/'build.yaml'): digest(output/'build'/'build.yaml')})
        pins.update(driver.provenance['dependencies'])
        pins[driver.provenance['compiler']] = driver.provenance['compiler_sha256']
        pins[str(Path(sys.executable).resolve())] = digest(Path(sys.executable).resolve())
        path = output/'checkpoint.json'
        base = {'source_hashes': pins, 'build': driver.provenance,
                'precision_probes': json.loads(json.dumps(driver.probes)),
                'parameters': parameters, 'cases': {}, 'inputs': {},
                'held_out_response_features_read': False, 'fitting_resumed': False}
        state = json.loads(path.read_text()) if path.exists() else base
        if any(state[k] != v for k, v in base.items() if k not in ('cases', 'inputs')):
            raise ValueError('precision checkpoint provenance differs')
        expected = {f'{batch}-{slot}-{number}/{bits}-{factor}' for batch, slot, number in keys
                    for bits in (53, 160, 320) for factor in (4, 8, 16)}
        if len(expected) != 18 or not set(state['cases']) <= expected:
            raise ValueError('precision checkpoint case table differs')
        inputs, ledger, _ = inputs_for_training(records, source, read_yaml(source['nominal']))
        state['opened_training_command_files'] = ledger
        for key in map(tuple, reg['source_keys']):
            label = '-'.join(map(str, key))
            command, rate, holding, epoch = inputs[key]
            warm = round(rate)
            if warm != rate:
                raise ValueError('nonintegral one-second warmup')
            initial = shared_brian_initial_state(parameters, rate)
            nominal = command+holding
            current = np.r_[np.full(warm, nominal[0]), nominal]
            inp_path = output/(label+'--input.npz')
            if inp_path.exists():
                with np.load(inp_path, allow_pickle=False) as retained:
                    if (not np.array_equal(retained['current'], current)
                        or not np.array_equal(retained['initial'], initial)):
                        raise ValueError('retained current or shared initial state changed')
            else:
                with inp_path.open('xb') as file:
                    np.savez_compressed(file, current=current, initial=initial)
            receipt = {'path': str(inp_path), 'sha256': digest(inp_path), 'warmup_samples': warm,
                       'rate_hz': rate, 'holding_pA': holding, 'initial': initial.tolist(),
                       'current_sha256': hashlib.sha256(current.tobytes()).hexdigest()}
            if label in state['inputs'] and state['inputs'][label] != receipt:
                raise ValueError('precision input receipt differs')
            state['inputs'][label] = receipt
            checkpoint(path, state)
            for bits in (53, 160, 320):
                for factor in (4, 8, 16):
                    case_id = f'{label}/{bits}-{factor}'
                    if case_id in state['cases']:
                        row = state['cases'][case_id]
                        if row['state'] == 'retained' and digest(row['arrays']) != row['array_sha256']:
                            raise ValueError('retained precision output changed')
                        continue
                    array = output/f'{label}--{bits}-{factor}.npz'
                    spec_path = output/f'{label}--{bits}-{factor}.json'
                    log = output/f'{label}--{bits}-{factor}.log'
                    if any(p.exists() for p in (array, spec_path, log)):
                        raise ValueError('orphan case artifact retained; refusing overwrite or automatic rerun')
                    spec = {'source_hashes': pins, 'input': str(inp_path), 'input_sha256': digest(inp_path),
                            'build_directory': str(output/'build'), 'build': driver.provenance,
                            'parameters': parameters, 'rate_hz': rate, 'bits': bits, 'factor': factor,
                            'output': str(array)}
                    with spec_path.open('x') as file:
                        json.dump(spec, file, sort_keys=True)
                    try:
                        with log.open('xb') as logfile:
                            subprocess.run([sys.executable, __file__, '--worker', str(spec_path)],
                                check=True, stdout=logfile, stderr=subprocess.STDOUT,
                                timeout=reg['maximum_case_seconds'])
                        with np.load(array, allow_pickle=False) as saved:
                            values = saved['states']
                        features = characterize(values[warm:, 0], command, rate, epoch, holding)
                        row = {'state': 'retained', 'arrays': str(array), 'array_sha256': digest(array),
                               'features': features, 'score': score_sweep(targets[key], features)}
                    except (subprocess.SubprocessError, ValueError, FloatingPointError) as exc:
                        row = {'state': 'failed-retained-no-replacement', 'error': str(exc),
                               'orphan_output_retained': array.exists(),
                               'orphan_array_sha256': digest(array) if array.exists() else None}
                    row.update(spec_sha256=digest(spec_path), log_sha256=digest(log),
                               bits=bits, factor=factor)
                    state['cases'][case_id] = row
                    if any(digest(p) != h for p, h in pins.items()):
                        raise ValueError('precision sources changed during diagnostic')
                    checkpoint(path, state)
                    print('Precision case retained', len(state['cases']), '/18', case_id, flush=True)
        summaries = {}
        for key in map(tuple, reg['source_keys']):
            label = '-'.join(map(str, key))
            rows = {}
            for bits in (53, 160, 320):
                for factor in (4, 8, 16):
                    name = f'{bits}-{factor}'
                    row = state['cases'][label+'/'+name]
                    rows[name] = None
                    if row['state'] == 'retained':
                        with np.load(row['arrays'], allow_pickle=False) as values:
                            rows[name] = {'voltage': values['states'][:, 0],
                                          'features': row['features'], 'loss': row['score']['loss']}
            comparisons = {a+'/'+b: comparison(rows[a], rows[b]) for a, b in pairs()}
            summaries[label] = {'comparisons': comparisons, 'reference_qualified': qualified(rows, comparisons)}
        if any(digest(p) != h for p, h in pins.items()):
            raise ValueError('precision sources changed before terminal snapshot')
        state.update(status=TERMINAL, summaries=summaries, production_transition_authorized=False,
                     scientific_goal_complete=False, frozen_baseline_modified=False)
        checkpoint(path, state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path)
    options = parser.parse_args()
    if options.worker:
        worker(options.worker)
    else:
        main()
