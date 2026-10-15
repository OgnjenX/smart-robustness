"""One predetermined positive training command; solver/refinement diagnosis only."""

from __future__ import annotations

import fcntl
import json
from pathlib import Path

import numpy as np
from patchseq_fit_objective import score_sweep
from patchseq_fit_search import replay as brian_replay
from patchseq_hh_compiled_driver import compiled_driver
from patchseq_hh_independent_replay import replay as numpy_replay
from patchseq_isolated_hh import Parameters
from patchseq_response_features import characterize
from run_patchseq_isolated_fit import checkpoint, fitting_records, prepare_inputs, read_yaml
from verify_patchseq_whole_file_integrity import digest

REG = Path('docs/validation-results/post2008-patchseq-mismatch-numerics-registration-1480.yaml')


def comparison(a, b, rate):
    error = np.abs(a-b)
    first = {}
    for threshold in (1e-10, 1e-8, 1e-5, .1, 1., 10.):
        positions = np.flatnonzero(error > threshold)
        first[str(threshold)] = int(positions[0])/rate if len(positions) else None
    return {'maximum_difference_mV': float(np.max(error)),
            'first_error_above_threshold_seconds': first}


def main():
    reg = read_yaml(REG)
    if digest(reg['original_checkpoint']) != reg['original_sha256']:
        raise ValueError('original snapshot changed')
    predecessor = json.loads(Path(reg['predecessor']).read_text())
    if predecessor.get('status') != 'fixed-candidate-mismatch-diagnostic-retained-not-a-fitting-result':
        raise ValueError('prior diagnostic incomplete')
    pins = dict(predecessor['source_hashes'])
    if any(digest(p) != h for p, h in pins.items()):
        raise ValueError('prior source chain differs')
    pins.update({str(REG): digest(REG), str(Path(__file__)): digest(__file__),
                 reg['predecessor']: digest(reg['predecessor']),
                 'scripts/patchseq_hh_independent_replay.py': digest('scripts/patchseq_hh_independent_replay.py')})
    original = json.loads(Path(reg['original_checkpoint']).read_text())
    parameter = original['strata'][reg['stratum']]['candidates'][str(reg['candidate_index'])]['parameters']
    protocol = read_yaml('docs/validation-results/post2008-patchseq-isolated-fit-registration-1436.yaml')
    records = fitting_records(read_yaml(protocol['training_catalog']), reg['stratum'])
    source = read_yaml('docs/validation-results/post2008-patchseq-fitting-target-registration-1429.yaml')
    output = Path(reg['output']); output.mkdir(parents=True, exist_ok=True)
    with (output/'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = output/'checkpoint.json'
        if path.exists():
            raise ValueError('diagnostic evidence exists; refusing overwrite')
        inputs, _, _ = prepare_inputs(records, source, read_yaml(source['nominal']))
        key = min(k for k, (c, _, _, e) in inputs.items() if c[e['step_start_index']] > c[0])
        command, rate, holding, epoch = inputs[key]
        target = next(s['features'] for r in records for s in r['sweeps']
                      if (r['source_batch'], r['slot'], s['sweep_number']) == key)
        state = {'source_hashes': pins, 'source_key': list(key), 'rate_hz': rate,
                 'candidate_parameters': parameter, 'replays': {}, 'comparisons': {},
                 'production_protocol_changed': False, 'fitting_resumed': False,
                 'held_out_response_features_read': False}
        arrays = {}
        with compiled_driver(output/'build') as (compiled, build):
            state['compiled_build'] = build
            pins[build['binary_path']] = build['binary_sha256']
            pins[str(output/'build'/'build.yaml')] = digest(output/'build'/'build.yaml')
            runs = [('compiled-dt', compiled, parameter, 1), ('numpy-dt', numpy_replay, parameter, 1),
                    ('brian-dt', brian_replay, Parameters(**parameter), 1),
                    ('compiled-half-dt', compiled, parameter, 2),
                    ('brian-half-dt', brian_replay, Parameters(**parameter), 2)]
            for name, engine, p, refinement in runs:
                try:
                    voltage = engine([p], command, rate, holding, 1., refinement=refinement)[0]
                    arrays[name] = voltage
                    features = characterize(voltage, command, rate, epoch, holding)
                    state['replays'][name] = {'features': features, 'score': score_sweep(target, features)}
                    np.savez_compressed(output/(name+'.npz'), predicted_mV=voltage)
                    state['replays'][name]['array_sha256'] = digest(output/(name+'.npz'))
                except (ValueError, FloatingPointError, OverflowError) as exc:
                    state['replays'][name] = {'error': str(exc)}
                checkpoint(path, state)
                print('Numerical diagnostic retained', name, flush=True)
            for a, b in [('compiled-dt', 'numpy-dt'), ('compiled-dt', 'brian-dt'),
                         ('compiled-half-dt', 'brian-half-dt'), ('brian-dt', 'brian-half-dt'),
                         ('compiled-dt', 'compiled-half-dt')]:
                if a in arrays and b in arrays:
                    state['comparisons'][a+'/'+b] = comparison(arrays[a], arrays[b], rate)
            if any(digest(p) != h for p, h in pins.items()):
                raise ValueError('source chain changed during numerical diagnosis')
            state['status'] = 'fixed-input-numerical-localization-retained-no-production-change'
            checkpoint(path, state)


if __name__ == '__main__':
    main()
