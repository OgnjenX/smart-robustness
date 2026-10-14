"""Prospective synthetic Brian2/compiled fitting comparison; no recording access."""

from __future__ import annotations

import fcntl
import json
from pathlib import Path

import numpy as np
import yaml
from patchseq_compiled_fit_engine import adapter, numerical_controls
from patchseq_fit_objective import score_sweep
from patchseq_fit_search import candidate_dict, candidates
from patchseq_fit_search import numerical_controls as primary_controls
from patchseq_fit_search import replay as primary_replay
from patchseq_hh_compiled_driver import compiled_driver
from patchseq_isolated_hh import Parameters
from patchseq_response_features import characterize
from patchseq_selected_fit_agreement import feature_agreement
from run_patchseq_isolated_fit import checkpoint
from verify_patchseq_isolated_fit_replay import close
from verify_patchseq_whole_file_integrity import digest

REG = Path('docs/validation-results/post2008-patchseq-compiled-fitting-equivalence-registration-1463.yaml')
FAILURES = (ValueError, FloatingPointError, OverflowError)


def synthetic_input(config, rate, amplitude):
    command = np.zeros(round(rate*config['duration_s']))
    start, stop = round(rate*config['step_start_s']), round(rate*config['step_stop_s'])
    command[start:stop] = amplitude
    epoch = {'step_start_index': start, 'step_stop_index': stop,
             'pre_window': [round(rate*t) for t in config['pre_window_s']]}
    return command, epoch


def traces(engine, rows, command, rate, holding, warmup):
    """Same candidate isolation on batch failure as the original fitting runner."""
    try:
        result = engine(rows, command, rate, holding, warmup)
        return [(v, None) for v in result]
    except FAILURES:
        result = []
        for row in rows:
            try:
                result.append((engine([row], command, rate, holding, warmup)[0], None))
            except FAILURES as exc:
                result.append((None, str(exc)))
        return result


def compare(left, right, command, rate, holding, epoch):
    a, ae = left
    b, be = right
    if a is None or b is None:
        return {'passed': (a is None) == (b is None), 'primary_error': ae,
                'compiled_error': be, 'failure_support_matches': (a is None) == (b is None)}
    af, bf = [characterize(v, command, rate, epoch, holding) for v in (a, b)]
    target = {'baseline_mean_mV': -65., 'early_step_mean_mV': -70.,
              'steady_step_mean_mV': -72., 'detected_event_count': 0,
              'first_event_latency_s': None, 'last_to_first_interval_ratio': None}
    al, bl = [score_sweep(target, f) for f in (af, bf)]
    difference = float(np.max(np.abs(a-b)))
    checks = {'waveform': difference <= 1e-5, 'features': feature_agreement(af, bf),
              'loss': close(al['loss'], bl['loss']), 'objective_gates': al['gates'] == bl['gates']}
    return {'passed': all(checks.values()), 'checks': checks, 'failure_support_matches': True,
            'maximum_voltage_difference_mV': difference,
            'primary_error': None, 'compiled_error': None}


def main():
    reg = yaml.safe_load(REG.read_text())
    if digest(reg['fit_protocol']) != reg['fit_protocol_sha256']:
        raise ValueError('original candidate protocol changed')
    fit = yaml.safe_load(Path(reg['fit_protocol']).read_text())
    seal = yaml.safe_load(Path(reg['execution_seal']).read_text())
    pins = dict(seal['source_sha256'])
    for name in ('patchseq_compiled_fit_engine.py', 'patchseq_hh_compiled_driver.py',
                 'patchseq_hh_replay_kernel.cpp', 'patchseq_selected_fit_agreement.py',
                 'patchseq_response_features.py', 'patchseq_hh_independent_replay.py',
                 'patchseq_hh_compiled_prototype.py', 'verify_patchseq_isolated_fit_replay.py'):
        p = Path('scripts')/name
        pins[str(p)] = digest(p)
    pins.update({str(REG): digest(REG), str(Path(__file__)): digest(__file__)})
    if any(digest(p) != h for p, h in pins.items()):
        raise ValueError('sealed source changed')
    output = Path(reg['output'])
    output.mkdir(parents=True, exist_ok=True)
    with (output/'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with compiled_driver(output/'build') as (compiled, build):
            engine = adapter(compiled)
            path = output/'checkpoint.json'
            state = json.loads(path.read_text()) if path.exists() else {
                'source_hashes': pins, 'compiled_build': build, 'cases': {},
                'biological_data_used': False, 'live_fitting_source_modified': False}
            if state['source_hashes'] != pins or state['compiled_build'] != build:
                raise ValueError('equivalence checkpoint provenance differs')
            rows = candidates(fit['search'])
            config = reg['short']
            rate, width = config['rate_hz'], config['batch_maximum']
            for amplitude in config['currents_pA']:
                command, epoch = synthetic_input(config, rate, amplitude)
                for start in range(0, len(rows), width):
                    ids = list(range(start, min(start+width, len(rows))))
                    keys = [f'short/{amplitude}/{i}' for i in ids]
                    if all(k in state['cases'] for k in keys):
                        continue
                    if any(k in state['cases'] for k in keys):
                        raise ValueError('partial synthetic batch')
                    args = (rows[start:start+width], command, rate,
                            config['holding_pA'], config['warmup_s'])
                    a, b = traces(primary_replay, *args), traces(engine, *args)
                    for i, key, left, right in zip(ids, keys, a, b, strict=True):
                        state['cases'][key] = {'candidate_index': i, 'parameters': candidate_dict(rows[i]),
                                              **compare(left, right, command, rate, config['holding_pA'], epoch)}
                    checkpoint(path, state)
                    print('Direct Brian2 equivalence cases', len(state['cases']), '/ 386', flush=True)
            config = reg['long']
            row = Parameters(**config['parameters'])
            for rate in config['rates_hz']:
                key = f'long/{rate}'
                if key in state['cases']:
                    continue
                command, epoch = synthetic_input(config, rate, config['current_pA'])
                args = ([row], command, rate, config['holding_pA'], config['warmup_s'])
                result = compare(traces(primary_replay, *args)[0], traces(engine, *args)[0],
                                 command, rate, config['holding_pA'], epoch)
                controls = []
                for function in (primary_controls, lambda *a: numerical_controls(engine, *a)):
                    try:
                        controls.append(function(row, command, rate, config['holding_pA'],
                                                 epoch, config['warmup_s']))
                    except FAILURES as exc:
                        controls.append({'error': str(exc)})
                agree = ('error' not in controls[0] and 'error' not in controls[1]
                         and feature_agreement(*controls))
                state['cases'][key] = {**result, 'controls': controls, 'controls_agree': agree,
                                       'passed': result['passed'] and agree}
                checkpoint(path, state)
            if any(digest(p) != h for p, h in pins.items()):
                raise ValueError('sources changed during equivalence checks')
            state.update(status='direct-Brian2-compiled-fitting-synthetic-equivalence-retained',
                         all_cases_pass=len(state['cases']) == 386 and all(
                             v['passed'] for v in state['cases'].values()),
                         fitting_transition_authorized=False, held_out_access_authorized=False,
                         network_promotion=False, frozen_baseline_modified=False)
            checkpoint(path, state)


if __name__ == '__main__':
    main()
