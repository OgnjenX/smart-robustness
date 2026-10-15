"""Prospective long synthetic backend comparison; no biological file reader.

Preparation only until an execution seal also pins an independent verifier.
Completed cases are resumable; interrupted cases are never silently rerun.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import yaml
from patchseq_fit_objective import score_sweep
from patchseq_mpfr_driver import MPFRDriver
from patchseq_precision_driver import FIELDS, PrecisionDriver, digest
from patchseq_response_features import characterize

REG = Path('docs/validation-results/post2008-patchseq-mpfr-long-synthetic-registration-1506.yaml')
SEAL = Path('docs/validation-results/post2008-patchseq-mpfr-long-synthetic-execution-seal-1507.yaml')
TERMINAL = 'synthetic-matrix-retained-awaiting-independent-assessment'


def case_table():
    rows = []
    for samples in (151000, 330000):
        for profile in ('passive', 'active'):
            for engine in ('boost', 'mpfr'):
                for bits in ((53, 160, 320) if engine == 'boost' else (160, 320)):
                    for factor in (4, 8, 16):
                        rows.append({'samples': samples, 'profile': profile, 'engine': engine,
                                     'bits': bits, 'factor': factor, 'repeat': 0})
            for bits in (160, 320):
                rows.append({'samples': samples, 'profile': profile, 'engine': 'mpfr',
                             'bits': bits, 'factor': 16, 'repeat': 1})
    return rows


def label(row):
    return '-'.join(str(row[k]) for k in ('samples', 'profile', 'engine', 'bits', 'factor', 'repeat'))


def synthetic_input(samples, profile):
    if samples not in (151000, 330000) or profile not in ('passive', 'active'):
        raise ValueError('outside fixed synthetic input table')
    parameters = dict(zip(FIELDS, [100., 10., -65., 2000., 500., 10., -60., 1000.], strict=True))
    if profile == 'passive':
        for name in ('sodium_nS', 'potassium_nS', 'm_current_nS'):
            parameters[name] = 0.
    # Command excludes exactly one second of added warmup. Release lasts 0.5 s.
    command = np.zeros(samples)
    start, stop, warm = 25000, samples-25000, 50000
    command[start:stop] = -50. if profile == 'passive' else 2000.
    current = np.r_[np.zeros(warm), command]
    epoch = {'step_start_index': start, 'step_stop_index': stop, 'pre_window': [15000, 20000]}
    return parameters, current, np.array([-65., .05, .6, .2, .05]), command, epoch


def atomic_checkpoint(path, data):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as file:
        json.dump(data, file, sort_keys=True, allow_nan=False)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def verify_pins(pins):
    if any(digest(p) != h for p, h in pins.items()):
        raise ValueError('synthetic source/build/input provenance changed')


def budget_allows_case(started, invocation_seconds, maximum_case_seconds):
    # Reserve a full timeout plus receipt/cleanup allowance before starting.
    return time.monotonic()-started+maximum_case_seconds+60 <= invocation_seconds


def worker(spec_path):
    spec = json.loads(Path(spec_path).read_text())
    verify_pins(spec['pins'])
    row = spec['case']
    with (Path(spec['output']).parent/'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        engine = (PrecisionDriver if row['engine'] == 'boost' else MPFRDriver)(spec['build'])
        if engine.provenance != spec['provenance']:
            raise ValueError('worker build receipt differs')
        p, current, initial, command, epoch = synthetic_input(row['samples'], row['profile'])
        started = time.perf_counter()
        states = engine.integrate(p, current, 50000., initial, row['bits'], row['factor'])
        integration_seconds = time.perf_counter()-started
        features = characterize(states[50000:, 0], command, 50000., epoch, 0.)
        target = {'baseline_mean_mV': -65., 'early_step_mean_mV': -60., 'steady_step_mean_mV': -55.,
                  'detected_event_count': 3, 'first_event_latency_s': .01, 'last_to_first_interval_ratio': 1.}
        score = score_sweep(target, features)
        compression_started = time.perf_counter()
        with Path(spec['output']).open('xb') as file:
            np.savez_compressed(file, states=states)
        compression_seconds = time.perf_counter()-compression_started
        verify_pins(spec['pins'])
        with Path(spec['receipt']).open('x') as file:
            json.dump({'integration_seconds': integration_seconds, 'compression_seconds': compression_seconds,
                       'features': features, 'score': score, 'shape': list(states.shape),
                       'current_sha256': hashlib.sha256(current.tobytes()).hexdigest(),
                       'arrays_sha256': digest(spec['output']),
                       'compressed_bytes': Path(spec['output']).stat().st_size,
                       'uncompressed_bytes': states.nbytes}, file, sort_keys=True, allow_nan=False)


def validate_resume(state, base):
    if any(state.get(k) != v for k, v in base.items() if k not in ('cases', 'status')):
        raise ValueError('synthetic execution configuration changed')
    expected = {label(row) for row in base['table']}
    if not set(state['cases']) <= expected:
        raise ValueError('extra synthetic case')
    for row in state['cases'].values():
        if row['status'] == 'running':
            raise ValueError('interrupted case retained: inspect process/artifacts; no automatic rerun')
        if row['status'] not in ('retained', 'failure-retained-no-replacement'):
            raise ValueError('unsupported case state')
        verify_pins(row['artifacts'])


def run():
    reg = yaml.safe_load(REG.read_text())
    seal = yaml.safe_load(SEAL.read_text())
    if seal.get('status') != 'sealed-before-long-synthetic-execution':
        raise ValueError('missing prospective execution seal')
    invocation_seconds = seal.get('maximum_invocation_seconds')
    if (type(invocation_seconds) is not int or not 3660 <= invocation_seconds <= 86400
        or seal.get('human_resource_budget_confirmed') is not True):
        raise ValueError('bounded execution budget must be confirmed and prospectively sealed')
    started = time.monotonic()
    pins = seal['source_hashes']
    if pins.get(str(REG)) != digest(REG) or pins.get(str(Path(__file__).relative_to(Path.cwd()))) != digest(__file__):
        raise ValueError('registration/runner not pinned by seal')
    verify_pins(pins)
    if not any('verify_patchseq_mpfr_synthetic.py' in p for p in pins):
        raise ValueError('independent verifier must be pinned before execution')
    if reg['case_table'] != case_table() or reg['total_cases'] != 68:
        raise ValueError('fixed synthetic matrix differs')
    output = Path(reg['output'])
    output.mkdir(parents=True, exist_ok=True)
    with (output/'runner.lock').open('a') as lock, (output/'worker.lock').open('a') as worker_lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # A live orphan worker blocks a replacement controller before any builds.
        fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(worker_lock, fcntl.LOCK_UN)
        drivers = {'boost': PrecisionDriver(output/'boost-build'), 'mpfr': MPFRDriver(output/'mpfr-build')}
        pins = dict(pins)
        pins[str(SEAL)] = digest(SEAL)
        for name, driver in drivers.items():
            provenance = driver.provenance
            pins.update(provenance.get('dependencies', provenance.get('source_hashes', {})))
            pins[provenance['compiler']] = digest(provenance['compiler'])
            pins[provenance['binary_path']] = provenance['binary_sha256']
            receipt = output/(name+'-build')/'build.yaml'
            pins[str(receipt)] = digest(receipt)
        pins[str(Path(sys.executable).resolve())] = digest(Path(sys.executable).resolve())
        base = {'pins': pins, 'table': case_table(), 'builds': {k: d.provenance for k, d in drivers.items()},
                'runtime': {k: version(k) for k in ('numpy', 'scipy', 'PyYAML')},
                'cases': {}, 'status': 'running-synthetic-only', 'biological_data_access': False,
                'fitting_restart_authorized': False, 'scientific_goal_complete': False}
        path = output/'checkpoint.json'
        state = json.loads(path.read_text()) if path.exists() else base
        validate_resume(state, base)
        verify_pins(pins)
        atomic_checkpoint(path, state)
        for row in case_table():
            key = label(row)
            if key in state['cases']:
                continue
            if not budget_allows_case(started, invocation_seconds, reg['maximum_case_seconds']):
                state['status'] = 'budget-boundary-checkpoint-not-terminal-matrix'
                atomic_checkpoint(path, state)
                print('Bounded invocation complete; remaining cases retained as pending, not failures', flush=True)
                return
            files = {suffix: output/(key+suffix) for suffix in ('.json', '.log', '.npz', '.receipt.json')}
            if any(p.exists() for p in files.values()):
                raise ValueError('orphan synthetic artifact retained; refusing overwrite')
            spec = {'case': row, 'pins': pins, 'build': str(output/(row['engine']+'-build')),
                    'provenance': drivers[row['engine']].provenance,
                    'output': str(files['.npz']), 'receipt': str(files['.receipt.json'])}
            with files['.json'].open('x') as file:
                json.dump(spec, file, sort_keys=True)
            state['cases'][key] = {'status': 'running', 'artifacts': {str(files['.json']): digest(files['.json'])}}
            atomic_checkpoint(path, state)
            began = time.perf_counter()
            try:
                with files['.log'].open('xb') as logfile:
                    subprocess.run([sys.executable, __file__, '--worker', str(files['.json'])], check=True,
                                   stdout=logfile, stderr=subprocess.STDOUT, timeout=reg['maximum_case_seconds'])
                receipt = json.loads(files['.receipt.json'].read_text())
                if receipt['arrays_sha256'] != digest(files['.npz']):
                    raise ValueError('worker array hash differs')
                result = {'status': 'retained', 'receipt': receipt}
            except (subprocess.SubprocessError, ValueError, FileNotFoundError) as exc:
                result = {'status': 'failure-retained-no-replacement', 'error': str(exc)}
            result.update(total_case_seconds=time.perf_counter()-began,
                          artifacts={str(p): digest(p) for p in files.values() if p.is_file()})
            state['cases'][key] = result
            verify_pins(pins)
            atomic_checkpoint(path, state)
            print('Synthetic case retained', len(state['cases']), '/68', key, result['status'], flush=True)
        state['status'] = TERMINAL
        atomic_checkpoint(path, state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', action='store_true', help='Print fixed case table; no builds or execution')
    parser.add_argument('--worker', type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    elif args.plan:
        print(json.dumps(case_table(), indent=2))
    else:
        run()
