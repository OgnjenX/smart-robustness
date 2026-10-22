"""Bounded synthetic pilot controller/worker. Preparation, NOT execution authority.

Every entry to a numerical worker validates a separate approved execution seal.
Only clean controller boundaries can resume; interrupted work is never replayed.
The independent simulated-array auditor must exist and be pinned before builds.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import scipy
import yaml
from patchseq_fit_objective import score_sweep
from patchseq_gbs_driver import Policy
from patchseq_gbs_hh_driver import HHDriver
from patchseq_gbs_pilot_protocol import (
    EXECUTION_SEAL,
    REGISTRATION,
    case_table,
    comparison_table,
    encoded,
    file_sha,
    input_manifest,
    input_signature,
    label,
    sha,
    synthetic_input,
    validate_case,
)
from patchseq_response_features import characterize
from patchseq_rkf78_reference import RKF78Driver
from verify_patchseq_gbs_pilot_protocol import validate_registration

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
REG = ROOT/REGISTRATION
SEAL = ROOT/EXECUTION_SEAL
PREPARATION = ROOT/'results/patchseq-gbs-pilot-build-preparation-1520'
AUDITOR = ROOT/'scripts/verify_patchseq_gbs_pilot.py'
TERMINAL = 'pilot-retained-awaiting-independent-audit'
STOPPED = 'pilot-stopped-after-execution-failure'
EXHAUSTED = 'pilot-budget-exhausted-incomplete'
TERMINALS = (TERMINAL, STOPPED, EXHAUSTED)
MAX_JSON_BYTES = 64*1024*1024
WORKER_ENV = {'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
    'VECLIB_MAXIMUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1', 'HDF5_PLUGIN_PRELOAD': '::',
    'MPLCONFIGDIR': '/private/tmp/mpl', 'PYTHONDONTWRITEBYTECODE': '1'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_pins(pins):
    require(type(pins) is dict and bool(pins), 'missing pilot provenance pins')
    for path, expected in pins.items():
        require(type(path) is str and type(expected) is str and file_sha(path) == expected,
                'pilot source/build/runtime/artifact provenance changed: '+str(path))


def read_json(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= MAX_JSON_BYTES,
            'missing, linked or oversized pilot JSON: '+str(path))
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate pilot JSON key')
            result[key] = value
        return result
    data = json.loads(path.read_text(), object_pairs_hook=unique)
    encoded(data)  # Reject nonfinite JSON numbers, including exponent overflow.
    return data


def write_json(path, value, *, atomic=False):
    path = Path(path)
    require(not path.is_symlink(), 'linked pilot JSON refused: '+str(path))
    raw = (encoded(value)+'\n').encode()
    require(len(raw) <= MAX_JSON_BYTES, 'pilot JSON exceeds its metadata bound')
    target = path.with_suffix('.tmp') if atomic else path
    require(not target.exists() and not target.is_symlink(), 'orphan pilot JSON retained: '+str(target))
    with target.open('xb') as file:
        file.write(raw)
        file.flush()
        os.fsync(file.fileno())
    if atomic:
        os.replace(target, path)


def repository_pins():
    # The array auditor is deliberately required even for build preparation.
    relative = [REGISTRATION, 'scripts/run_patchseq_gbs_pilot.py', 'scripts/verify_patchseq_gbs_pilot.py',
        'scripts/patchseq_gbs_pilot_array_audit.py',
        'scripts/patchseq_gbs_pilot_protocol.py', 'scripts/verify_patchseq_gbs_pilot_protocol.py',
        'scripts/patchseq_gbs_hh_driver.py', 'scripts/patchseq_gbs_hh.cpp',
        'scripts/patchseq_gbs_guarded_core.hpp', 'scripts/patchseq_gbs_core.hpp',
        'scripts/patchseq_gbs_mpfr_number.hpp', 'scripts/patchseq_gbs_odeint_bridge.hpp',
        'scripts/third_party/patchseq_bounded_gbs_dense.hpp', 'scripts/third_party/BOOST_LICENSE_1_0.txt',
        'scripts/patchseq_gbs_driver.py', 'scripts/patchseq_mpfr_driver.py', 'scripts/patchseq_mpfr_kernel.cpp',
        'scripts/patchseq_rkf78_reference.cpp', 'scripts/patchseq_rkf78_reference.py',
        'scripts/patchseq_response_features.py', 'scripts/patchseq_fit_objective.py',
        'scripts/verify_patchseq_fitting_targets.py', 'scripts/verify_patchseq_whole_file_integrity.py']
    return {str(ROOT/path): file_sha(ROOT/path) for path in relative}


def runtime_fingerprint():
    roots = {Path(module.__file__).resolve().parent for module in (np, scipy, yaml)}
    for module in (np, scipy):
        extra = Path(module.__file__).resolve().parent.parent/(module.__name__+'.libs')
        if extra.is_dir():
            roots.add(extra)
    files = {str(Path(sys.executable).resolve())}
    for root in roots:
        for path in root.rglob('*'):
            if (path.is_file() and '__pycache__' not in path.parts
                and (path.suffix in ('.py', '.dylib', '.pyd', '.dll', '.h', '.a') or '.so' in path.name)):
                files.add(str(path.resolve()))
    return {'python': sys.version, 'platform': platform.platform(),
        'versions': {name: version(name) for name in ('numpy', 'scipy', 'PyYAML')},
        'required_worker_environment': dict(WORKER_ENV),
        'file_hashes': {path: file_sha(path) for path in sorted(files)}}


def merge_pins(*groups):
    result = {}
    for group in groups:
        for path, value in group.items():
            require(path not in result or result[path] == value, 'conflicting pilot pin')
            result[path] = value
    return result


def read_preparation():
    require(not PREPARATION.is_symlink(), 'linked pilot build preparation refused')
    document = read_json(PREPARATION/'preparation.json')
    require(type(document) is dict and set(document) == {'status', 'source_hashes', 'runtime', 'pins',
        'builds', 'preparation_seconds', 'biological_access', 'pilot_executed', 'fitting_restart'},
        'invalid pilot build preparation schema')
    require(type(document['builds']) is dict and set(document['builds']) == {'gbs', 'rkf78'},
            'pilot build support differs')
    require(document['status'] == 'built-and-pinned-no-pilot-trajectories', 'invalid pilot build preparation')
    require(document['source_hashes'] == repository_pins(), 'prepared pilot sources differ')
    require(document['runtime'] == runtime_fingerprint(), 'prepared pilot runtime differs')
    require(document['pilot_executed'] is False and document['biological_access'] is False
        and document['fitting_restart'] is False, 'build preparation claims an unauthorized operation')
    require(type(document['preparation_seconds']) is float and math.isfinite(document['preparation_seconds'])
        and document['preparation_seconds'] >= 0, 'invalid build preparation timing')
    expected_pins = merge_pins(document['source_hashes'], document['runtime']['file_hashes'])
    for engine in ('gbs', 'rkf78'):
        build = document['builds'][engine]
        root = PREPARATION/(engine+'-build')
        require(build['binary_path'] == str((root/('hh.so' if engine == 'gbs' else 'reference.so')).resolve()),
                'prepared binary path differs')
        require(read_json(root/'build.json') == build, 'prepared build receipt differs')
        expected_pins = merge_pins(expected_pins, build['source_hashes'],
            {build['binary_path']: build['binary_sha256'], str(root/'build.json'): file_sha(root/'build.json')})
    require(document['pins'] == expected_pins, 'incomplete or altered preparation pin support')
    verify_pins(expected_pins)
    return document


def prepare_builds():
    require(not PREPARATION.is_symlink() and not (PREPARATION/'prepare.lock').is_symlink(),
            'linked pilot build preparation or lock refused')
    reg = yaml.safe_load(REG.read_text())
    validate_registration(reg)
    sources = repository_pins()  # Fails before any build while array auditor is absent.
    if (PREPARATION/'preparation.json').exists():
        return read_preparation()
    PREPARATION.mkdir(parents=True, exist_ok=True)
    with (PREPARATION/'prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require({p.name for p in PREPARATION.iterdir()} <= {'prepare.lock'},
                'orphan pilot build preparation retained; no automatic repair')
        began = time.perf_counter()
        drivers = {'gbs': HHDriver(PREPARATION/'gbs-build'), 'rkf78': RKF78Driver(PREPARATION/'rkf78-build')}
        runtime = runtime_fingerprint()
        pins = merge_pins(sources, runtime['file_hashes'])
        for name, driver in drivers.items():
            pins = merge_pins(pins, driver.provenance['source_hashes'],
                {driver.provenance['binary_path']: driver.provenance['binary_sha256'],
                 str(PREPARATION/(name+'-build')/'build.json'): file_sha(PREPARATION/(name+'-build')/'build.json')})
        verify_pins(pins)
        document = {'status': 'built-and-pinned-no-pilot-trajectories', 'source_hashes': sources,
            'runtime': runtime, 'pins': pins, 'builds': {name: driver.provenance for name, driver in drivers.items()},
            'preparation_seconds': time.perf_counter()-began,
            'biological_access': False, 'pilot_executed': False, 'fitting_restart': False}
        write_json(PREPARATION/'preparation.json', document)
        return document


def load_authorization():
    reg = yaml.safe_load(REG.read_text())
    validate_registration(reg)
    # No preparation/build/model call is reached without this separate file.
    seal = yaml.safe_load(SEAL.read_text())
    require(type(seal) is dict and set(seal) == {'schema_version', 'status', 'human_execution_approved',
        'human_approval_record', 'registration_sha256', 'preparation_path', 'preparation_sha256',
        'resources', 'allowed_operations', 'source_hashes'}, 'invalid pilot execution seal schema')
    require(type(seal['schema_version']) is int and seal['schema_version'] == 1
        and seal['status'] == 'sealed-before-short-synthetic-pilot'
        and seal['human_execution_approved'] is True
        and type(seal['human_approval_record']) is str and bool(seal['human_approval_record'].strip()),
        'pilot requires a distinct human execution approval and source seal')
    require(seal['registration_sha256'] == file_sha(REG) and encoded(seal['resources']) == encoded(reg['resources'])
        and encoded(seal['allowed_operations']) == encoded(['short-synthetic-pilot', 'independent-pilot-audit']),
        'pilot approval scope or resource budget differs')
    require(seal['preparation_path'] == str((PREPARATION/'preparation.json').relative_to(ROOT))
        and seal['preparation_sha256'] == file_sha(PREPARATION/'preparation.json'),
        'pilot build preparation seal differs')
    prep = read_preparation()
    require(seal['source_hashes'] == prep['source_hashes'], 'pilot sealed source set differs')
    pins = merge_pins(prep['pins'], {str(SEAL): file_sha(SEAL),
                                    str(PREPARATION/'preparation.json'): file_sha(PREPARATION/'preparation.json')})
    return {'reg': reg, 'prep': prep, 'pins': pins, 'registration_sha256': file_sha(REG),
            'seal_sha256': file_sha(SEAL), 'preparation_sha256': file_sha(PREPARATION/'preparation.json'),
            'output': ROOT/reg['output']}


def case_files(root, key):
    return {name: root/(key+suffix) for name, suffix in {
        'spec': '.spec.json', 'log': '.log', 'array': '.npz', 'receipt': '.receipt.json', 'restart': '.restart.json'}.items()}


def output_bytes(root):
    total = 0
    for path in root.iterdir():
        require(not path.is_symlink() and path.is_file(), 'unexpected linked or nested pilot artifact')
        total += path.stat().st_size
    return total


def validate_inventory(root, state):
    names = {'runner.lock', 'worker.lock', 'checkpoint.json'}
    if state['status'] in TERMINALS:
        names |= {'audit.lock', 'independent-assessment.json'}
    for key in state['cases']:
        names.update(path.name for path in case_files(root, key).values())
    for path in root.iterdir():
        require(not path.is_symlink() and path.is_file() and path.name in names,
                'unlisted, linked or nested pilot artifact retained')


def spec_for(auth, row):
    key = label(row)
    return {'schema_version': 1, 'case_key': key, 'case': row, 'input_signature': input_signature(row),
        'registration_sha256': auth['registration_sha256'], 'seal_sha256': auth['seal_sha256'],
        'preparation_sha256': auth['preparation_sha256'], 'pins': auth['pins'],
        'build_provenance': auth['prep']['builds'][row['engine']],
        'files': {name: str(path) for name, path in case_files(auth['output'], key).items()}}


def new_state(auth):
    return {'schema_version': 1, 'registration_sha256': auth['registration_sha256'],
        'seal_sha256': auth['seal_sha256'], 'preparation_sha256': auth['preparation_sha256'],
        'pins': auth['pins'], 'table': case_table(), 'input_manifest_sha256': sha(input_manifest()),
        'comparison_table_sha256': sha(comparison_table()), 'builds': auth['prep']['builds'],
        'runtime': auth['prep']['runtime'], 'status': 'running',
        'controller': {'state': 'running', 'pid': os.getpid(), 'invocation': 1,
                       'cumulative_charged_seconds': 0., 'measured_invocation_seconds': 0.},
        'cases': {label(row): {'case': row, 'status': 'pending', 'artifacts': {},
                               'case_seconds': None, 'error': None} for row in case_table()},
        'biological_access': False, 'fitting_restart': False, 'long_reference_qualified': False,
        'scientific_goal_complete': False}


def validate_receipt_binding(receipt, row, files):
    names = {'schema_version', 'case_key', 'case', 'input_signature', 'shape', 'arrays_sha256',
        'uncompressed_bytes', 'compressed_bytes', 'features', 'score', 'final_state', 'solver_statistics',
        'maximum_embedded_estimate', 'driver_open_seconds', 'forward_call_seconds',
        'checkpoint_serialization_seconds', 'postprocess_seconds', 'compression_seconds',
        'peak_process_RSS_bytes', 'scientific_goal_complete', 'biological_access', 'fitting_restart'}
    require(type(receipt) is dict and set(receipt) == names, 'worker receipt schema differs')
    signature = input_signature(row)
    require(type(receipt['schema_version']) is int and receipt['schema_version'] == 1
        and receipt['case_key'] == label(row) and encoded(receipt['case']) == encoded(row)
        and encoded(receipt['input_signature']) == encoded(signature)
        and encoded(receipt['shape']) == encoded(signature['shape'])
        and receipt['arrays_sha256'] == file_sha(files['array']), 'worker receipt binding differs')
    for name, expected in (('uncompressed_bytes', signature['uncompressed_array_bytes']),
                           ('compressed_bytes', files['array'].stat().st_size)):
        require(type(receipt[name]) is int and receipt[name] == expected, 'worker receipt size differs')
    for name in ('scientific_goal_complete', 'biological_access', 'fitting_restart'):
        require(receipt[name] is False, 'worker receipt makes an unauthorized claim')
    for name in ('driver_open_seconds', 'forward_call_seconds', 'checkpoint_serialization_seconds',
                 'postprocess_seconds', 'compression_seconds'):
        require(type(receipt[name]) is float and math.isfinite(receipt[name]) and receipt[name] >= 0,
                'worker receipt timing differs')
    require(type(receipt['peak_process_RSS_bytes']) is int and receipt['peak_process_RSS_bytes'] > 0,
            'worker receipt memory accounting differs')
    require((receipt['features'] is None and receipt['score'] is None) if row['role'] == 'landing' else
            (type(receipt['features']) is dict and type(receipt['score']) is dict),
            'worker feature scoring scope differs')
    # This is admission/retention checking only. Complete saved arrays, exact
    # checkpoint payloads, features/loss and every comparison require the
    # separately implemented and source-pinned independent array auditor.


def validate_state(state, auth, *, resume=False):
    expected = new_state(auth)
    require(type(state) is dict and set(state) == set(expected), 'pilot checkpoint schema differs')
    for key in expected.keys()-{'status', 'controller', 'cases'}:
        require(encoded(state[key]) == encoded(expected[key]), 'pilot checkpoint configuration differs: '+key)
    require(set(state['cases']) == set(expected['cases']), 'pilot case ledger support differs')
    controller = state['controller']
    require(set(controller) == set(expected['controller']), 'pilot controller schema differs')
    require(state['status'] in ('running', 'clean-boundary', *TERMINALS), 'unknown pilot checkpoint state')
    require(controller['state'] == ('running' if state['status'] == 'running' else 'closed-clean'),
            'pilot controller status is inconsistent')
    require(type(controller['pid']) is int and controller['pid'] > 0
        and type(controller['invocation']) is int and controller['invocation'] > 0,
        'invalid pilot controller identity')
    require(type(controller['measured_invocation_seconds']) is float
        and math.isfinite(controller['measured_invocation_seconds']) and controller['measured_invocation_seconds'] >= 0,
        'invalid measured pilot invocation time')
    spent = controller['cumulative_charged_seconds']
    require(type(spent) is float and math.isfinite(spent) and spent >= 0, 'invalid cumulative pilot budget')
    for key, entry in state['cases'].items():
        require(type(entry) is dict and set(entry) == set(expected['cases'][key]), 'pilot case ledger schema differs')
        require(encoded(entry['case']) == encoded(expected['cases'][key]['case']), 'pilot case definition differs')
        require(entry['status'] in ('pending', 'running', 'retained', 'execution-failure-retained'), 'unknown pilot case state')
        files = case_files(auth['output'], key)
        actual = {str(path) for path in files.values() if path.exists()}
        if entry['status'] == 'pending':
            require(entry['artifacts'] == {} and not actual and entry['case_seconds'] is None and entry['error'] is None,
                    'orphan artifact or altered pending pilot case')
        elif entry['status'] == 'running':
            require(not resume, 'interrupted pilot worker retained; inspect real process and evidence')
            require(entry['artifacts'] == {str(files['spec']): file_sha(files['spec'])}, 'running pilot spec differs')
        else:
            require(set(entry['artifacts']) == actual, 'pilot artifact support differs')
            require({str(files['spec']), str(files['log'])} <= actual, 'pilot spec/log missing')
            verify_pins(entry['artifacts'])
            require(type(entry['case_seconds']) is float and math.isfinite(entry['case_seconds'])
                    and entry['case_seconds'] >= 0, 'invalid pilot case timing')
            require(encoded(read_json(files['spec'])) == encoded(spec_for(auth, entry['case'])), 'retained pilot spec differs')
            if entry['status'] == 'retained':
                wanted = {str(files[name]) for name in ('spec', 'log', 'array', 'receipt')}
                if entry['case']['role'] == 'checkpoint':
                    wanted.add(str(files['restart']))
                require(actual == wanted and entry['error'] is None, 'successful pilot evidence incomplete')
                receipt = read_json(files['receipt'])
                validate_receipt_binding(receipt, entry['case'], files)
            else:
                require(type(entry['error']) is str and bool(entry['error'].strip()),
                        'retained pilot failure lacks an error record')
    require(spent+1e-6 >= sum(e['case_seconds'] or 0. for e in state['cases'].values()), 'pilot cumulative budget reset')
    if state['status'] != 'running':
        require(all(e['status'] != 'running' for e in state['cases'].values()), 'closed pilot still has a running case')
    if state['status'] == TERMINAL:
        require(all(e['status'] == 'retained' for e in state['cases'].values()), 'terminal pilot support incomplete')
    if state['status'] == STOPPED:
        require(any(e['status'] == 'execution-failure-retained' for e in state['cases'].values()), 'failure status lacks a retained failure')
    validate_inventory(auth['output'], state)
    if resume:
        require(state['status'] == 'clean-boundary' and controller['state'] == 'closed-clean',
                'pilot requires an authoritatively clean controller boundary; no interrupted or terminal replay')
        require(all(e['status'] in ('pending', 'retained') for e in state['cases'].values()), 'failed pilot cannot auto-resume')
    return state


def open_engine(auth, row):
    name = row['engine']
    folder = PREPARATION/(name+'-build')
    require((folder/'build.json').is_file()
        and Path(auth['prep']['builds'][name]['binary_path']).is_file(), 'missing prepared pilot binary; no rebuild')
    engine = (HHDriver if name == 'gbs' else RKF78Driver)(folder)
    require(engine.provenance == auth['prep']['builds'][name], 'pilot worker build receipt differs')
    return engine


def forward(engine, row, reg, files):
    p, full, initial, command, epoch = synthetic_input(row['samples'], row['profile'])
    current = full if row['stop_index'] is None else full[:row['stop_index']+1]
    before = (encoded(p), current.tobytes(), initial.tobytes(), command.tobytes())
    checkpoint_seconds = 0.
    began = time.perf_counter()
    if row['engine'] == 'gbs':
        policy = Policy(**reg['primary_policies'][row['policy']])
        if row['role'] == 'checkpoint':
            with engine.context(p, current, reg['inputs']['rate_hz'], initial, row['bits'], policy) as ctx:
                prefix = ctx.advance(reg['restart_control_native_index'])
                saved = ctx.snapshot()
            checkpoint_start = time.perf_counter()
            write_json(files['restart'], saved)
            restored = read_json(files['restart'])
            checkpoint_seconds = time.perf_counter()-checkpoint_start
            with engine.context(p, current, reg['inputs']['rate_hz'], initial, row['bits'], policy) as ctx:
                ctx.restore(restored)
                suffix = ctx.advance(len(current)-1)
                final, stats = ctx.snapshot(), ctx.stats
            states = np.concatenate((prefix, suffix[1:]))
        else:
            with engine.context(p, current, reg['inputs']['rate_hz'], initial, row['bits'], policy) as ctx:
                states = ctx.advance(len(current)-1)
                final, stats = ctx.snapshot(), ctx.stats
        embedded = None
    else:
        limits = reg['resources']
        result = engine.integrate(p, current, reg['inputs']['rate_hz'], initial, row['bits'], row['factor'],
            maximum_rhs=limits['maximum_independent_rhs'], maximum_substeps=limits['maximum_independent_substeps'])
        states, final = result.states, {'kind': 'rkf78-exact-final', 'values': list(result.final_exact)}
        stats = {'position': len(current)-1, 'rhs': result.rhs_evaluations, 'accepted': result.accepted_substeps,
                 'rejected': 0, 'dense': 0}
        embedded = result.maximum_embedded_estimate.tolist()
    forward_seconds = time.perf_counter()-began
    require(before == (encoded(p), current.tobytes(), initial.tobytes(), command.tobytes()), 'pilot input buffer mutated')
    require(type(states) is np.ndarray and states.dtype == np.float64
        and list(states.shape) == input_signature(row)['shape'] and np.isfinite(states).all(), 'invalid pilot output array')
    require(np.array_equal(states[0].view(np.uint64), initial.view(np.uint64)), 'pilot initialization changed')
    require(np.all(states[:, 1:] >= -1e-8) and np.all(states[:, 1:] <= 1.0+1e-8), 'pilot observed gate bound exceeded')
    require(set(stats) == {'position', 'rhs', 'accepted', 'rejected', 'dense'}
        and all(type(value) is int and value >= 0 for value in stats.values())
        and stats['position'] == len(current)-1 and stats['dense'] <= len(current)-1,
        'pilot solver work accounting differs')
    if row['engine'] == 'rkf78':
        require(stats['rhs'] == (len(current)-1)*row['factor']*13
            and stats['accepted'] == (len(current)-1)*row['factor'], 'independent pilot work support differs')
    return states, command, epoch, final, stats, embedded, forward_seconds, checkpoint_seconds


def worker(spec_path):
    auth = load_authorization()  # No --worker bypass of the execution decision.
    require(all(os.environ.get(key) == value for key, value in WORKER_ENV.items()), 'pilot worker environment differs')
    spec_path = Path(spec_path)
    spec = read_json(spec_path)
    row = validate_case(spec['case'])
    files = case_files(auth['output'], label(row))
    require(spec_path == files['spec'] and encoded(spec) == encoded(spec_for(auth, row)), 'pilot child specification differs')
    require(not auth['output'].is_symlink()
        and all(not (auth['output']/name).is_symlink() for name in ('runner.lock', 'worker.lock')),
        'linked pilot output or lock refused')
    with (auth['output']/'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = validate_state(read_json(auth['output']/'checkpoint.json'), auth)
        require(state['status'] == 'running' and state['controller']['state'] == 'running'
            and state['cases'][label(row)]['status'] == 'running', 'worker is not an active approved ledger case')
        require(state['controller']['pid'] == os.getppid(), 'pilot worker parent is not the recorded live controller')
        with (auth['output']/'runner.lock').open('r') as parent_lock:
            try:
                fcntl.flock(parent_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                fcntl.flock(parent_lock, fcntl.LOCK_UN)
                raise ValueError('pilot worker controller lock is not held')
        require(not any(files[name].exists() for name in ('array', 'receipt', 'restart')), 'orphan worker artifact retained')
        verify_pins(auth['pins'])
        engine_start = time.perf_counter()
        engine = open_engine(auth, row)
        engine_seconds = time.perf_counter()-engine_start
        states, command, epoch, final, stats, embedded, seconds, checkpoint_seconds = forward(engine, row, auth['reg'], files)
        post_start = time.perf_counter()
        if row['role'] == 'landing':
            features = score = None
        else:
            features = characterize(states[auth['reg']['inputs']['warmup_samples']:, 0], command,
                                    auth['reg']['inputs']['rate_hz'], epoch, 0.)
            score = score_sweep(auth['reg']['synthetic_target'], features)
        post_seconds = time.perf_counter()-post_start
        compression_start = time.perf_counter()
        with files['array'].open('xb') as file:
            np.savez_compressed(file, states=states)
        compression_seconds = time.perf_counter()-compression_start
        require(files['array'].stat().st_size <= states.nbytes+65536, 'pilot compressed array unexpectedly oversized')
        verify_pins(auth['pins'])
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        receipt = {'schema_version': 1, 'case_key': label(row), 'case': row, 'input_signature': input_signature(row),
            'shape': list(states.shape), 'arrays_sha256': file_sha(files['array']), 'uncompressed_bytes': states.nbytes,
            'compressed_bytes': files['array'].stat().st_size, 'features': features, 'score': score,
            'final_state': final, 'solver_statistics': stats, 'maximum_embedded_estimate': embedded,
            'driver_open_seconds': engine_seconds, 'forward_call_seconds': seconds,
            'checkpoint_serialization_seconds': checkpoint_seconds, 'postprocess_seconds': post_seconds,
            'compression_seconds': compression_seconds, 'peak_process_RSS_bytes': int(peak if sys.platform == 'darwin' else peak*1024),
            'scientific_goal_complete': False, 'biological_access': False, 'fitting_restart': False}
        write_json(files['receipt'], receipt)
        require(output_bytes(auth['output']) <= auth['reg']['resources']['maximum_output_bytes'], 'pilot output budget exceeded')


def budget_allows(prior, elapsed, resources):
    require(all(type(x) in (int, float) and not isinstance(x, bool) and math.isfinite(x) and x >= 0
                for x in (prior, elapsed)), 'invalid pilot elapsed budget')
    reserve = resources['maximum_case_seconds']+resources['cleanup_reserve_seconds']
    return (elapsed+reserve <= resources['maximum_invocation_seconds']
            and prior+elapsed+reserve <= resources['maximum_total_controller_seconds'])


def run_child(spec, logfile, timeout):
    environment = {**os.environ, **WORKER_ENV}
    subprocess.run([sys.executable, str(SCRIPT), '--worker', str(spec)], check=True,
        stdout=logfile, stderr=subprocess.STDOUT, timeout=timeout, env=environment)


def run(stop_after_cases=None):
    started = time.monotonic()
    require(stop_after_cases is None or (type(stop_after_cases) is int and stop_after_cases > 0), 'invalid clean-stop count')
    auth = load_authorization()
    root, limits = auth['output'], auth['reg']['resources']
    require(not root.is_symlink(), 'linked pilot output refused')
    root.mkdir(parents=True, exist_ok=True)
    require(all(not (root/name).is_symlink() for name in ('runner.lock', 'worker.lock')), 'linked pilot lock refused')
    with (root/'runner.lock').open('a') as controller_lock, (root/'worker.lock').open('a') as worker_lock:
        fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(worker_lock, fcntl.LOCK_UN)
        checkpoint = root/'checkpoint.json'
        if checkpoint.exists():
            state = read_json(checkpoint)
            if state.get('status') in TERMINALS:
                validate_state(state, auth)
                return state
            validate_state(state, auth, resume=True)
            prior = state['controller']['cumulative_charged_seconds']
            invocation = state['controller']['invocation']+1
        else:
            require({p.name for p in root.iterdir()} <= {'runner.lock', 'worker.lock'}, 'orphan pilot output retained')
            state, prior, invocation = new_state(auth), 0., 1
        state['status'] = 'running'
        state['controller'] = {'state': 'running', 'pid': os.getpid(), 'invocation': invocation,
            'cumulative_charged_seconds': float(prior), 'measured_invocation_seconds': 0.}
        write_json(checkpoint, state, atomic=True)
        completed = 0
        def save(status=None):
            elapsed = time.monotonic()-started
            require(elapsed >= 0 and math.isfinite(elapsed), 'pilot clock moved backward')
            state['controller']['measured_invocation_seconds'] = elapsed
            state['controller']['cumulative_charged_seconds'] = float(prior+elapsed+(limits['cleanup_reserve_seconds'] if status else 0))
            if status:
                state['status'] = status
                state['controller']['state'] = 'closed-clean'
                if state['controller']['cumulative_charged_seconds'] > limits['maximum_total_controller_seconds']:
                    state['status'] = EXHAUSTED
            write_json(checkpoint, state, atomic=True)
        for row in case_table():
            key = label(row)
            if state['cases'][key]['status'] == 'retained':
                continue
            if not budget_allows(prior, time.monotonic()-started, limits):
                save(EXHAUSTED)
                return state
            files = case_files(root, key)
            require(not any(path.exists() or path.is_symlink() for path in files.values()), 'orphan pilot case artifact retained')
            spec = spec_for(auth, row)
            reserve_bytes = input_signature(row)['uncompressed_array_bytes']+len(encoded(spec).encode())+2*1024*1024
            if output_bytes(root)+reserve_bytes > limits['maximum_output_bytes']:
                save(EXHAUSTED)
                return state
            case_started = time.perf_counter()
            write_json(files['spec'], spec)
            entry = state['cases'][key]
            entry['status'] = 'running'
            entry['artifacts'] = {str(files['spec']): file_sha(files['spec'])}
            save()
            failure = None
            try:
                with files['log'].open('xb') as logfile:
                    run_child(files['spec'], logfile, limits['maximum_case_seconds'])
                require(files['receipt'].is_file() and files['array'].is_file(), 'worker success lacks required artifacts')
                receipt = read_json(files['receipt'])
                validate_receipt_binding(receipt, row, files)
            except (subprocess.SubprocessError, ValueError, OSError) as exc:
                failure = type(exc).__name__+': '+str(exc)
            # A timed-out observed child is not proof that every worker is gone.
            fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(worker_lock, fcntl.LOCK_UN)
            verify_pins(auth['pins'])
            if output_bytes(root) > limits['maximum_output_bytes']:
                failure = 'pilot output budget exceeded; artifacts retained'
            entry.update(status='execution-failure-retained' if failure else 'retained',
                case_seconds=float(time.perf_counter()-case_started), error=failure,
                artifacts={str(path): file_sha(path) for path in files.values() if path.is_file()})
            save()
            if failure:
                save(STOPPED)
                return state
            completed += 1
            if stop_after_cases is not None and completed >= stop_after_cases:
                save(TERMINAL if all(e['status'] == 'retained' for e in state['cases'].values()) else 'clean-boundary')
                return state
        save(TERMINAL)
        return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--plan', action='store_true', help='Print fixed metadata only, no builds or execution')
    mode.add_argument('--prepare-builds', action='store_true', help='Build/pin after auditor implementation; no pilot trajectory')
    mode.add_argument('--worker', type=Path)
    parser.add_argument('--stop-after-cases', type=int, help='Clean case-boundary pause, never reduces required support or resets budget')
    args = parser.parse_args()
    if args.stop_after_cases is not None and (args.plan or args.prepare_builds or args.worker):
        parser.error('clean-stop count is only valid for the controller')
    if args.plan:
        print(encoded({'cases': case_table(), 'inputs': input_manifest(), 'comparisons': comparison_table()}))
    elif args.prepare_builds:
        print(encoded({'status': prepare_builds()['status']}))
    elif args.worker:
        worker(args.worker)
    else:
        result = run(args.stop_after_cases)
        print(encoded({'status': result['status'], 'scientific_goal_complete': False}))
