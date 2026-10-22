"""Bounded independent pilot1515 evidence audit; never runs a neuron or builds.

Both entrypoints require the distinct approved source/build/runtime seal.
Only a terminal worker followed by final provenance/resource rechecks can write
an exclusive final assessment. Interrupted audit evidence is never overwritten.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
import zipfile
from contextlib import ExitStack
from importlib.metadata import version
from pathlib import Path

import numpy as np
import scipy
import yaml
from patchseq_gbs_pilot_array_audit import ArrayAudit
from verify_patchseq_gbs_pilot_protocol import (
    canonical,
    case_name,
    expected_rows,
    sha,
    validate_registration,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
REG = ROOT/'docs/validation-results/post2008-patchseq-gbs-synthetic-pilot-registration-1515.yaml'
SEAL = ROOT/'docs/validation-results/post2008-patchseq-gbs-synthetic-pilot-execution-seal-1516.yaml'
PREP = ROOT/'results/patchseq-gbs-pilot-build-preparation-1520'
MAX_JSON = 64*1024*1024
TERMINALS = ('pilot-retained-awaiting-independent-audit', 'pilot-stopped-after-execution-failure',
             'pilot-budget-exhausted-incomplete')
AUDIT_NAMES = {'audit.lock', 'audit-control.json', 'audit.log', 'audit-analysis.json',
               'audit-outcome.json', 'independent-assessment.json'}
ENV = {'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
    'VECLIB_MAXIMUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1', 'HDF5_PLUGIN_PRELOAD': '::',
    'MPLCONFIGDIR': '/private/tmp/mpl', 'PYTHONDONTWRITEBYTECODE': '1'}
REQUIRED_PROBES = {
    'gbs': {'mpfr': '4.2.2', 'header': '4.2.2', 'gmp': '6.3.0', 'tls': 1,
        'emin': -1073741823, 'emax': 1073741823, 'precision160': 160, 'precision320': 320,
        'cancellation160': 1, 'cancellation320': 1},
    'rkf78': {'mpfr': '4.2.2', 'header': '4.2.2', 'gmp': '6.3.0', 'range_and_tls': 1,
        'precision160': 160, 'precision320': 320},
    'rkf78_table_sha256': '1618bce9dc73fb8b4ea5ba33ac3af42ecbec2c9c8706e221ea0b4c0e4cf51313'}


def require(condition, message):
    if not condition:
        raise ValueError('independent pilot evidence: '+message)


def digest(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'missing or linked regular file: '+str(path))
    value = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def unique(items):
    result = {}
    for key, value in items:
        require(key not in result, 'duplicate metadata key')
        result[key] = value
    return result


def read_json(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= MAX_JSON, 'invalid bounded JSON file')
    data = json.loads(path.read_text(), object_pairs_hook=unique)
    canonical(data)
    return data


def read_yaml(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= MAX_JSON, 'invalid bounded YAML file')
    class Loader(yaml.SafeLoader):
        pass
    def mapping(loader, node):
        return unique(loader.construct_pairs(node))
    Loader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    result = yaml.load(path.read_text(), Loader=Loader)
    canonical(result)
    return result


def write_exclusive(path, data):
    path = Path(path)
    raw = (canonical(data)+'\n').encode()
    require(len(raw) <= MAX_JSON and not path.exists() and not path.is_symlink(), 'existing or oversized audit output retained')
    with path.open('xb') as file:
        file.write(raw)
        file.flush()
        os.fsync(file.fileno())


def merge(*maps):
    result = {}
    for mapping in maps:
        require(type(mapping) is dict, 'pin map is not a dictionary')
        for path, value in mapping.items():
            require(path not in result or result[path] == value, 'contradictory source pins')
            result[path] = value
    return result


def verify_pins(pins):
    require(type(pins) is dict and bool(pins), 'empty source/build/runtime closure')
    for name, expected in pins.items():
        require(type(name) is str and type(expected) is str and re.fullmatch('[0-9a-f]{64}', expected) is not None,
                'malformed source pin')
        path = Path(name)
        require(path.is_absolute() and str(path.resolve()) == name
            and path.suffix.lower() not in ('.nwb', '.h5', '.hdf5'), 'unsupported source path or biological-body access')
        if path.is_relative_to(ROOT/'results'):
            require(path.is_relative_to(ROOT/'results/patchseq-mpfr-toolchain-1504') or path.is_relative_to(PREP),
                    'unrelated raw results cannot enter the audit source closure')
        require(digest(path) == expected, 'source/build/runtime file changed: '+name)


def repository_sources():
    # Independent explicit closure, not a call into the controller's pin helper.
    names = ['docs/validation-results/post2008-patchseq-gbs-synthetic-pilot-registration-1515.yaml',
        'scripts/run_patchseq_gbs_pilot.py', 'scripts/verify_patchseq_gbs_pilot.py',
        'scripts/patchseq_gbs_pilot_array_audit.py', 'scripts/patchseq_gbs_pilot_protocol.py',
        'scripts/verify_patchseq_gbs_pilot_protocol.py', 'scripts/patchseq_gbs_hh_driver.py',
        'scripts/patchseq_gbs_hh.cpp', 'scripts/patchseq_gbs_guarded_core.hpp', 'scripts/patchseq_gbs_core.hpp',
        'scripts/patchseq_gbs_mpfr_number.hpp', 'scripts/patchseq_gbs_odeint_bridge.hpp',
        'scripts/third_party/patchseq_bounded_gbs_dense.hpp', 'scripts/third_party/BOOST_LICENSE_1_0.txt',
        'scripts/patchseq_gbs_driver.py', 'scripts/patchseq_mpfr_driver.py', 'scripts/patchseq_mpfr_kernel.cpp',
        'scripts/patchseq_rkf78_reference.cpp', 'scripts/patchseq_rkf78_reference.py',
        'scripts/patchseq_response_features.py', 'scripts/patchseq_fit_objective.py',
        'scripts/verify_patchseq_fitting_targets.py', 'scripts/verify_patchseq_whole_file_integrity.py']
    return {str(ROOT/name): digest(ROOT/name) for name in names}


def runtime():
    directories = [Path(module.__file__).resolve().parent for module in (np, scipy, yaml)]
    for module in (np, scipy):
        extra = Path(module.__file__).resolve().parent.parent/(module.__name__+'.libs')
        if extra.is_dir():
            directories.append(extra)
    names = {str(Path(sys.executable).resolve())}
    for directory in directories:
        for path in directory.rglob('*'):
            if (path.is_file() and '__pycache__' not in path.parts
                and (path.suffix in ('.py', '.dylib', '.pyd', '.dll', '.h', '.a') or '.so' in path.name)):
                names.add(str(path.resolve()))
    return {'python': sys.version, 'platform': platform.platform(),
        'versions': {name: version(name) for name in ('numpy', 'scipy', 'PyYAML')},
        'required_worker_environment': dict(ENV), 'file_hashes': {name: digest(name) for name in sorted(names)}}


def checked_preparation():
    """Read-only build/probe verification, explicitly not an execution decision."""
    reg = read_yaml(REG)
    validate_registration(reg)
    require(not PREP.is_symlink(), 'linked preparation directory')
    prep = read_json(PREP/'preparation.json')
    require(type(prep) is dict and set(prep) == {'status', 'source_hashes', 'runtime', 'pins', 'builds',
        'arithmetic_probes', 'preparation_seconds', 'biological_access', 'pilot_executed', 'fitting_restart'}, 'preparation schema differs')
    require(canonical(prep['arithmetic_probes']) == canonical(REQUIRED_PROBES), 'actual arithmetic probe evidence differs')
    require(prep['status'] == 'built-and-pinned-no-pilot-trajectories'
        and canonical(prep['source_hashes']) == canonical(repository_sources())
        and canonical(prep['runtime']) == canonical(runtime()), 'prepared source set or actual runtime differs')
    require(all(prep[name] is False for name in ('pilot_executed', 'biological_access', 'fitting_restart')),
            'preparation claims an unauthorized operation')
    nonnegative_float(prep['preparation_seconds'])
    require(type(prep['builds']) is dict and set(prep['builds']) == {'gbs', 'rkf78'}, 'forward build support differs')
    pins = merge(prep['source_hashes'], prep['runtime']['file_hashes'])
    for engine, filename in (('gbs', 'hh.so'), ('rkf78', 'reference.so')):
        folder, build = PREP/(engine+'-build'), prep['builds'][engine]
        require(not folder.is_symlink() and build['binary_path'] == str((folder/filename).resolve())
            and canonical(build) == canonical(read_json(folder/'build.json')), 'exact build receipt/path differs')
        pins = merge(pins, build['source_hashes'],
            {build['binary_path']: build['binary_sha256'], str(folder/'build.json'): digest(folder/'build.json')})
    require(canonical(pins) == canonical(prep['pins']), 'preparation source closure is incomplete or expanded')
    verify_pins(pins)
    return prep, pins


def load_contract():
    reg = read_yaml(REG)
    validate_registration(reg)
    seal = read_yaml(SEAL)  # Missing approval refuses before output access or prepared-build checking.
    require(type(seal) is dict and set(seal) == {'schema_version', 'status', 'human_execution_approved',
        'human_approval_record', 'registration_sha256', 'preparation_path', 'preparation_sha256',
        'resources', 'allowed_operations', 'source_hashes'}, 'execution decision schema differs')
    require(type(seal['schema_version']) is int and seal['schema_version'] == 1
        and seal['status'] == 'sealed-before-short-synthetic-pilot' and seal['human_execution_approved'] is True
        and type(seal['human_approval_record']) is str and bool(seal['human_approval_record'].strip()),
        'distinct human execution approval is missing')
    require(seal['registration_sha256'] == digest(REG)
        and canonical(seal['resources']) == canonical(reg['resources'])
        and canonical(seal['allowed_operations']) == canonical(['short-synthetic-pilot', 'independent-pilot-audit']),
        'execution decision scope differs')
    require(not PREP.is_symlink() and seal['preparation_path'] == str((PREP/'preparation.json').relative_to(ROOT))
        and seal['preparation_sha256'] == digest(PREP/'preparation.json'), 'prepared provenance binding differs')
    prep, pins = checked_preparation()
    require(canonical(seal['source_hashes']) == canonical(prep['source_hashes']), 'approved source support differs')
    pins = merge(pins, {str(SEAL): digest(SEAL), str(PREP/'preparation.json'): digest(PREP/'preparation.json')})
    verify_pins(pins)
    return {'reg': reg, 'prep': prep, 'pins': pins, 'registration_sha256': digest(REG),
            'seal_sha256': digest(SEAL), 'preparation_sha256': digest(PREP/'preparation.json'), 'output': ROOT/reg['output']}


def nonnegative_float(value):
    require(type(value) is float and math.isfinite(value) and value >= 0., 'invalid measured time')
    return value


def case_paths(root, key):
    return {name: root/(key+suffix) for name, suffix in {
        'spec': '.spec.json', 'log': '.log', 'array': '.npz', 'receipt': '.receipt.json', 'restart': '.restart.json'}.items()}


def inventory(root, allowed):
    require(root.is_dir() and not root.is_symlink(), 'missing or linked output directory')
    total = 0
    for path in root.iterdir():
        require(path.name in allowed and path.is_file() and not path.is_symlink(), 'unlisted nested or linked output')
        total += path.stat().st_size
    return total


def allowed_outputs():
    return ({'runner.lock', 'worker.lock', 'checkpoint.json'} | AUDIT_NAMES |
            {path.name for row in expected_rows() for path in case_paths(Path('.'), case_name(row)).values()})


def write_audit(auth, name, data):
    require(name in AUDIT_NAMES-{'audit.lock', 'audit.log'}, 'unregistered audit output name')
    cost = len(canonical(data).encode())+1
    require(inventory(auth['output'], allowed_outputs())+cost <= auth['reg']['resources']['maximum_output_bytes'],
            'audit write would exceed the retained output budget')
    write_exclusive(auth['output']/name, data)


def check_checkpoint(auth):
    root, reg = auth['output'], auth['reg']
    state = read_json(root/'checkpoint.json')
    expected_keys = {'schema_version', 'registration_sha256', 'seal_sha256', 'preparation_sha256', 'pins', 'table',
        'input_manifest_sha256', 'comparison_table_sha256', 'builds', 'runtime', 'status', 'controller', 'cases',
        'biological_access', 'fitting_restart', 'long_reference_qualified', 'scientific_goal_complete'}
    require(type(state) is dict and set(state) == expected_keys and type(state['schema_version']) is int
        and state['schema_version'] == 1 and state['status'] in TERMINALS, 'checkpoint is not an admissible terminal ledger')
    for name in ('registration_sha256', 'seal_sha256', 'preparation_sha256', 'pins'):
        require(canonical(state[name]) == canonical(auth[name]), 'checkpoint authorization identity differs')
    require(canonical(state['builds']) == canonical(auth['prep']['builds'])
        and canonical(state['runtime']) == canonical(auth['prep']['runtime']), 'checkpoint build/runtime differs')
    require(all(state[name] is False for name in ('biological_access', 'fitting_restart',
        'long_reference_qualified', 'scientific_goal_complete')), 'checkpoint makes an unauthorized claim')
    model = ArrayAudit(reg, auth['prep']['builds'])
    require(canonical(state['table']) == canonical(expected_rows()) and set(state['cases']) == set(model.rows)
        and state['input_manifest_sha256'] == sha(model.manifest)
        and state['comparison_table_sha256'] == sha(model.pairs), 'incomplete or altered fixed ledger support')
    controller = state['controller']
    require(type(controller) is dict and set(controller) == {'state', 'pid', 'invocation',
        'cumulative_charged_seconds', 'measured_invocation_seconds'} and controller['state'] == 'closed-clean'
        and type(controller['pid']) is int and controller['pid'] > 0
        and type(controller['invocation']) is int and controller['invocation'] > 0, 'controller is not authoritatively closed')
    charged = nonnegative_float(controller['cumulative_charged_seconds'])
    measured = nonnegative_float(controller['measured_invocation_seconds'])
    require(measured <= reg['resources']['maximum_invocation_seconds']
        and measured <= charged, 'invocation timing support differs')
    artifacts = {str(root/'checkpoint.json'): digest(root/'checkpoint.json')}
    allowed = {'runner.lock', 'worker.lock', 'checkpoint.json'} | AUDIT_NAMES
    case_total = 0.
    for key, row in model.rows.items():
        entry, files = state['cases'][key], case_paths(root, key)
        allowed.update(path.name for path in files.values())
        require(type(entry) is dict and set(entry) == {'case', 'status', 'artifacts', 'case_seconds', 'error'}
            and canonical(entry['case']) == canonical(row), 'case ledger schema or typed identity differs')
        actual = {str(path): digest(path) for path in files.values() if path.exists() or path.is_symlink()}
        require(type(entry['artifacts']) is dict and canonical(entry['artifacts']) == canonical(actual),
                'case artifact support or hashes differ')
        if entry['status'] == 'pending':
            require(not actual and entry['case_seconds'] is None and entry['error'] is None, 'pending case has orphan evidence')
        else:
            require(entry['status'] in ('retained', 'execution-failure-retained')
                and {str(files['spec']), str(files['log'])} <= set(actual), 'interrupted or unsupported case ledger state')
            seconds = nonnegative_float(entry['case_seconds'])
            case_total += seconds
            spec = {'schema_version': 1, 'case_key': key, 'case': row, 'input_signature': model.manifest[key],
                'registration_sha256': auth['registration_sha256'], 'seal_sha256': auth['seal_sha256'],
                'preparation_sha256': auth['preparation_sha256'], 'pins': auth['pins'],
                'build_provenance': auth['prep']['builds'][row['engine']],
                'files': {name: str(path) for name, path in files.items()}}
            require(canonical(read_json(files['spec'])) == canonical(spec), 'case spec binding differs')
            if entry['status'] == 'retained':
                needed = {str(files[name]) for name in ('spec', 'log', 'array', 'receipt')}
                if row['role'] == 'checkpoint':
                    needed.add(str(files['restart']))
                require(set(actual) == needed and entry['error'] is None, 'retained case support is incomplete')
            else:
                require(type(entry['error']) is str and bool(entry['error'].strip()), 'failure evidence lacks an error')
        artifacts.update(actual)
    require(charged+1e-6 >= case_total, 'cumulative controller budget was reset')
    require(state['status'] != TERMINALS[0] or all(r['status'] == 'retained' for r in state['cases'].values()),
            'complete status hides missing or failed cases')
    require(state['status'] != TERMINALS[1] or any(r['status'] == 'execution-failure-retained' for r in state['cases'].values()),
            'stopped status lacks a retained failure')
    size = inventory(root, allowed)
    require(size <= reg['resources']['maximum_output_bytes'], 'retained output exceeds storage bound')
    return state, model, artifacts, size


def load_states(path, shape):
    """Validate bounded ZIP member and NPY header before NumPy allocation."""
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'missing or linked array archive')
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        require(len(members) == 1 and members[0].filename == 'states.npy'
            and not members[0].flag_bits & 1 and members[0].compress_type == zipfile.ZIP_DEFLATED,
            'array archive member support or encoding differs')
        member = members[0]
        require(np.prod(shape, dtype=np.int64)*8 <= member.file_size <= np.prod(shape, dtype=np.int64)*8+65536,
                'array archive would exceed declared native storage')
        with archive.open(member) as stream:
            require(np.lib.format.read_magic(stream) == (1, 0), 'unregistered NPY header version')
            actual_shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream, max_header_size=16384)
            require(list(actual_shape) == shape and fortran is False and dtype == np.dtype(np.float64),
                    'array header shape layout or dtype differs')
            require(stream.tell()+np.prod(shape, dtype=np.int64)*8 == member.file_size, 'array member has truncated or extra data')
    with np.load(path, allow_pickle=False, max_header_size=16384) as data:
        require(data.files == ['states'], 'array keys differ')
        return data['states']


def recheck(auth, artifacts):
    verify_pins(auth['pins'])
    require(all(digest(path) == value for path, value in artifacts.items()), 'raw evidence changed during assessment')


def analyze(auth):
    state, model, artifacts, initial_size = check_checkpoint(auth)
    reports, timings = {}, []
    uncompressed = 0
    for key, row in model.rows.items():
        entry, paths = state['cases'][key], case_paths(auth['output'], key)
        if entry['status'] != 'retained':
            reports[key] = {'case_key': key, 'passed': False, 'state': 'unavailable-evidence-preserved', 'ledger': entry}
            continue
        states = load_states(paths['array'], model.manifest[key]['shape'])
        receipt = read_json(paths['receipt'])
        require(receipt['arrays_sha256'] == digest(paths['array'])
            and type(receipt['compressed_bytes']) is int and receipt['compressed_bytes'] == paths['array'].stat().st_size,
            'array receipt hash/size differs')
        restart = read_json(paths['restart']) if row['role'] == 'checkpoint' else None
        reports[key] = model.inspect(key, states, receipt, restart=restart)
        phases = math.fsum(receipt[k] for k in ('driver_open_seconds', 'forward_call_seconds',
                                              'postprocess_seconds', 'compression_seconds'))
        require(phases <= entry['case_seconds']+1e-6
            and receipt['checkpoint_serialization_seconds'] <= receipt['forward_call_seconds']+1e-6,
            'worker stage timing exceeds retained end-to-end support')
        timings.append({'case_key': key, 'total_case_seconds': entry['case_seconds'], 'phase_seconds': phases,
                        'within_case_plus_cleanup_bound': entry['case_seconds'] <= 660.})
        uncompressed += states.nbytes
    comparisons = []
    for pair in model.pairs:
        left, right = pair['left'], pair['right']
        if any(state['cases'][key]['status'] != 'retained' for key in (left, right)):
            comparisons.append({**pair, 'passed': False, 'state': 'unavailable-evidence-preserved'})
            continue
        arrays = [load_states(case_paths(auth['output'], key)['array'], model.manifest[key]['shape']) for key in (left, right)]
        comparisons.append(model.compare(pair, *arrays, reports[left], reports[right]))
    summary = model.summarize(reports, comparisons)
    complete = all(item['status'] == 'retained' for item in state['cases'].values())
    charged = state['controller']['cumulative_charged_seconds']
    resources = {'controller_charged_seconds': charged, 'output_bytes_before_audit': initial_size,
        'native_array_uncompressed_bytes': uncompressed, 'complete_native_array_support': complete,
        'controller_budget_passed': charged <= auth['reg']['resources']['maximum_total_controller_seconds'],
        'case_timing_bounds_passed': all(item['within_case_plus_cleanup_bound'] for item in timings), 'case_timings': timings}
    if complete:
        require(uncompressed == auth['reg']['resources']['total_native_array_uncompressed_bytes'], 'full native storage support differs')
    recheck(auth, artifacts)
    return {'schema_version': 1, 'state': 'independent-analysis-awaiting-terminal-parent-admission',
        'checkpoint_sha256': artifacts[str(auth['output']/'checkpoint.json')], 'source_hashes': auth['pins'],
        'raw_artifact_hashes': artifacts, 'cases': reports, 'comparisons': comparisons,
        'array_summary': summary, 'resources': resources, 'biological_access': False,
        'fitting_restart': False, 'long_reference_qualified': False, 'scientific_goal_complete': False}


def validate_analysis(auth, analysis, artifacts):
    """Do not accept a child summary with missing identities or waived gates."""
    require(type(analysis) is dict and set(analysis) == {'schema_version', 'state', 'checkpoint_sha256',
        'source_hashes', 'raw_artifact_hashes', 'cases', 'comparisons', 'array_summary', 'resources',
        'biological_access', 'fitting_restart', 'long_reference_qualified', 'scientific_goal_complete'},
        'analysis schema differs')
    require(type(analysis['schema_version']) is int and analysis['schema_version'] == 1
        and analysis['state'] == 'independent-analysis-awaiting-terminal-parent-admission'
        and analysis['checkpoint_sha256'] == artifacts[str(auth['output']/'checkpoint.json')]
        and canonical(analysis['source_hashes']) == canonical(auth['pins'])
        and canonical(analysis['raw_artifact_hashes']) == canonical(artifacts)
        and all(analysis[k] is False for k in ('biological_access', 'fitting_restart',
            'long_reference_qualified', 'scientific_goal_complete')), 'analysis provenance or authority differs')
    state, model, current_artifacts, current_size = check_checkpoint(auth)
    require(current_artifacts == artifacts, 'raw support changed after child completion')
    require(canonical(model.summarize(analysis['cases'], analysis['comparisons'])) == canonical(analysis['array_summary']),
            'analysis omitted identities changed result support or waived mandatory comparisons')
    timing, size = [], 0
    for key in model.rows:
        entry, report = state['cases'][key], analysis['cases'][key]
        if entry['status'] != 'retained':
            require(canonical(report) == canonical({'case_key': key, 'passed': False,
                'state': 'unavailable-evidence-preserved', 'ledger': entry}), 'failed/pending identity was not preserved')
            continue
        require(type(report.get('checks')) is dict
            and set(report['checks']) == {'features_reconstructed', 'six_term_score_reconstructed', 'passive_analytic_voltage_agreement'}
            and all(v is None or type(v) is bool for v in report['checks'].values())
            and report['passed'] is all(v for v in report['checks'].values() if v is not None),
            'case analysis result disagrees with its complete check support')
        paths = case_paths(auth['output'], key)
        receipt = read_json(paths['receipt'])
        phases = math.fsum(receipt[k] for k in ('driver_open_seconds', 'forward_call_seconds',
                                              'postprocess_seconds', 'compression_seconds'))
        timing.append({'case_key': key, 'total_case_seconds': entry['case_seconds'], 'phase_seconds': phases,
                       'within_case_plus_cleanup_bound': entry['case_seconds'] <= 660.})
        size += model.manifest[key]['uncompressed_array_bytes']
    resources = analysis['resources']
    require(type(resources) is dict and set(resources) == {'controller_charged_seconds', 'output_bytes_before_audit',
        'native_array_uncompressed_bytes', 'complete_native_array_support', 'controller_budget_passed',
        'case_timing_bounds_passed', 'case_timings'}, 'analysis resource support differs')
    charged = state['controller']['cumulative_charged_seconds']
    expected = {'controller_charged_seconds': charged, 'native_array_uncompressed_bytes': size,
        'complete_native_array_support': all(e['status'] == 'retained' for e in state['cases'].values()),
        'controller_budget_passed': charged <= auth['reg']['resources']['maximum_total_controller_seconds'],
        'case_timing_bounds_passed': all(e['within_case_plus_cleanup_bound'] for e in timing), 'case_timings': timing}
    require(canonical({k: resources[k] for k in expected}) == canonical(expected)
        and type(resources['output_bytes_before_audit']) is int
        and sum(Path(path).stat().st_size for path in artifacts) <= resources['output_bytes_before_audit'] <= current_size,
        'analysis resource totals differ from retained complete evidence')


def lock_held(path):
    require(path.is_file() and not path.is_symlink(), 'missing or linked live-parent lock')
    with path.open('r') as file:
        try:
            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        fcntl.flock(file, fcntl.LOCK_UN)
    raise ValueError('independent audit requires its live parent holding all locks')


def audit_worker(control_path):
    auth = load_contract()
    root = auth['output']
    require(all(os.environ.get(key) == value for key, value in ENV.items()), 'audit worker environment differs')
    require(Path(control_path) == root/'audit-control.json', 'worker control path differs')
    control = read_json(control_path)
    require(type(control) is dict and set(control) == {'schema_version', 'launcher_pid', 'checkpoint_sha256',
        'authorization_sha256', 'maximum_audit_seconds', 'prelaunch_seconds', 'worker_timeout_seconds'}
        and type(control['schema_version']) is int and control['schema_version'] == 1
        and type(control['launcher_pid']) is int and control['launcher_pid'] == os.getppid()
        and control['authorization_sha256'] == sha(auth['pins'])
        and control['checkpoint_sha256'] == digest(root/'checkpoint.json')
        and type(control['maximum_audit_seconds']) is int
        and control['maximum_audit_seconds'] == auth['reg']['resources']['maximum_independent_audit_seconds'],
        'worker is not bound to its live approved parent and exact checkpoint')
    prelaunch, timeout = nonnegative_float(control['prelaunch_seconds']), nonnegative_float(control['worker_timeout_seconds'])
    require(timeout > 0. and prelaunch+timeout+60. <= control['maximum_audit_seconds'], 'worker runtime bound differs')
    for name in ('runner.lock', 'worker.lock', 'audit.lock'):
        lock_held(root/name)
    require(not any((root/name).exists() or (root/name).is_symlink() for name in
        ('audit-analysis.json', 'audit-outcome.json', 'independent-assessment.json')), 'prior audit evidence retained')
    state, _model, _artifacts, _size = check_checkpoint(auth)
    require(state['controller']['cumulative_charged_seconds']+prelaunch+timeout+60.
            <= auth['reg']['resources']['maximum_controller_plus_audit_seconds'], 'combined worker audit budget differs')
    report = analyze(auth)
    write_audit(auth, 'audit-analysis.json', report)


def run_audit():
    began = time.monotonic()
    auth = load_contract()
    root, limits = auth['output'], auth['reg']['resources']
    require(root.is_dir() and not root.is_symlink(), 'missing or linked pilot output')
    with ExitStack() as stack:
        for name in ('runner.lock', 'worker.lock', 'audit.lock'):
            path = root/name
            require(not path.is_symlink(), 'linked audit/controller lock')
            file = stack.enter_context(path.open('a'))
            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not any((root/name).exists() or (root/name).is_symlink() for name in AUDIT_NAMES-{'audit.lock'}),
                'existing or interrupted audit retained; no automatic replay')
        state, _model, artifacts, size = check_checkpoint(auth)
        require(size+2*1024*1024 <= limits['maximum_output_bytes'], 'insufficient audit storage headroom')
        elapsed = float(time.monotonic()-began)
        available = min(limits['maximum_independent_audit_seconds'],
                        limits['maximum_controller_plus_audit_seconds']-state['controller']['cumulative_charged_seconds'])
        timeout = math.nextafter(float(available-elapsed-60.), 0.)
        require(timeout > 0., 'audit admission exhausted the fixed audit budget')
        control = {'schema_version': 1, 'launcher_pid': os.getpid(),
            'checkpoint_sha256': artifacts[str(root/'checkpoint.json')], 'authorization_sha256': sha(auth['pins']),
            'maximum_audit_seconds': limits['maximum_independent_audit_seconds'],
            'prelaunch_seconds': elapsed, 'worker_timeout_seconds': timeout}
        write_audit(auth, 'audit-control.json', control)
        control_hash = digest(root/'audit-control.json')
        failure = None
        try:
            with (root/'audit.log').open('xb') as log:
                subprocess.run([sys.executable, str(SCRIPT), '--audit-worker', str(root/'audit-control.json')],
                    check=True, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, **ENV}, timeout=timeout)
            analysis_hash = digest(root/'audit-analysis.json')
            analysis = read_json(root/'audit-analysis.json')
            require(digest(root/'audit-analysis.json') == analysis_hash, 'analysis changed while being read')
            require(digest(root/'audit-control.json') == control_hash, 'audit control changed during child execution')
            validate_analysis(auth, analysis, artifacts)
            recheck(auth, artifacts)
            check_checkpoint(auth)
            charged = float(time.monotonic()-began+60.)
            require(charged <= limits['maximum_independent_audit_seconds']
                and analysis['resources']['controller_charged_seconds']+charged <= limits['maximum_controller_plus_audit_seconds'],
                'audit or combined controller-plus-audit time exceeded')
            outcome = {'status': 'worker-terminal-success-awaiting-exclusive-assessment',
                'audit_charged_seconds_including_cleanup_reserve': charged,
                'checkpoint_sha256': control['checkpoint_sha256'], 'control_sha256': control_hash,
                'log_sha256': digest(root/'audit.log'), 'analysis_sha256': analysis_hash}
            write_audit(auth, 'audit-outcome.json', outcome)
            final = {**analysis, 'state': 'terminal-independent-pilot-evidence-assessment',
                'audit_outcome_sha256': digest(root/'audit-outcome.json'), 'terminal_audit_outcome': outcome,
                'short_pilot_admissibility_passed': (analysis['array_summary']['short_pilot_array_gates_passed']
                    and analysis['resources']['controller_budget_passed'] and analysis['resources']['case_timing_bounds_passed'])}
            raw = canonical(final)
            # A final complete recheck immediately precedes exclusive publication.
            recheck(auth, artifacts)
            require(inventory(root, allowed_outputs())+len(raw.encode())+1
                <= limits['maximum_output_bytes'], 'final assessment would exceed output budget')
            for name, expected in (('audit-analysis.json', analysis_hash), ('audit-control.json', outcome['control_sha256']),
                                   ('audit.log', outcome['log_sha256']), ('audit-outcome.json', final['audit_outcome_sha256'])):
                require(digest(root/name) == expected, 'audit evidence changed immediately before publication')
            require(time.monotonic()-began+60. <= limits['maximum_independent_audit_seconds'], 'final recheck exhausted audit budget')
            write_audit(auth, 'independent-assessment.json', final)
            return final
        except (ValueError, OSError, KeyError, TypeError, ArithmeticError,
                zipfile.BadZipFile, EOFError, subprocess.SubprocessError) as exc:
            failure = type(exc).__name__+': '+str(exc)
        if not (root/'audit-outcome.json').exists():
            write_audit(auth, 'audit-outcome.json', {'status': 'audit-failure-retained-no-qualification-no-replay',
                'error': failure, 'elapsed_seconds': float(time.monotonic()-began),
                'checkpoint_sha256': control['checkpoint_sha256'], 'control_sha256': digest(root/'audit-control.json'),
                'log_sha256': digest(root/'audit.log'), 'scientific_goal_complete': False})
        raise ValueError('independent audit failed; evidence retained and no automatic replay: '+failure)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--audit-worker', type=Path)
    mode.add_argument('--check-preparation', action='store_true', help='Read-only source/build/runtime/probe checks; no execution authority')
    args = parser.parse_args()
    if args.check_preparation:
        _preparation, pins = checked_preparation()
        print(canonical({'status': 'preparation-verified-no-execution-authority', 'pin_count': len(pins),
                         'pilot_executed': False, 'execution_authorized': False}))
    elif args.audit_worker:
        audit_worker(args.audit_worker)
    else:
        result = run_audit()
        print(canonical({'state': result['state'], 'scientific_goal_complete': False}))
