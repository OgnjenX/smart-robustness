"""Synthetic HH adapter with explicit shared initialization and frozen-RHS reuse.

This is the PRIMARY GBS route, not the separately implemented forward reference.
No fit runner, response-file loader or network code is imported here.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import math
import platform
import shlex
import shutil
import subprocess
import threading
from dataclasses import asdict
from pathlib import Path

import numpy as np
from patchseq_gbs_driver import (
    BUDGETS,
    FLAGS,
    SCALARS,
    Policy,
    _array,
    canonical,
    digest,
    identity,
    verify_pins,
)
from patchseq_mpfr_driver import FIELDS, checked_toolchain

SCRIPTS = Path(__file__).resolve().parent
SOURCE = SCRIPTS/'patchseq_gbs_hh.cpp'


def parameters_array(parameters):
    if (type(parameters) is not dict or set(parameters) != set(FIELDS)
        or any(isinstance(value, (bool, np.bool_)) for value in parameters.values())):
        raise ValueError('exact original HH parameter fields are required')
    values = _array([parameters[key] for key in FIELDS], (8,))
    if (values[0] <= 0 or values[1] <= 0 or values[7] <= 0
        or any(values[j] < 0 for j in (3, 4, 5))):
        raise ValueError('invalid capacitance, conductance or slow time constant')
    return values


def prepare_build(directory):
    """Hash the actual transitive headers and flags; retain partial builds."""
    prefix, prerequisite_pins = checked_toolchain()
    compiler = shutil.which('clang++')
    boost = next((p for p in map(Path, ['/opt/homebrew/include', '/usr/include', '/usr/local/include'])
                  if (p/'boost/numeric/odeint/stepper/bulirsch_stoer_dense_out.hpp').is_file()), None)
    if compiler is None or boost is None:
        raise RuntimeError('optional GBS prototype requires clang++ and Boost')
    compiler = str(Path(compiler).resolve())
    flags = [*FLAGS, '-I', str(SCRIPTS), '-I', str(prefix/'include'), '-I', str(boost)]
    dependencies = subprocess.run([compiler, *flags, '-M', '-MT', 'gbs-hh', str(SOURCE)],
                                  check=True, capture_output=True, text=True, timeout=60).stdout
    paths = [Path(p).resolve() for p in shlex.split(dependencies.partition(':')[2].replace('\\\n', ' '))]
    paths += [Path(__file__).resolve(), Path(compiler), SCRIPTS/'third_party/BOOST_LICENSE_1_0.txt',
              SCRIPTS/'patchseq_gbs_driver.py', SCRIPTS/'patchseq_mpfr_driver.py']
    pins = {str(Path(p).resolve()): h for p, h in prerequisite_pins.items()}
    pins.update({str(p): digest(p) for p in paths})
    libraries = [str(prefix/'lib/libmpfr.a'), str(prefix/'lib/libgmp.a')]
    pins.update({p: digest(p) for p in libraries})
    configuration = {'source_hashes': pins, 'compiler': compiler, 'flags': flags,
        'compiler_version': subprocess.run([compiler, '--version'], check=True, capture_output=True,
                                            text=True, timeout=10).stdout,
        'static_libraries': libraries, 'platform': platform.platform(), 'machine': platform.machine()}
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    binary, receipt, lock = (directory/name for name in ('hh.so', 'build.json', 'build.lock'))
    if lock.exists():
        raise ValueError('active or orphan GBS build lock; refusing overwrite')
    if binary.exists() or receipt.exists():
        if not binary.is_file() or not receipt.is_file():
            raise ValueError('partial GBS build retained; refusing overwrite')
        saved = json.loads(receipt.read_text())
        if (set(saved) != set(configuration) | {'binary_path', 'binary_sha256'}
            or any(saved.get(k) != v for k, v in configuration.items())
            or saved['binary_path'] != str(binary) or saved['binary_sha256'] != digest(binary)):
            raise ValueError('GBS build receipt or binary provenance differs')
        verify_pins(pins)
        return binary, saved
    # The lock is exclusive; a failed compile retains its artifacts, not an
    # apparently valid receipt. No stale-lock deletion or automatic repair.
    with lock.open('x') as file:
        file.write('bounded-gbs-hh-build\n')
    try:
        if binary.exists() or receipt.exists():
            raise ValueError('GBS build appeared while acquiring lock')
        subprocess.run([compiler, *flags, str(SOURCE), *libraries, '-lm', '-o', str(binary)],
                       check=True, capture_output=True, text=True, timeout=120)
        verify_pins(pins)
        configuration.update(binary_path=str(binary), binary_sha256=digest(binary))
        with receipt.open('x') as file:
            file.write(canonical(configuration)+'\n')
    finally:
        lock.unlink()
    return binary, configuration


class HHDriver:
    """Per-context owned state and serialized API access; independent contexts may run concurrently."""

    def __init__(self, directory):
        binary, receipt = prepare_build(directory)
        self._receipt = canonical(receipt)
        self._lib = ctypes.CDLL(str(binary))
        vector = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
        uints = np.ctypeslib.ndpointer(dtype=np.uint64, flags='C_CONTIGUOUS')
        error = [ctypes.c_void_p, ctypes.c_size_t]
        self._lib.gbs_hh_create.argtypes = [ctypes.c_int, ctypes.c_size_t, vector, vector, ctypes.c_double,
            vector, ctypes.POINTER(ctypes.c_char_p), uints, ctypes.c_char_p, *error]
        self._lib.gbs_hh_create.restype = ctypes.c_void_p
        self._lib.gbs_hh_destroy.argtypes = [ctypes.c_void_p]
        self._lib.gbs_hh_destroy.restype = None
        self._lib.gbs_hh_advance.argtypes = [ctypes.c_void_p, ctypes.c_size_t, vector, *error]
        self._lib.gbs_hh_stats.argtypes = [ctypes.c_void_p, uints, *error]
        self._lib.gbs_hh_snapshot.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, *error]
        self._lib.gbs_hh_restore.argtypes = [ctypes.c_void_p, ctypes.c_char_p, *error]
        self._lib.gbs_hh_environment.argtypes = error
        self._lib.gbs_hh_rhs_exact.argtypes = [ctypes.c_int, vector, vector, ctypes.c_double,
            ctypes.c_void_p, ctypes.c_size_t, *error]
        self._lib.gbs_hh_gate_probe.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p), *error]
        self._lib.gbs_hh_accepted_gate_probe.argtypes = [ctypes.c_int, *error]
        self._lib.gbs_hh_dense_gate_probe.argtypes = [ctypes.c_int, *error]
        for name in ('advance', 'stats', 'snapshot', 'restore', 'environment', 'rhs_exact',
                     'gate_probe', 'accepted_gate_probe', 'dense_gate_probe'):
            getattr(self._lib, 'gbs_hh_'+name).restype = ctypes.c_int
        buffer = ctypes.create_string_buffer(4096)
        if self._lib.gbs_hh_environment(buffer, len(buffer)):
            raise ValueError('GBS actual arithmetic environment probe failed')
        environment = json.loads(buffer.value)
        if environment != {'mpfr': '4.2.2', 'header': '4.2.2', 'gmp': '6.3.0', 'tls': 1,
                                'emin': -1073741823, 'emax': 1073741823,
                                'precision160': 160, 'precision320': 320,
                                'cancellation160': 1, 'cancellation320': 1}:
            raise ValueError('GBS actual widths, versions or exponent range differs')
        self._environment = canonical(environment)

    @property
    def environment(self):
        return json.loads(self._environment)

    @property
    def provenance(self):
        return json.loads(self._receipt)

    def verify(self):
        receipt = self.provenance
        verify_pins(receipt['source_hashes'])
        if digest(receipt['binary_path']) != receipt['binary_sha256']:
            raise ValueError('loaded GBS binary changed')

    def context(self, parameters, current, rate, initial, bits, policy):
        self.verify()
        return HHContext(self, parameters, current, rate, initial, bits, policy)

    def derivative_exact(self, parameters, state, current, bits):
        self.verify()
        if (type(bits) is not int or bits not in (160, 320)
            or isinstance(current, bool) or not isinstance(current, (float, int))
            or (type(current) is int and abs(current) > 2**53)
            or not math.isfinite(current)):
            raise ValueError('invalid explicit HH derivative probe')
        p, state = parameters_array(parameters), _array(state, (5,))
        output, error = ctypes.create_string_buffer(4096), ctypes.create_string_buffer(4096)
        code = self._lib.gbs_hh_rhs_exact(bits, p, state, float(current), output, len(output), error, len(error))
        if code:
            raise ValueError(error.value.decode())
        self.verify()
        return output.value.decode().split('|')


class HHContext:
    def __init__(self, driver, parameters, current, rate, initial, bits, policy):
        self._pointer = None
        self._lock = threading.RLock()
        self._failed = False
        self._driver = driver
        if (type(bits) is not int or bits not in (160, 320) or not isinstance(policy, Policy)
            or isinstance(rate, (bool, np.bool_)) or not isinstance(rate, (float, int, np.floating))
            or (isinstance(rate, np.floating) and rate.dtype.itemsize > 8)
            or not math.isfinite(rate) or not 20000 < rate <= 1000000):
            raise ValueError('invalid precision, policy or native rate')
        raw = np.asarray(current)
        if raw.ndim != 1 or not 2 <= len(raw) <= 1000000:
            raise ValueError('invalid native current support')
        current = _array(raw, raw.shape)
        initial = _array(initial, (5,))
        p = parameters_array(parameters)
        edges = np.flatnonzero(current[1:-1].view(np.uint64) != current[:-2].view(np.uint64))+1
        self.barriers = tuple(sorted({0, len(current)-1, *range(4096, len(current)-1, 4096), *map(int, edges)}))
        self._configuration = canonical({'kind': 'gbs-hh-primary-5-v1', 'bits': bits,
            'policy': asdict(policy), 'rate_binary64': float(rate).hex(), 'samples': len(current),
            'current_sha256': hashlib.sha256(current.tobytes()).hexdigest(),
            'initial': [float(x).hex() for x in initial], 'parameters': [float(x).hex() for x in p],
            'build_sha256': identity(driver.provenance),
            'barrier_intervals': 4096, 'environment': driver.environment})
        scalar_array = (ctypes.c_char_p*5)(*(getattr(policy, k).encode() for k in SCALARS))
        budgets = np.array([getattr(policy, k) for k in BUDGETS], dtype=np.uint64)
        error = ctypes.create_string_buffer(4096)
        self._pointer = driver._lib.gbs_hh_create(bits, len(current), p, current, float(rate), initial,
            scalar_array, budgets, identity(json.loads(self._configuration)).encode(),
            error, len(error))
        if not self._pointer:
            raise ValueError('GBS context creation failed: '+error.value.decode())

    def _ready(self):
        if not self._pointer:
            raise ValueError('closed GBS context')
        if self._failed:
            raise ValueError('failed context cannot be reused')

    def _call(self, name, *arguments):
        error = ctypes.create_string_buffer(4096)
        code = getattr(self._driver._lib, 'gbs_hh_'+name)(self._pointer, *arguments, error, len(error))
        if code:
            raise ValueError(error.value.decode())

    @property
    def stats(self):
        with self._lock:
            if not self._pointer:
                raise ValueError('closed GBS context')
            values = np.empty(5, dtype=np.uint64)
            self._call('stats', values)
            return dict(zip(('position', 'rhs', 'accepted', 'rejected', 'dense'), map(int, values), strict=True))

    def advance(self, stop):
        with self._lock:
            self._ready()
            start = self.stats['position']
            if type(stop) is not int or stop < start or stop not in self.barriers:
                raise ValueError('advance must end at a prescribed restart barrier')
            self._driver.verify()
            output = np.full((stop-start+1, 5), np.nan, dtype=np.float64)
            try:
                self._call('advance', stop, output)
                self._driver.verify()
            except ValueError:
                self._failed = True
                raise
            if not np.isfinite(output).all():
                self._failed = True
                raise ValueError('nonfinite GBS observations')
            return output

    def snapshot(self):
        with self._lock:
            self._ready()
            self._driver.verify()
            buffer = ctypes.create_string_buffer(16384)
            self._call('snapshot', buffer, len(buffer))
            contents = {'schema': 'gbs-hh-checkpoint-v1', 'configuration': json.loads(self._configuration),
                        'payload': buffer.value.decode()}
            # An integrity checksum, NOT an authentication/signature scheme.
            return {**contents, 'sha256': identity(contents)}

    def restore(self, envelope):
        with self._lock:
            self._ready()
            self._driver.verify()
            if (type(envelope) is not dict or set(envelope) != {'schema', 'configuration', 'payload', 'sha256'}
                or envelope['schema'] != 'gbs-hh-checkpoint-v1'
                or canonical(envelope['configuration']) != self._configuration
                or type(envelope['payload']) is not str or '\x00' in envelope['payload']
                or len(envelope['payload']) > 16384
                or identity({k: v for k, v in envelope.items() if k != 'sha256'}) != envelope['sha256']):
                raise ValueError('checkpoint schema, integrity or input/build/policy identity differs')
            self._call('restore', envelope['payload'].encode())

    def close(self):
        with self._lock:
            if self._pointer:
                self._driver._lib.gbs_hh_destroy(self._pointer)
                self._pointer = None

    def __enter__(self):
        self._ready()
        return self

    def __exit__(self, *unused):
        self.close()

    def __del__(self):
        if getattr(self, '_pointer', None):
            self.close()
