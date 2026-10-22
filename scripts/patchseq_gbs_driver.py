"""Bounded generic GBS prototype for manufactured linear systems, not HH fitting.

The two equations are y'[j] = rate[j]*y[j] + gain[j]*current. Times are in ms.
The autonomous RHS receives a fixed segment current, not a rounded stage clock.
Local error control is the stock infinity norm of error divided by
atol + rtol*(abs(y) + abs(step*dy/dt)); both scale factors are exactly one.
Dense-error control is explicitly enabled, with the upstream factor-10 gate.
These local controls are NOT the scientific whole-trajectory acceptance gates.
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
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np
from patchseq_mpfr_driver import checked_toolchain

SCRIPTS = Path(__file__).resolve().parent
SOURCE = SCRIPTS/'patchseq_gbs_kernel.cpp'
FLAGS = ('-std=c++17', '-O2', '-shared', '-fPIC', '-fno-fast-math', '-ffp-contract=off')
SCALARS = ('absolute', 'relative', 'initial_step_ms', 'maximum_step_ms', 'minimum_step_ms')
BUDGETS = ('maximum_rhs', 'maximum_steps', 'maximum_rejections')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


@dataclass(frozen=True)
class Policy:
    # Engineering test defaults only. A future HH experiment must register its
    # own complete policy and budgets prospectively; this class is not a seal.
    absolute: str = '1e-20'
    relative: str = '1e-20'
    initial_step_ms: str = '0.02'
    maximum_step_ms: str = '0.2'
    minimum_step_ms: str = '1e-30'
    maximum_rhs: int = 1_000_000
    maximum_steps: int = 100_000
    maximum_rejections: int = 10_000

    def __post_init__(self):
        values = []
        for key in SCALARS:
            text = getattr(self, key)
            if type(text) is not str or not 1 <= len(text) <= 4096 or '\x00' in text:
                raise ValueError('policy constants must be explicit decimal strings')
            try:
                number = Decimal(text)
            except InvalidOperation as exc:
                raise ValueError('invalid policy decimal') from exc
            if not number.is_finite() or number <= 0:
                raise ValueError('policy constants must be positive and finite')
            values.append(number)
        if not values[4] <= values[2] <= values[3]:
            raise ValueError('invalid minimum/initial/maximum step ordering')
        for key in BUDGETS:
            value = getattr(self, key)
            if type(value) is not int or not 0 <= value <= 2**63-1:
                raise ValueError('budgets must be nonnegative bounded integers')


def verify_pins(pins):
    if any(digest(path) != expected for path, expected in pins.items()):
        raise ValueError('GBS source, dependency or compiler provenance changed')


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
    dependencies = subprocess.run([compiler, *flags, '-M', '-MT', 'gbs-linear', str(SOURCE)],
                                  check=True, capture_output=True, text=True, timeout=60).stdout
    paths = [Path(p).resolve() for p in shlex.split(dependencies.partition(':')[2].replace('\\\n', ' '))]
    paths += [Path(__file__).resolve(), Path(compiler), SCRIPTS/'third_party/BOOST_LICENSE_1_0.txt']
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
    binary, receipt, lock = (directory/name for name in ('linear.so', 'build.json', 'build.lock'))
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
        file.write('bounded-gbs-manufactured-build\n')
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


class LinearDriver:
    """Per-context owned state and serialized API access; independent contexts may run concurrently."""

    def __init__(self, directory):
        binary, receipt = prepare_build(directory)
        self._receipt = canonical(receipt)
        self._lib = ctypes.CDLL(str(binary))
        vector = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
        uints = np.ctypeslib.ndpointer(dtype=np.uint64, flags='C_CONTIGUOUS')
        error = [ctypes.c_void_p, ctypes.c_size_t]
        self._lib.gbs_linear_create.argtypes = [ctypes.c_int, ctypes.c_size_t, vector, ctypes.c_double,
            vector, vector, vector, ctypes.POINTER(ctypes.c_char_p), uints, ctypes.c_char_p, *error]
        self._lib.gbs_linear_create.restype = ctypes.c_void_p
        self._lib.gbs_linear_destroy.argtypes = [ctypes.c_void_p]
        self._lib.gbs_linear_destroy.restype = None
        self._lib.gbs_linear_advance.argtypes = [ctypes.c_void_p, ctypes.c_size_t, vector, *error]
        self._lib.gbs_linear_stats.argtypes = [ctypes.c_void_p, uints, *error]
        self._lib.gbs_linear_snapshot.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, *error]
        self._lib.gbs_linear_restore.argtypes = [ctypes.c_void_p, ctypes.c_char_p, *error]
        self._lib.gbs_linear_environment.argtypes = error
        self._lib.gbs_linear_stock_probe.argtypes = [ctypes.c_int, ctypes.c_int, uints, *error]
        self._lib.gbs_linear_nonfinite_probe.argtypes = [ctypes.c_int, *error]
        for name in ('advance', 'stats', 'snapshot', 'restore', 'environment', 'stock_probe', 'nonfinite_probe'):
            getattr(self._lib, 'gbs_linear_'+name).restype = ctypes.c_int
        buffer = ctypes.create_string_buffer(4096)
        if self._lib.gbs_linear_environment(buffer, len(buffer)):
            raise ValueError('GBS actual arithmetic environment probe failed')
        environment = json.loads(buffer.value)
        if environment != {'mpfr': '4.2.2', 'header': '4.2.2', 'gmp': '6.3.0', 'tls': 1,
                                'emin': -1073741823, 'emax': 1073741823,
                                'precision160': 160, 'precision320': 320}:
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

    def context(self, current, rate, initial, rates, gains, bits, policy):
        self.verify()
        return LinearContext(self, current, rate, initial, rates, gains, bits, policy)


def _array(value, shape):
    raw = np.asarray(value)
    if (raw.dtype.kind not in 'fiu' or raw.shape != shape
        or (raw.dtype.kind == 'f' and raw.dtype.itemsize > 8)
        or (raw.dtype.kind == 'i' and ((raw < -(2**53)).any() or (raw > 2**53).any()))
        or (raw.dtype.kind == 'u' and (raw > 2**53).any())):
        raise ValueError('manufactured inputs must be real numeric arrays of the required shape')
    result = np.array(raw, dtype=np.float64, order='C', copy=True)
    if not np.isfinite(result).all():
        raise ValueError('manufactured inputs must be finite')
    return result


class LinearContext:
    def __init__(self, driver, current, rate, initial, rates, gains, bits, policy):
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
        initial, rates, gains = (_array(value, (2,)) for value in (initial, rates, gains))
        edges = np.flatnonzero(current[1:-1].view(np.uint64) != current[:-2].view(np.uint64))+1
        self.barriers = tuple(sorted({0, len(current)-1, *range(4096, len(current)-1, 4096), *map(int, edges)}))
        self._configuration = canonical({'kind': 'manufactured-linear-2-v1', 'bits': bits,
            'policy': asdict(policy), 'rate_binary64': float(rate).hex(), 'samples': len(current),
            'current_sha256': hashlib.sha256(current.tobytes()).hexdigest(),
            'initial': [float(x).hex() for x in initial], 'rates': [float(x).hex() for x in rates],
            'gains': [float(x).hex() for x in gains], 'build_sha256': identity(driver.provenance),
            'barrier_intervals': 4096, 'environment': driver.environment})
        scalar_array = (ctypes.c_char_p*5)(*(getattr(policy, k).encode() for k in SCALARS))
        budgets = np.array([getattr(policy, k) for k in BUDGETS], dtype=np.uint64)
        error = ctypes.create_string_buffer(4096)
        self._pointer = driver._lib.gbs_linear_create(bits, len(current), current, float(rate), initial,
            rates, gains, scalar_array, budgets, identity(json.loads(self._configuration)).encode(),
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
        code = getattr(self._driver._lib, 'gbs_linear_'+name)(self._pointer, *arguments, error, len(error))
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
            output = np.full((stop-start+1, 2), np.nan, dtype=np.float64)
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
            contents = {'schema': 'gbs-linear-checkpoint-v1', 'configuration': json.loads(self._configuration),
                        'payload': buffer.value.decode()}
            # An integrity checksum, NOT an authentication/signature scheme.
            return {**contents, 'sha256': identity(contents)}

    def restore(self, envelope):
        with self._lock:
            self._ready()
            self._driver.verify()
            if (type(envelope) is not dict or set(envelope) != {'schema', 'configuration', 'payload', 'sha256'}
                or envelope['schema'] != 'gbs-linear-checkpoint-v1'
                or canonical(envelope['configuration']) != self._configuration
                or type(envelope['payload']) is not str or '\x00' in envelope['payload']
                or len(envelope['payload']) > 16384
                or identity({k: v for k, v in envelope.items() if k != 'sha256'}) != envelope['sha256']):
                raise ValueError('checkpoint schema, integrity or input/build/policy identity differs')
            self._call('restore', envelope['payload'].encode())

    def close(self):
        with self._lock:
            if self._pointer:
                self._driver._lib.gbs_linear_destroy(self._pointer)
                self._pointer = None

    def __enter__(self):
        self._ready()
        return self

    def __exit__(self, *unused):
        self.close()

    def __del__(self):
        if getattr(self, '_pointer', None):
            self.close()
