"""Synthetic-only chunk/state prototype; no source recording or fitting reader."""

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
from pathlib import Path

import numpy as np
import yaml
from patchseq_precision_driver import FIELDS, digest

SOURCE = Path(__file__).with_name('patchseq_precision_successor_kernel.cpp').resolve()
PREDECESSOR = Path(__file__).with_name('patchseq_hh_precision_kernel.cpp').resolve()
PREDECESSOR_SHA = '1d6b33d709948f5abfcddb4d6bf5a7049e48255e04c7faf7b66a65723c19f7d8'
HELPER = Path(__file__).with_name('patchseq_precision_driver.py').resolve()
HELPER_SHA = '145ffd59d8ab6bf6679fc1f04cfff340519d06b9cc955ebf8bbdf0af402096d0'
FLAGS = ['-std=c++17', '-O2', '-shared', '-fPIC', '-fno-fast-math', '-ffp-contract=off']


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def signature(value):
    return hashlib.sha256(encode(value)).hexdigest()


def prepare_build(directory):
    compiler = shutil.which('clang++')
    include = next((p for p in map(Path, ['/opt/homebrew/include', '/usr/include', '/usr/local/include'])
                    if (p/'boost/multiprecision/cpp_bin_float.hpp').is_file()), None)
    if not compiler or include is None:
        raise RuntimeError('optional successor requires clang++ and Boost headers')
    if digest(PREDECESSOR) != PREDECESSOR_SHA:
        raise ValueError('sealed predecessor arithmetic source changed')
    if digest(HELPER) != HELPER_SHA:
        raise ValueError('sealed Python field/hash helper changed')
    compiler = str(Path(compiler).resolve())
    output = subprocess.run([compiler, *FLAGS, '-I', str(include), '-M', '-MT', 'successor', str(SOURCE)],
                            check=True, capture_output=True, text=True, timeout=60).stdout
    dependencies = {str(Path(p).resolve()): digest(p) for p in
                    shlex.split(output.partition(':')[2].replace('\\\n', ' '))}
    if str(SOURCE) not in dependencies or dependencies.get(str(PREDECESSOR)) != PREDECESSOR_SHA:
        raise ValueError('successor dependency closure incomplete')
    receipt = {'source_sha256': digest(SOURCE), 'driver_sha256': digest(__file__),
               'python_sources': {str(HELPER): HELPER_SHA}, 'parameter_fields': list(FIELDS),
               'compiler': compiler, 'compiler_sha256': digest(compiler),
               'compiler_version': subprocess.run([compiler, '--version'], check=True,
                    capture_output=True, text=True, timeout=10).stdout,
               'flags': FLAGS, 'include_root': str(include), 'dependencies': dependencies,
               'platform': platform.platform(), 'machine': platform.machine()}
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    binary, path = directory/'kernel.so', directory/'build.yaml'
    if binary.exists() or path.exists():
        if not binary.is_file() or not path.is_file():
            raise ValueError('partial successor build retained; refusing overwrite')
        saved = yaml.safe_load(path.read_text())
        if (any(saved.get(k) != v for k, v in receipt.items())
            or saved.get('binary_path') != str(binary.resolve())
            or saved.get('binary_sha256') != digest(binary)):
            raise ValueError('successor build provenance changed')
        return binary, saved
    subprocess.run([compiler, *FLAGS, '-I', str(include), str(SOURCE), '-o', str(binary)],
                   check=True, capture_output=True, timeout=120)
    if (digest(compiler) != receipt['compiler_sha256'] or digest(HELPER) != HELPER_SHA
        or any(digest(p) != h for p, h in dependencies.items())):
        raise ValueError('successor compiler/dependencies changed during build')
    receipt.update(binary_path=str(binary.resolve()), binary_sha256=digest(binary))
    with path.open('x') as file:
        yaml.safe_dump(receipt, file, sort_keys=False)
    return binary, receipt


class SuccessorDriver:
    def __init__(self, directory):
        binary, self.provenance = prepare_build(directory)
        self.build_identity = signature(self.provenance)
        self.lib = ctypes.CDLL(str(binary))
        pointer = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
        self.lib.successor_create.argtypes = [ctypes.c_int, pointer, pointer, ctypes.c_double, ctypes.c_int]
        self.lib.successor_create.restype = ctypes.c_void_p
        self.lib.successor_destroy.argtypes = [ctypes.c_void_p]
        self.lib.successor_destroy.restype = None
        self.lib.successor_advance.argtypes = [ctypes.c_void_p, ctypes.c_size_t, pointer, pointer]
        self.lib.successor_advance.restype = ctypes.c_int
        self.lib.successor_snapshot.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        self.lib.successor_snapshot.restype = ctypes.c_size_t
        self.lib.successor_restore.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
        self.lib.successor_restore.restype = ctypes.c_int
        self.lib.precision_digits.argtypes = [ctypes.c_int]
        self.lib.precision_digits.restype = ctypes.c_int
        self.lib.precision_probe.argtypes = [ctypes.c_int]
        self.lib.precision_probe.restype = ctypes.c_double
        self.probes = {bits: {'digits': self.lib.precision_digits(bits),
                             'cancellation': self.lib.precision_probe(bits)} for bits in (53, 160, 320)}
        if any(p['digits'] != bits or p['cancellation'] != (0 if bits == 53 else 2.**-100)
               for bits, p in self.probes.items()):
            raise ValueError('successor precision probes failed')

    def session(self, parameters, current, rate, initial, bits, factor):
        return Session(self, parameters, current, rate, initial, bits, factor)


class Session:
    """One locked native context, advancing absolute native-grid interval indices."""

    def __init__(self, driver, parameters, current, rate, initial, bits, factor):
        self._lock = threading.RLock()
        self.handle = None
        if (set(parameters) != set(FIELDS) or type(bits) is not int or bits not in (53, 160, 320)
            or type(factor) is not int or factor not in (1, 2, 4, 8, 16)
            or isinstance(rate, bool) or not math.isfinite(rate) or not 20000 < rate <= 1000000):
            raise ValueError('invalid successor session configuration')
        self._current = np.array(current, dtype=np.float64, copy=True, order='C')
        initial = np.array(initial, dtype=np.float64, copy=True, order='C')
        p = np.array([parameters[k] for k in FIELDS], dtype=np.float64)
        if (self._current.ndim != 1 or not 2 <= len(self._current) <= 1000000 or initial.shape != (5,)
            or not np.isfinite(p).all() or not np.isfinite(initial).all() or not np.isfinite(self._current).all()):
            raise ValueError('invalid explicit current/initial/parameter arrays')
        self._current.flags.writeable = False
        self.driver, self._cursor, self.failed = driver, 0, False
        self._configuration_bytes = encode({'build_identity': driver.build_identity, 'bits': bits, 'factor': factor,
                              'rate_hz': float(rate), 'parameters': p.tolist(), 'initial': initial.tolist(),
                              'current_samples': len(self._current),
                              'current_sha256': hashlib.sha256(self._current.tobytes()).hexdigest()})
        self.handle = driver.lib.successor_create(bits, p, initial, rate, factor)
        if not self.handle:
            raise ValueError('successor refused invalid model parameters or initial state')

    def _available(self):
        if not self.handle or self.failed:
            raise RuntimeError('closed or failed successor context cannot continue')

    @property
    def cursor(self):
        return self._cursor

    def advance(self, stop):
        """Return boundary-inclusive states; adjacent chunks overlap by one row."""
        with self._lock:
            self._available()
            if type(stop) is not int or not self.cursor <= stop < len(self._current):
                raise ValueError('invalid absolute native interval endpoint')
            count = stop-self.cursor
            out = np.empty((count+1, 5), dtype=np.float64)
            values = self._current[self.cursor:stop] if count else self._current[:1]
            code = self.driver.lib.successor_advance(self.handle, count, values, out)
            if code or not np.isfinite(out).all():
                self.failed = True
                raise ValueError('retained successor integration failure; context poisoned')
            self._cursor = stop
            return out

    def snapshot(self):
        with self._lock:
            self._available()
            length = self.driver.lib.successor_snapshot(self.handle, None, 0)
            if not 1 < length <= 8192:
                raise ValueError('invalid full-precision checkpoint size')
            buffer = ctypes.create_string_buffer(length)
            if self.driver.lib.successor_snapshot(self.handle, buffer, length) != length:
                raise ValueError('successor checkpoint serialization failed')
            row = {'format': 1, 'configuration': json.loads(self._configuration_bytes),
                   'cursor': self.cursor, 'state': buffer.value.decode('ascii')}
            return {**row, 'sha256': signature(row)}

    def restore(self, checkpoint):
        with self._lock:
            self._available()
            if self.cursor != 0:
                raise ValueError('restore requires a fresh context; no in-place rewind')
            if set(checkpoint) != {'format', 'configuration', 'cursor', 'state', 'sha256'}:
                raise ValueError('checkpoint schema differs')
            row = {k: v for k, v in checkpoint.items() if k != 'sha256'}
            if (type(row['format']) is not int or row['format'] != 1
                or signature(row) != checkpoint['sha256']
                or encode(row['configuration']) != self._configuration_bytes
                or type(row['cursor']) is not int or not 0 <= row['cursor'] < len(self._current)
                or type(row['state']) is not str):
                raise ValueError('checkpoint checksum, configuration or position differs')
            text = row['state'].encode('ascii')
            if (len(text) > 8192 or b'\0' in text
                or self.driver.lib.successor_restore(self.handle, text, row['cursor'])):
                raise ValueError('invalid canonical full-precision state')
            self._cursor = row['cursor']

    def close(self):
        with self._lock:
            if self.handle:
                self.driver.lib.successor_destroy(self.handle)
                self.handle = None

    def __enter__(self):
        self._available()
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        if hasattr(self, '_lock'):
            self.close()
