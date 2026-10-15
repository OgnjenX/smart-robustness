"""Diagnostic-only build receipts and explicit-state multiprecision replay."""

from __future__ import annotations

import ctypes
import hashlib
import math
import platform
import shlex
import shutil
import subprocess
from pathlib import Path

import numpy as np
import yaml

SOURCE = Path(__file__).with_name('patchseq_hh_precision_kernel.cpp').resolve()
FLAGS = ['-std=c++17', '-O2', '-shared', '-fPIC', '-fno-fast-math', '-ffp-contract=off']
FIELDS = ('capacitance_pF', 'leak_nS', 'leak_reversal_mV', 'sodium_nS', 'potassium_nS',
          'm_current_nS', 'threshold_shift_mV', 'm_tau_max_ms')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_build(directory):
    compiler = shutil.which('clang++')
    include = next((p for p in map(Path, ['/opt/homebrew/include', '/usr/include',
                                        '/usr/local/include'])
                    if (p/'boost/multiprecision/cpp_bin_float.hpp').is_file()), None)
    if compiler is None or include is None:
        raise RuntimeError('optional precision diagnostic requires clang++ and Boost headers')
    compiler = str(Path(compiler).resolve())
    dependency_list = subprocess.run([compiler, '-std=c++17', '-I', str(include), '-M',
                                      '-MT', 'precision', str(SOURCE)],
                                     check=True, capture_output=True, text=True, timeout=60).stdout
    paths = shlex.split(dependency_list.partition(':')[2].replace('\\\n', ' '))
    dependencies = {str(Path(p).resolve()): digest(p) for p in paths}
    if str(SOURCE) not in dependencies or not any('/boost/' in p for p in dependencies):
        raise ValueError('missing source or Boost dependency closure')
    signature = {
        'source_sha256': digest(SOURCE), 'driver_sha256': digest(__file__),
        'compiler': compiler, 'compiler_sha256': digest(compiler),
        'compiler_version': subprocess.run([compiler, '--version'], check=True,
            capture_output=True, text=True, timeout=10).stdout,
        'flags': FLAGS, 'include_root': str(include), 'dependencies': dependencies,
        'platform': platform.platform(), 'machine': platform.machine(),
    }
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    library, receipt = directory/'kernel.so', directory/'build.yaml'
    if library.exists() or receipt.exists():
        if not library.is_file() or not receipt.is_file():
            raise ValueError('partial precision build retained; refusing overwrite')
        saved = yaml.safe_load(receipt.read_text())
        if (any(saved.get(k) != v for k, v in signature.items())
            or saved.get('binary_sha256') != digest(library)
            or saved.get('binary_path') != str(library.resolve())):
            raise ValueError('precision build receipt differs; refusing rebuild')
        return library, saved
    subprocess.run([compiler, *FLAGS, '-I', str(include), str(SOURCE), '-o', str(library)],
                   check=True, capture_output=True, timeout=120)
    if (digest(compiler) != signature['compiler_sha256']
        or any(digest(p) != h for p, h in dependencies.items())):
        raise ValueError('compiler or headers changed during build')
    signature.update(binary_sha256=digest(library), binary_path=str(library.resolve()))
    with receipt.open('x') as file:
        yaml.safe_dump(signature, file, sort_keys=False)
    return library, signature


class PrecisionDriver:
    """Only accepts explicit current and initial state; never opens biological data."""

    def __init__(self, directory):
        library, self.provenance = prepare_build(directory)
        self.lib = ctypes.CDLL(str(library))
        pointer = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
        self.lib.precision_integrate.argtypes = [ctypes.c_int, ctypes.c_size_t, pointer,
            pointer, ctypes.c_double, ctypes.c_int, pointer, pointer]
        self.lib.precision_integrate.restype = ctypes.c_int
        self.lib.precision_digits.argtypes = [ctypes.c_int]
        self.lib.precision_digits.restype = ctypes.c_int
        self.lib.precision_probe.argtypes = [ctypes.c_int]
        self.lib.precision_probe.restype = ctypes.c_double
        self.probes = {bits: {'digits': self.lib.precision_digits(bits),
                             'cancellation': self.lib.precision_probe(bits)}
                       for bits in (53, 160, 320)}
        if any(p['digits'] != bits or p['cancellation'] != (0 if bits == 53 else 2.**-100)
               for bits, p in self.probes.items()):
            raise ValueError('retained build failed arithmetic precision probes')

    def integrate(self, parameters, current, rate_hz, initial, bits, factor):
        if (set(parameters) != set(FIELDS) or type(bits) is not int or bits not in self.probes
            or type(factor) is not int or factor not in (1, 2, 4, 8, 16)
            or not math.isfinite(rate_hz) or not 20000 < rate_hz <= 1000000):
            raise ValueError('invalid precision replay specification')
        current = np.ascontiguousarray(current, dtype=np.float64)
        initial = np.ascontiguousarray(initial, dtype=np.float64)
        p = np.array([parameters[k] for k in FIELDS], dtype=np.float64)
        if (current.ndim != 1 or not 2 <= len(current) <= 1000000 or initial.shape != (5,)
            or not np.isfinite(current).all() or not np.isfinite(initial).all()
            or not np.isfinite(p).all()):
            raise ValueError('invalid precision current, parameters or explicit initial state')
        states = np.empty((len(current), 5), dtype=np.float64)
        code = self.lib.precision_integrate(bits, len(current), p, current, rate_hz,
                                             factor, initial, states)
        if code or not np.isfinite(states).all():
            raise ValueError('precision replay failed with retained failure code '+str(code))
        return states


def shared_brian_initial_state(parameters, rate_hz):
    """Capture original equilibrium rule once, before warmup; do not integrate."""
    import brian2 as brian
    from patchseq_isolated_hh import Parameters, population

    previous = brian.prefs.codegen.target
    try:
        brian.prefs.codegen.target = 'numpy'
        clock = brian.Clock(dt=(1/rate_hz)*brian.second)
        drive = brian.TimedArray(np.zeros((2, 1))*brian.pA, dt=clock.dt)
        group = population([Parameters(**parameters)], clock, drive,
                           [parameters['leak_reversal_mV']], brian)
        network = brian.Network(group)  # Explicit ownership; no run or response access.
        state = np.array([float(group.v[0]/brian.mV), float(group.m[0]), float(group.h[0]),
                          float(group.n[0]), float(group.p[0])])
        assert network.t == 0*brian.second
        return state
    finally:
        brian.prefs.codegen.target = previous
