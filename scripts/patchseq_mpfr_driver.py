"""Synthetic-only MPFR candidate; requires an isolated upstream-tested toolchain."""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import platform
import shlex
import shutil
import subprocess
from pathlib import Path

import numpy as np
import yaml

ROOT = Path('results/patchseq-mpfr-toolchain-1504')
SOURCE = Path(__file__).with_name('patchseq_mpfr_kernel.cpp').resolve()
FIELDS = ('capacitance_pF', 'leak_nS', 'leak_reversal_mV', 'sodium_nS', 'potassium_nS',
          'm_current_nS', 'threshold_shift_mV', 'm_tau_max_ms')
FLAGS = ['-std=c++17', '-O2', '-shared', '-fPIC', '-fno-fast-math', '-ffp-contract=off']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_toolchain():
    path = ROOT/'checkpoint.json'
    if not path.is_file():
        raise RuntimeError('isolated MPFR dependency toolchain is not available')
    raw = path.read_bytes()
    data = json.loads(raw)
    if data.get('status') != 'isolated-dependencies-built-and-upstream-tests-passed-not-backend-equivalence':
        raise RuntimeError('isolated dependency build/tests are not terminal-success')
    prefix = (ROOT/'prefix').resolve()
    if data['configuration']['prefix'] != str(prefix):
        raise ValueError('isolated dependency prefix differs')
    pins = {**data['configuration']['source_hashes'], **data['installed_sha256']}
    if any(digest(p) != h for p, h in pins.items()):
        raise ValueError('isolated dependency source/tool/product hashes differ')
    for name in ('gmp-check', 'mpfr-check'):
        stage = data['stages'][name]
        if stage['status'] != 'passed' or digest(ROOT/(name+'.log')) != stage['log_sha256']:
            raise ValueError('upstream dependency test evidence differs')
    for dependency in data['configuration']['dependencies']:
        if digest(ROOT/dependency['archive']) != dependency['sha256']:
            raise ValueError('dependency archive changed')
    pins[str(path)] = hashlib.sha256(raw).hexdigest()
    return prefix, pins


def prepare_build(directory):
    prefix, pins = checked_toolchain()
    compiler = shutil.which('clang++')
    if compiler is None:
        raise RuntimeError('MPFR candidate requires clang++')
    compiler = str(Path(compiler).resolve())
    includes = ['-I', str(prefix/'include')]
    dependencies = subprocess.run([compiler, *FLAGS, *includes, '-M', '-MT', 'mpfr-candidate', str(SOURCE)],
                                  check=True, capture_output=True, text=True, timeout=60).stdout
    for path in shlex.split(dependencies.partition(':')[2].replace('\\\n', ' ')):
        pins[str(Path(path).resolve())] = digest(path)
    pins[compiler] = digest(compiler)
    pins[str(Path(__file__))] = digest(__file__)
    config = {'source_hashes': pins, 'compiler': compiler,
        'compiler_version': subprocess.run([compiler, '--version'], check=True, capture_output=True,
                                            text=True, timeout=10).stdout,
        'flags': FLAGS, 'prefix': str(prefix), 'platform': platform.platform(), 'machine': platform.machine(),
        'static_libraries': [str(prefix/'lib'/'libmpfr.a'), str(prefix/'lib'/'libgmp.a')]}
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    binary, receipt = directory/'kernel.so', directory/'build.yaml'
    if binary.exists() or receipt.exists():
        if not binary.is_file() or not receipt.is_file():
            raise ValueError('partial MPFR candidate build retained; refusing overwrite')
        saved = yaml.safe_load(receipt.read_text())
        if (any(saved.get(k) != v for k, v in config.items())
            or saved.get('binary_path') != str(binary.resolve())
            or saved.get('binary_sha256') != digest(binary)):
            raise ValueError('MPFR candidate build provenance changed')
        return binary, saved
    subprocess.run([compiler, *FLAGS, *includes, str(SOURCE), *config['static_libraries'],
                    '-lm', '-o', str(binary)], check=True, capture_output=True, timeout=120)
    if any(digest(p) != h for p, h in pins.items()):
        raise ValueError('MPFR candidate build input changed')
    config.update(binary_path=str(binary.resolve()), binary_sha256=digest(binary))
    with receipt.open('x') as file:
        yaml.safe_dump(config, file, sort_keys=False)
    return binary, config


class MPFRDriver:
    def __init__(self, directory):
        binary, self.provenance = prepare_build(directory)
        self.lib = ctypes.CDLL(str(binary))
        vector = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
        self.lib.mpfr_hh_integrate.argtypes = [ctypes.c_int, ctypes.c_size_t, vector, vector,
                                              ctypes.c_double, ctypes.c_int, vector, vector]
        self.lib.mpfr_hh_integrate.restype = ctypes.c_int
        self.lib.mpfr_hh_precision.argtypes = [ctypes.c_int]
        self.lib.mpfr_hh_precision.restype = ctypes.c_int
        self.lib.mpfr_hh_cancellation.argtypes = [ctypes.c_int]
        self.lib.mpfr_hh_cancellation.restype = ctypes.c_double
        for function in ('mpfr_hh_version', 'mpfr_hh_header_version', 'mpfr_hh_gmp_version'):
            getattr(self.lib, function).argtypes = []
            getattr(self.lib, function).restype = ctypes.c_char_p
        self.lib.mpfr_hh_tls.argtypes = []
        self.lib.mpfr_hh_tls.restype = ctypes.c_int
        self.versions = {'mpfr': self.lib.mpfr_hh_version().decode(),
                         'header': self.lib.mpfr_hh_header_version().decode(),
                         'gmp': self.lib.mpfr_hh_gmp_version().decode(), 'tls': self.lib.mpfr_hh_tls()}
        if self.versions != {'mpfr': '4.2.2', 'header': '4.2.2', 'gmp': '6.3.0', 'tls': 1}:
            raise ValueError('actual dependency version/header/thread-support mismatch')
        self.probes = {b: {'digits': self.lib.mpfr_hh_precision(b),
                           'cancellation': self.lib.mpfr_hh_cancellation(b)} for b in (160, 320)}
        if any(p['digits'] != b or p['cancellation'] != 2.**-100 for b, p in self.probes.items()):
            raise ValueError('MPFR candidate precision probes failed')

    def integrate(self, parameters, current, rate, initial, bits, factor):
        if (set(parameters) != set(FIELDS) or type(bits) is not int or bits not in (160, 320)
            or type(factor) is not int or factor not in (1, 2, 4, 8, 16)
            or isinstance(rate, bool) or not math.isfinite(rate) or not 20000 < rate <= 1000000):
            raise ValueError('invalid MPFR candidate specification; 53-bit control uses the sealed native engine')
        p = np.array([parameters[k] for k in FIELDS], dtype=np.float64)
        current = np.ascontiguousarray(current, dtype=np.float64)
        initial = np.ascontiguousarray(initial, dtype=np.float64)
        if (current.ndim != 1 or not 2 <= len(current) <= 1000000 or initial.shape != (5,)
            or not np.isfinite(current).all() or not np.isfinite(initial).all() or not np.isfinite(p).all()):
            raise ValueError('invalid explicit MPFR replay arrays')
        states = np.empty((len(current), 5), dtype=np.float64)
        code = self.lib.mpfr_hh_integrate(bits, len(current), p, current, rate, factor, initial, states)
        if code or not np.isfinite(states).all():
            raise ValueError('retained MPFR candidate integration failure code '+str(code))
        return states
