"""Provenance-recording optional verifier driver; not installed into live runners."""

from __future__ import annotations

import ctypes
import hashlib
import math
import platform
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path

import numpy as np
import yaml
from patchseq_hh_compiled_prototype import FLAGS, SOURCE
from patchseq_hh_independent_replay import parameters

KERNEL_SHA256 = 'c53d113be1f16f580b18fb379ed44cf569fcc49139a505256e6a43057dd27721'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_build(directory):
    compiler = shutil.which('clang++')
    if compiler is None:
        raise RuntimeError('optional driver requires an explicitly available clang++')
    compiler = str(Path(compiler).resolve())
    if digest(SOURCE) != KERNEL_SHA256:
        raise ValueError('equivalence-tested kernel source differs')
    compiler_version = subprocess.run([compiler, '--version'], check=True, capture_output=True,
                                      text=True, timeout=10).stdout
    signature = {'kernel_sha256': KERNEL_SHA256, 'compiler': compiler,
                 'compiler_sha256': digest(compiler), 'compiler_version': compiler_version,
                 'flags': list(FLAGS), 'platform': platform.system(), 'machine': platform.machine(),
                 'numpy_version': version('numpy'), 'driver_sha256': digest(__file__)}
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    library, receipt = directory/'kernel.so', directory/'build.yaml'
    if library.exists() or receipt.exists():
        if not (library.is_file() and receipt.is_file()):
            raise ValueError('partial compiled build retained; refusing silent overwrite')
        saved = yaml.safe_load(receipt.read_text())
        if any(saved.get(k) != v for k, v in signature.items()) or digest(library) != saved['binary_sha256']:
            raise ValueError('compiled build provenance changed; refusing rebuild')
        if saved.get('binary_path') != str(library.resolve()):
            raise ValueError('compiled receipt belongs to a different artifact location')
        return library, saved
    subprocess.run([compiler, *FLAGS, str(SOURCE), '-o', str(library)], check=True,
                   capture_output=True, timeout=60)
    if digest(SOURCE) != KERNEL_SHA256 or digest(compiler) != signature['compiler_sha256']:
        raise ValueError('kernel or compiler changed during build')
    signature.update(binary_sha256=digest(library), binary_path=str(library.resolve()))
    with receipt.open('x') as file:
        yaml.safe_dump(signature, file, sort_keys=False)
    return library, signature


@contextmanager
def compiled_driver(artifact_directory=None):
    """Retain exact binary if a sealed result directory is supplied; else temporary."""
    with tempfile.TemporaryDirectory(prefix='patchseq-verifier-build-') as temporary:
        library, provenance = prepare_build(artifact_directory or temporary)
        loaded = ctypes.CDLL(str(library))
        function = loaded.integrate_hh
        matrix = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS')
        vector = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
        function.argtypes = [ctypes.c_size_t, ctypes.c_size_t, matrix, matrix,
                             ctypes.c_double, vector, matrix]
        function.restype = ctypes.c_int

        def integrate(rows, current_pA, rate_hz, initial_mV):
            p = np.ascontiguousarray(parameters(rows))
            current = np.ascontiguousarray(current_pA, dtype=np.float64)
            initial = np.ascontiguousarray(initial_mV, dtype=np.float64)
            if (current.ndim != 2 or current.shape[1] != len(rows) or len(current) < 2
                or len(current) > 10000000 or current.size > 20000000
                or initial.shape != (len(rows),) or not np.isfinite(initial).all()
                or not np.isfinite(current).all() or not math.isfinite(rate_hz)
                or not 20000 < rate_hz <= 1000000):
                raise ValueError('invalid compiled driver inputs')
            out = np.empty(current.T.shape, dtype=np.float64, order='C')
            if function(len(current), len(rows), p, current, rate_hz, initial, out):
                raise ValueError('compiled driver numerical or input failure')
            return out

        def replay(rows, command_pA, rate_hz, holding_pA, warmup_s=1., refinement=1):
            command = np.asarray(command_pA, dtype=float)
            if (command.ndim != 1 or len(command) < 2 or not np.isfinite(command).all()
                or not math.isfinite(holding_pA) or not math.isfinite(warmup_s) or warmup_s <= 0
                or type(refinement) is not int or refinement not in (1, 2)):
                raise ValueError('invalid compiled replay inputs')
            rate = rate_hz*refinement
            if not math.isfinite(rate) or not 20000 < rate <= 1000000:
                raise ValueError('invalid compiled replay rate')
            count = round(warmup_s*rate)
            if abs(count-warmup_s*rate) > 1e-8:
                raise ValueError('nonintegral compiled warmup')
            parameters(rows)
            nominal = command+holding_pA
            samples = np.r_[np.full(count, nominal[0]), np.repeat(nominal, refinement)]
            current = np.broadcast_to(samples[:, None], (len(samples), len(rows)))
            voltage = integrate(rows, current, rate, [p['leak_reversal_mV'] for p in rows])
            return voltage[:, count::refinement]

        yield replay, provenance
        if digest(library) != provenance['binary_sha256'] or digest(SOURCE) != KERNEL_SHA256:
            raise ValueError('compiled artifact changed during verification context')
