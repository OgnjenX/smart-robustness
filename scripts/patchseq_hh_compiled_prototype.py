"""Optional synthetic-only compiled verifier prototype; never selected by runners."""

from __future__ import annotations

import ctypes
import math
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from patchseq_hh_independent_replay import parameters

SOURCE = Path(__file__).with_name('patchseq_hh_replay_kernel.cpp')
FLAGS = ('-std=c++17', '-O2', '-fPIC', '-shared', '-fno-fast-math', '-ffp-contract=off')


@contextmanager
def compiled_engine():
    compiler = shutil.which('clang++')
    if compiler is None:
        raise RuntimeError('optional compiled prototype requires clang++')
    with tempfile.TemporaryDirectory(prefix='patchseq-hh-kernel-') as directory:
        library = Path(directory)/'kernel.so'
        subprocess.run([compiler, *FLAGS, str(SOURCE), '-o', str(library)], check=True,
                       capture_output=True, timeout=60)
        loaded = ctypes.CDLL(str(library))
        function = loaded.integrate_hh
        pointer = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS')
        vector = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
        function.argtypes = [ctypes.c_size_t, ctypes.c_size_t, pointer, pointer,
                             ctypes.c_double, vector, pointer]
        function.restype = ctypes.c_int

        def integrate(rows, sampled_current_pA, rate_hz, initial_mV):
            p = np.ascontiguousarray(parameters(rows))
            current = np.ascontiguousarray(sampled_current_pA, dtype=np.float64)
            initial = np.ascontiguousarray(initial_mV, dtype=np.float64)
            if (current.ndim != 2 or current.shape[1] != len(rows) or len(current) < 2
                or len(current) > 10000000 or current.size > 20000000
                or initial.shape != (len(rows),) or not np.isfinite(initial).all()
                or not np.isfinite(current).all() or not math.isfinite(rate_hz)
                or not 20000 < rate_hz <= 1000000):
                raise ValueError('invalid compiled synthetic inputs')
            out = np.empty(current.T.shape, dtype=np.float64, order='C')
            status = function(len(current), len(rows), p, current, rate_hz, initial, out)
            if status:
                raise ValueError('compiled synthetic input or numerical failure: '+str(status))
            return out

        yield integrate
