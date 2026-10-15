"""Synthetic-only tests; never load a training or held-out recording."""

from __future__ import annotations

import ctypes
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope='module')
def precision_kernel(tmp_path_factory):
    compiler = shutil.which('clang++') or shutil.which('g++')
    roots = [Path('/opt/homebrew/include'), Path('/usr/include'), Path('/usr/local/include')]
    include = next((p for p in roots if (p/'boost/multiprecision/cpp_bin_float.hpp').is_file()), None)
    if not compiler or include is None:
        pytest.skip('optional diagnostic needs a C++ compiler and Boost headers')
    source = Path(__file__).resolve().parents[1]/'scripts/patchseq_hh_precision_kernel.cpp'
    binary = tmp_path_factory.mktemp('precision-prototype')/'kernel.so'
    subprocess.run([compiler, '-std=c++17', '-O2', '-shared', '-fPIC', '-fno-fast-math',
                    '-ffp-contract=off', '-I', str(include), str(source), '-o', str(binary)],
                   check=True, capture_output=True, timeout=120)
    lib = ctypes.CDLL(str(binary))
    pointer = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
    lib.precision_integrate.argtypes = [ctypes.c_int, ctypes.c_size_t, pointer, pointer,
                                       ctypes.c_double, ctypes.c_int, pointer, pointer]
    lib.precision_integrate.restype = ctypes.c_int
    lib.precision_digits.argtypes = [ctypes.c_int]
    lib.precision_digits.restype = ctypes.c_int
    lib.precision_probe.argtypes = [ctypes.c_int]
    lib.precision_probe.restype = ctypes.c_double
    return lib


def integrate(lib, bits, parameters, current, initial, factor=1):
    states = np.full((len(current), 5), np.nan)
    code = lib.precision_integrate(bits, len(current), np.array(parameters, dtype=float),
                                   np.array(current, dtype=float), 50000., factor,
                                   np.array(initial, dtype=float), states)
    return code, states


def test_actual_precision(precision_kernel):
    assert [precision_kernel.precision_digits(p) for p in (53, 160, 320)] == [53, 160, 320]
    assert precision_kernel.precision_probe(53) == 0
    assert precision_kernel.precision_probe(160) == 2.**-100
    assert precision_kernel.precision_probe(320) == 2.**-100


@pytest.mark.parametrize('bits', [53, 160, 320])
def test_passive_current_edges_and_shared_initial_state(precision_kernel, bits):
    p = [100., 10., -65., 0., 0., 0., -60., 1000.]
    initial = [-65., .1, .2, .3, .4]
    current = np.r_[np.zeros(30), np.full(30, 100.), np.zeros(31)]
    code, states = integrate(precision_kernel, bits, p, current, initial, 4)
    assert code == 0
    np.testing.assert_array_equal(states[0], initial)
    # Output is before each update; the current at sample 30 first affects sample 31.
    np.testing.assert_array_equal(states[:31, 0], -65.)
    expected = [-65.]
    for value in current[:-1]:
        steady = -65.+value/10.
        expected.append(steady+(expected[-1]-steady)*np.exp(-.02/10.))
    np.testing.assert_allclose(states[:, 0], expected, atol=2e-12, rtol=0)


def oracle(p, current, initial, factor):
    """Independent Python RK4 at 150 decimal digits, lifting exact binary64 inputs."""
    mp = pytest.importorskip('mpmath').mp.clone()
    mp.dps = 150

    def lift(value):
        numerator, denominator = float(value).as_integer_ratio()
        return mp.mpf(numerator)/denominator

    p = list(map(lift, p))
    state = list(map(lift, initial))
    step = lift(1000./(50000.*factor))

    def rhs(y, injected):
        v, m, h, n, slow = y
        u = v-p[6]

        def trap(x):
            return x/mp.expm1(x) if x else mp.mpf(1)

        am = lift(1.28)*trap((13-u)/4)
        bm = lift(1.4)*trap((u-40)/5)
        ah = lift(.128)*mp.exp((17-u)/18)
        bh = 4/(1+mp.exp((40-u)/5))
        an = lift(.16)*trap((15-u)/5)
        bn = lift(.5)*mp.exp((10-u)/40)
        inf = 1/(1+mp.exp(-(v+35)/10))
        tau = p[7]/(lift(3.3)*mp.exp((v+35)/20)+mp.exp(-(v+35)/20))
        inward = (p[1]*(p[2]-v)+p[3]*m**3*h*(50-v)
                  +p[4]*n**4*(-90-v)+p[5]*slow*(-90-v)+injected)
        return [inward/p[0], am*(1-m)-bm*m, ah*(1-h)-bh*h,
                an*(1-n)-bn*n, (inf-slow)/tau]

    output = [list(map(float, state))]
    for current_value in current[:-1]:
        injected = lift(current_value)
        for _ in range(factor):
            k1 = rhs(state, injected)
            k2 = rhs([y+step*k/2 for y, k in zip(state, k1, strict=True)], injected)
            k3 = rhs([y+step*k/2 for y, k in zip(state, k2, strict=True)], injected)
            k4 = rhs([y+step*k for y, k in zip(state, k3, strict=True)], injected)
            state = [y+step*(a+2*b+2*c+d)/6 for y, a, b, c, d in
                     zip(state, k1, k2, k3, k4, strict=True)]
        output.append(list(map(float, state)))
    return np.array(output)


@pytest.mark.parametrize('bits', [53, 160, 320])
@pytest.mark.parametrize('factor', [1, 4])
def test_active_against_independent_multiprecision_rk4(precision_kernel, bits, factor):
    p = [100., 10., -65., 2000., 500., 10., -60., 1000.]
    initial = [-65., .05, .6, .2, .05]
    current = np.r_[np.full(10, -50.), np.full(10, 125.), np.zeros(11)]
    code, states = integrate(precision_kernel, bits, p, current, initial, factor)
    assert code == 0
    expected = oracle(p, current, initial, factor)
    np.testing.assert_allclose(states, expected, atol=2e-13 if bits == 53 else 0, rtol=0)


@pytest.mark.parametrize('defect', ['precision', 'factor', 'nan', 'capacitance', 'gate'])
def test_rejects_invalid_inputs_without_clipping(precision_kernel, defect):
    p = [100., 10., -65., 2000., 500., 10., -60., 1000.]
    current, initial = [0., 0.], [-65., .05, .6, .2, .05]
    bits, factor = 160, 1
    if defect == 'precision':
        bits = 80
    elif defect == 'factor':
        factor = 3
    elif defect == 'nan':
        current[-1] = np.nan
    elif defect == 'capacitance':
        p[0] = 0
    else:
        initial[1] = 1.5
    code, _ = integrate(precision_kernel, bits, p, current, initial, factor)
    assert code != 0
