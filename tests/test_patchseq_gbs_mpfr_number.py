"""Exact-bit arithmetic foundation only, not a neuron/solver benchmark."""

from __future__ import annotations

import ctypes
import hashlib
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from patchseq_mpfr_driver import checked_toolchain

ROOT = Path(__file__).parents[1]


@pytest.fixture(scope='module')
def numbers(tmp_path_factory):
    compiler = shutil.which('clang++')
    boost = next((p for p in map(Path, ['/opt/homebrew/include', '/usr/include', '/usr/local/include'])
                  if (p/'boost/numeric/odeint/stepper/bulirsch_stoer_dense_out.hpp').is_file()), None)
    if compiler is None or boost is None:
        pytest.skip('optional numerical prototype requires clang++ and Boost')
    try:
        prefix, pins = checked_toolchain()
    except RuntimeError as exc:
        pytest.skip(str(exc))
    sources = [ROOT/'scripts/patchseq_gbs_mpfr_number.hpp', ROOT/'tests/patchseq_gbs_mpfr_number_probe.cpp',
               ROOT/'scripts/patchseq_gbs_odeint_bridge.hpp']
    pins.update({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    flags = ['-std=c++17', '-O2', '-shared', '-fPIC', '-fno-fast-math', '-ffp-contract=off',
             '-I', str(ROOT/'scripts'), '-I', str(prefix/'include'), '-I', str(boost)]
    dependencies = subprocess.run([compiler, *flags, '-M', '-MT', 'gbs-number', str(sources[1])],
                                  check=True, capture_output=True, text=True, timeout=60).stdout
    for p in shlex.split(dependencies.partition(':')[2].replace('\\\n', ' ')):
        pins[str(Path(p).resolve())] = hashlib.sha256(Path(p).read_bytes()).hexdigest()
    pins[str(Path(compiler).resolve())] = hashlib.sha256(Path(compiler).resolve().read_bytes()).hexdigest()
    binary = tmp_path_factory.mktemp('gbs-number')/'probe.so'
    built = subprocess.run([compiler, *flags, str(sources[1]), str(prefix/'lib/libmpfr.a'),
                            str(prefix/'lib/libgmp.a'), '-lm', '-o', str(binary)],
                           capture_output=True, text=True, timeout=120, check=False)
    assert built.returncode == 0, built.stderr
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in pins.items())
    lib = ctypes.CDLL(str(binary))
    lib.gbs_number_eval.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
                                     ctypes.c_void_p, ctypes.c_size_t]
    lib.gbs_number_lift.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.c_void_p, ctypes.c_size_t]
    lib.gbs_number_restore.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t]
    lib.gbs_number_metadata.argtypes = [ctypes.c_int]
    lib.gbs_number_stepper_probe.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
    lib.gbs_number_binomial.argtypes = [ctypes.c_int, ctypes.c_uint, ctypes.c_uint,
                                      ctypes.c_void_p, ctypes.c_size_t]
    for name in ('eval', 'lift', 'restore', 'metadata', 'stepper_probe', 'binomial'):
        getattr(lib, 'gbs_number_'+name).restype = ctypes.c_int
    return lib


def result(numbers, bits, op, a='0', b='0'):
    output = ctypes.create_string_buffer(1024)
    code = numbers.gbs_number_eval(bits, op.encode(), str(a).encode(), str(b).encode(), output, len(output))
    assert code == 0, (code, bits, op, a, b)
    return output.value.decode()


def decode(text, mp):
    digits, exponent = text.split('p')
    return mp.mpf(int(digits, 16))*mp.power(2, int(exponent))


@pytest.mark.parametrize('bits', [160, 320])
def test_width_temporaries_traits_ownership_nan_and_gbs_construction(numbers, bits):
    assert numbers.gbs_number_metadata(bits) == bits


@pytest.mark.parametrize('bits', [160, 320])
@pytest.mark.parametrize('value', [0., -0., 1.28, 1.4, .128, .16, .5, 3.3, .02, 1e-100, 1e100])
def test_original_binary64_lift_exact_and_zero_sign_preserved(numbers, bits, value):
    mp = pytest.importorskip('mpmath').mp.clone()
    mp.prec = bits+100
    output = ctypes.create_string_buffer(1024)
    assert numbers.gbs_number_lift(bits, value, output, len(output)) == 0
    text = output.value.decode()
    n, d = value.as_integer_ratio()
    assert decode(text, mp) == mp.mpf(n)/d
    if value == 0:
        import math
        assert text == ('-0p0' if math.copysign(1., value) < 0 else '0p0')


@pytest.mark.parametrize('bits', [160, 320])
@pytest.mark.parametrize('op,a,b', [
    ('add', '1', '1e-40'), ('sub', '1.0000000000000000000000000000001', '1'),
    ('mul', '-1.28', '1e100'), ('div', '1', '7'), ('div', '-1e-100', '3.3'),
    ('compound', '.128', '3.3'), ('mixed', '.1', '7'), ('neg', '-1.28', '0'),
    ('abs', '-1e-80', '0'), ('epsilon', '0', '0'), ('uint64', '0', '0'), ('int64', '0', '0'),
])
def test_arithmetic_against_independently_rounded_mpmath(numbers, bits, op, a, b):
    rounded = pytest.importorskip('mpmath').mp.clone()
    rounded.prec = bits
    x, y = rounded.mpf(a), rounded.mpf(b)
    functions = {'add': lambda: x+y, 'sub': lambda: x-y, 'mul': lambda: x*y,
                 'div': lambda: x/y, 'compound': lambda: (((x+y)*y)-x)/y,
                 'mixed': lambda: (2*x+1)/(3+y), 'neg': lambda: -x, 'abs': lambda: abs(x),
                 'epsilon': lambda: rounded.power(2, 1-bits),
                 'uint64': lambda: rounded.mpf(2**64-1), 'int64': lambda: rounded.mpf(-2**63)}
    expected = functions[op]()
    wide = rounded.clone()
    wide.prec = bits+100
    assert decode(result(numbers, bits, op, a, b), wide) == wide.mpf(expected)


@pytest.mark.parametrize('bits', [160, 320])
@pytest.mark.parametrize('op,a,b', [('sqrt', '2', '0'), ('exp', '-100', '0'),
                                   ('exp', '.02', '0'), ('expm1', '1e-40', '0'),
                                   ('log', '3.3', '0'), ('log10', '7', '0'),
                                   ('pow', '.02', '.33333333333333333333333333333333')])
def test_transcendentals_agree_with_wider_independent_reference(numbers, bits, op, a, b):
    mp = pytest.importorskip('mpmath').mp.clone()
    mp.prec = bits
    # Inputs are rounded to the target width before evaluating the wider oracle.
    x, y = mp.mpf(a), mp.mpf(b)
    mp.prec = bits+150
    functions = {'sqrt': lambda: mp.sqrt(x), 'exp': lambda: mp.exp(x),
                 'expm1': lambda: mp.expm1(x), 'log': lambda: mp.log(x),
                 'log10': lambda: mp.log10(x), 'pow': lambda: mp.power(x, y)}
    expected = functions[op]()
    actual = decode(result(numbers, bits, op, a, b), mp)
    assert abs(actual-expected) <= abs(expected)*mp.power(2, 1-bits)


@pytest.mark.parametrize('bits', [160, 320])
def test_full_internal_roundtrip_preserves_digits_not_in_exported_double(numbers, bits):
    text = result(numbers, bits, 'add', '1', '1e-40')
    output = ctypes.create_string_buffer(1024)
    assert numbers.gbs_number_restore(bits, text.encode(), output, len(output)) == 0
    assert output.value.decode() == text
    assert text != result(numbers, bits, 'add', '1', '0')
    for zero in (b'0p0', b'-0p0'):
        assert numbers.gbs_number_restore(bits, zero, output, len(output)) == 0
        assert output.value == zero


@pytest.mark.parametrize('bits', [160, 320])
@pytest.mark.parametrize('bad', ['', '1', '1p', '1p0', '+1p0', '1p+0', 'NaN', 'inf',
                                '0p1', '00p0', '1p0 trailing', '1p9999999999999999999999999'])
def test_malformed_noncanonical_or_inexact_state_is_rejected(numbers, bits, bad):
    output = ctypes.create_string_buffer(1024)
    assert numbers.gbs_number_restore(bits, bad.encode(), output, len(output)) != 0


@pytest.mark.parametrize('bits', [53, 161, 321])
def test_unsupported_width_rejected(numbers, bits):
    assert numbers.gbs_number_metadata(bits) == 0


def test_cross_width_state_not_silently_imported(numbers):
    text = result(numbers, 160, 'div', '1', '7')
    output = ctypes.create_string_buffer(1024)
    assert numbers.gbs_number_restore(320, text.encode(), output, len(output)) != 0


@pytest.mark.parametrize('bits', [160, 320])
def test_arithmetic_usable_in_actual_dense_stepper_without_neuron(numbers, bits):
    mp = pytest.importorskip('mpmath').mp.clone()
    mp.prec = bits+100
    output = ctypes.create_string_buffer(1024)
    assert numbers.gbs_number_stepper_probe(bits, output, len(output)) == 0
    endpoint, midpoint, time = [decode(x, mp) for x in output.value.decode().split('|')]
    assert time > 0
    assert abs(endpoint-mp.exp(-time)) < mp.mpf('1e-18')
    assert abs(midpoint-mp.exp(-time/2)) < mp.mpf('1e-18')


@pytest.mark.parametrize('bits', [160, 320])
def test_exact_small_binomial_bridge_and_bounds(numbers, bits):
    import math

    mp = pytest.importorskip('mpmath').mp.clone()
    mp.prec = bits
    output = ctypes.create_string_buffer(1024)
    for n in range(33):
        for k in range(n+1):
            assert numbers.gbs_number_binomial(bits, n, k, output, len(output)) == 0
            assert decode(output.value.decode(), mp) == math.comb(n, k)
    for n, k in [(33, 1), (3, 4), (2**32-1, 0)]:
        assert numbers.gbs_number_binomial(bits, n, k, output, len(output)) != 0


@pytest.mark.parametrize('bits', [160, 320])
def test_round_to_nearest_ties_to_even_at_actual_width(numbers, bits):
    from decimal import Decimal, localcontext

    mp = pytest.importorskip('mpmath').mp.clone()
    mp.prec = bits+100
    with localcontext() as decimal:
        decimal.prec = 500
        half = Decimal(2)**(-bits)
        even_low = result(numbers, bits, 'add', '1', format(half, 'f'))
        even_high = result(numbers, bits, 'add', '1', format(3*half, 'f'))
    assert decode(even_low, mp) == 1
    assert decode(even_high, mp) == 1+mp.power(2, 2-bits)


@pytest.mark.parametrize('op,a,b', [('div', '1', '0'), ('sqrt', '-1', '0'),
                                   ('exp', 'inf', '0'), ('add', 'nan', '1')])
def test_nonfinite_results_are_not_clipped_into_finite_checkpoints(numbers, op, a, b):
    output = ctypes.create_string_buffer(1024)
    assert numbers.gbs_number_eval(160, op.encode(), a.encode(), b.encode(), output, len(output)) != 0
