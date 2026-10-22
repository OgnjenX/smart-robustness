"""Independent fixed-native-grid RKF78 preparation, not a fit or qualification runner.

The raw MPFR stepper and RHS do not use the primary Workspace, GBS engine or
arithmetic adapter. GMP/MPFR and nonnumerical build/input helpers remain shared.
No adaptive retries, interpolation, recording loader or automatic refinement.
"""

from __future__ import annotations

import ctypes
import json
import math
import platform
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
from patchseq_gbs_driver import FLAGS, _array, canonical, digest, identity, verify_pins
from patchseq_mpfr_driver import checked_toolchain

SCRIPTS = Path(__file__).resolve().parent
SOURCE = SCRIPTS/'patchseq_rkf78_reference.cpp'
FIELDS = ('capacitance_pF', 'leak_nS', 'leak_reversal_mV', 'sodium_nS', 'potassium_nS',
          'm_current_nS', 'threshold_shift_mV', 'm_tau_max_ms')


def parameters_array(parameters):
    if (type(parameters) is not dict or set(parameters) != set(FIELDS)
        or any(isinstance(value, (bool, np.bool_)) for value in parameters.values())):
        raise ValueError('exact independent HH parameter fields required')
    result = _array([parameters[field] for field in FIELDS], (8,))
    if (result[0] <= 0 or result[1] <= 0 or result[7] <= 0 or any(result[j] < 0 for j in (3, 4, 5))):
        raise ValueError('invalid independent HH parameters')
    return result


def prepare_build(directory):
    prefix, prerequisite_pins = checked_toolchain()
    compiler = shutil.which('clang++')
    if not compiler:
        raise RuntimeError('optional independent reference requires clang++')
    compiler = str(Path(compiler).resolve())
    flags = [*FLAGS, '-I', str(prefix/'include')]
    deps = subprocess.run([compiler, *flags, '-M', '-MT', 'rkf78-reference', str(SOURCE)],
                          check=True, capture_output=True, text=True, timeout=60).stdout
    paths = [Path(p).resolve() for p in shlex.split(deps.partition(':')[2].replace('\\\n', ' '))]
    paths += [Path(__file__).resolve(), Path(compiler), SCRIPTS/'patchseq_gbs_driver.py',
              SCRIPTS/'patchseq_mpfr_driver.py', SCRIPTS/'third_party/BOOST_LICENSE_1_0.txt']
    libraries = [str(prefix/'lib/libmpfr.a'), str(prefix/'lib/libgmp.a')]
    pins = {str(Path(p).resolve()): h for p, h in prerequisite_pins.items()}
    pins.update({str(p): digest(p) for p in paths})
    pins.update({p: digest(p) for p in libraries})
    configuration = {'source_hashes': pins, 'flags': flags, 'compiler': compiler,
        'compiler_version': subprocess.run([compiler, '--version'], check=True, capture_output=True,
                                            text=True, timeout=10).stdout,
        'static_libraries': libraries, 'platform': platform.platform(), 'machine': platform.machine()}
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    binary, receipt, lock = (directory/name for name in ('reference.so', 'build.json', 'build.lock'))
    if lock.exists():
        raise ValueError('active or orphan independent build lock; refusing overwrite')
    if binary.exists() or receipt.exists():
        if not binary.is_file() or not receipt.is_file():
            raise ValueError('partial independent build retained; refusing overwrite')
        saved = json.loads(receipt.read_text())
        if (set(saved) != set(configuration) | {'binary_path', 'binary_sha256'}
            or any(saved.get(k) != v for k, v in configuration.items())
            or saved['binary_path'] != str(binary) or saved['binary_sha256'] != digest(binary)):
            raise ValueError('independent build provenance differs')
        verify_pins(pins)
        return binary, saved
    with lock.open('x') as file:
        file.write('independent-raw-MPFR-RKF78-build\n')
    try:
        if binary.exists() or receipt.exists():
            raise ValueError('independent build appeared while acquiring lock')
        subprocess.run([compiler, *flags, str(SOURCE), *libraries, '-lm', '-o', str(binary)],
                       check=True, capture_output=True, text=True, timeout=120)
        verify_pins(pins)
        configuration.update(binary_path=str(binary), binary_sha256=digest(binary))
        with receipt.open('x') as file:
            file.write(canonical(configuration)+'\n')
    finally:
        lock.unlink()
    return binary, configuration


@dataclass(frozen=True)
class ReferenceResult:
    states: np.ndarray
    rhs_evaluations: int
    accepted_substeps: int
    maximum_embedded_estimate: np.ndarray
    final_exact: tuple[str, ...]


class ReferenceFailure(ValueError):
    def __init__(self, message, counts):
        super().__init__(message)
        self.rhs_evaluations, self.accepted_substeps = map(int, counts)


def exact_export(text, bits):
    """Independent validation/export of the retained integer-significand encoding."""
    if type(bits) is not int or bits not in (160, 320) or type(text) is not str or len(text) > bits//4+32:
        raise ValueError('invalid independent exact encoding')
    if text in ('0p0', '-0p0'):
        return -0.0 if text[0] == '-' else 0.0
    fields = text.split('p')
    if len(fields) != 2:
        raise ValueError('invalid independent exact encoding')
    digits, exponent = fields
    magnitude = digits.removeprefix('-')
    if (len(magnitude) != bits//4 or any(c not in '0123456789abcdef' for c in magnitude)
        or int(magnitude, 16).bit_length() != bits or str(int(exponent)) != exponent
        or not -1073741823-bits <= int(exponent) <= 1073741823):
        raise ValueError('noncanonical independent exact state')
    return float.fromhex(('-' if digits.startswith('-') else '')+'0x'+magnitude+'p'+exponent)


class RKF78Driver:
    def __init__(self, directory):
        binary, receipt = prepare_build(directory)
        self._receipt = canonical(receipt)
        self._lib = ctypes.CDLL(str(binary))
        vector = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
        uints = np.ctypeslib.ndpointer(dtype=np.uint64, flags='C_CONTIGUOUS')
        error = [ctypes.c_void_p, ctypes.c_size_t]
        self._lib.rkf78_integrate.argtypes = [ctypes.c_int, ctypes.c_size_t, vector, vector,
            ctypes.c_double, ctypes.c_int, vector, ctypes.c_uint64, ctypes.c_uint64, vector,
            uints, vector, *error, *error]
        self._lib.rkf78_integrate.restype = ctypes.c_int
        self._lib.rkf78_rhs_exact.argtypes = [ctypes.c_int, vector, vector, ctypes.c_double, *error, *error]
        self._lib.rkf78_rhs_exact.restype = ctypes.c_int
        self._lib.rkf78_table.argtypes = []
        self._lib.rkf78_table.restype = ctypes.c_char_p
        self._lib.rkf78_precision.argtypes = [ctypes.c_int]
        self._lib.rkf78_precision.restype = ctypes.c_int
        for name in ('mpfr_version', 'header_version', 'gmp_version'):
            function = getattr(self._lib, 'rkf78_'+name)
            function.argtypes, function.restype = [], ctypes.c_char_p
        self._lib.rkf78_environment.argtypes, self._lib.rkf78_environment.restype = [], ctypes.c_int
        environment = {'mpfr': self._lib.rkf78_mpfr_version().decode(),
            'header': self._lib.rkf78_header_version().decode(), 'gmp': self._lib.rkf78_gmp_version().decode(),
            'range_and_tls': self._lib.rkf78_environment(),
            'precision160': self._lib.rkf78_precision(160), 'precision320': self._lib.rkf78_precision(320)}
        if environment != {'mpfr': '4.2.2', 'header': '4.2.2', 'gmp': '6.3.0',
                                'range_and_tls': 1, 'precision160': 160, 'precision320': 320}:
            raise ValueError('independent arithmetic version, range or precision probe failed')
        self._environment = canonical(environment)
        table = json.loads(self._lib.rkf78_table())
        if set(table) != {'a', 'b', 'db', 'c'} or any(len(table[k]) != 13 for k in table):
            raise ValueError('independent tableau support differs')
        for i, row in enumerate(table['a']):
            if len(row) != i or sum(map(Fraction, row)) != Fraction(table['c'][i]):
                raise ValueError('independent rational tableau row sums differ')
        if sum(map(Fraction, table['b'])) != 1 or sum(map(Fraction, table['db'])) != 0:
            raise ValueError('independent weight sums differ')
        self._table = canonical(table)

    @property
    def environment(self):
        return json.loads(self._environment)

    @property
    def provenance(self):
        return json.loads(self._receipt)

    @property
    def table(self):
        return json.loads(self._table)

    @property
    def table_sha256(self):
        return identity(self.table)

    def verify(self):
        receipt = self.provenance
        verify_pins(receipt['source_hashes'])
        if digest(receipt['binary_path']) != receipt['binary_sha256']:
            raise ValueError('independent loaded binary changed')
        if canonical(json.loads(self._lib.rkf78_table())) != self._table:
            raise ValueError('independent runtime tableau changed')

    def derivative_exact(self, parameters, state, current, bits):
        self.verify()
        if (type(bits) is not int or bits not in (160, 320) or isinstance(current, bool)
            or not isinstance(current, (float, int)) or not math.isfinite(current)
            or (type(current) is int and abs(current) > 2**53)):
            raise ValueError('invalid independent derivative probe')
        p, state = parameters_array(parameters), _array(state, (5,))
        output, error = ctypes.create_string_buffer(4096), ctypes.create_string_buffer(4096)
        code = self._lib.rkf78_rhs_exact(bits, p, state, float(current), output, len(output), error, len(error))
        if code:
            raise ValueError(error.value.decode())
        self.verify()
        return tuple(output.value.decode().split('|'))

    def integrate(self, parameters, current, rate, initial, bits, factor, *,
                  maximum_rhs=1_000_000, maximum_substeps=100_000):
        self.verify()
        if (type(bits) is not int or bits not in (160, 320) or type(factor) is not int
            or factor not in (1, 2, 4, 8, 16) or isinstance(rate, (bool, np.bool_))
            or not isinstance(rate, (float, int, np.floating))
            or (isinstance(rate, np.floating) and rate.dtype.itemsize > 8)
            or not math.isfinite(rate) or not 20000 < rate <= 1000000):
            raise ValueError('invalid independent fixed-grid specification')
        for budget in (maximum_rhs, maximum_substeps):
            if type(budget) is not int or not 0 <= budget <= 2**63-1:
                raise ValueError('invalid independent evaluation/substep budget')
        raw = np.asarray(current)
        if raw.ndim != 1 or not 2 <= len(raw) <= 1000000:
            raise ValueError('invalid independent current support')
        current, initial = _array(raw, raw.shape), _array(initial, (5,))
        p = parameters_array(parameters)
        states = np.full((len(current), 5), np.nan, dtype=np.float64)
        counts = np.zeros(2, dtype=np.uint64)
        embedded = np.full(5, np.nan, dtype=np.float64)
        final, error = ctypes.create_string_buffer(4096), ctypes.create_string_buffer(4096)
        code = self._lib.rkf78_integrate(bits, len(current), p, current, float(rate), factor, initial,
            maximum_rhs, maximum_substeps, states, counts, embedded, final, len(final), error, len(error))
        self.verify()
        if code:
            raise ReferenceFailure(error.value.decode(), counts)
        if (not np.isfinite(states).all() or not np.isfinite(embedded).all() or (embedded < 0).any()
            or tuple(map(int, counts)) != ((len(current)-1)*factor*13, (len(current)-1)*factor)):
            raise ValueError('independent incomplete output or work accounting')
        exact = tuple(final.value.decode().split('|'))
        if len(exact) != 5:
            raise ValueError('independent final state support differs')
        exported = np.array([exact_export(text, bits) for text in exact], dtype=np.float64)
        if not np.array_equal(exported.view(np.uint64), states[-1].view(np.uint64)):
            raise ValueError('independent exact final state and binary64 observation differ')
        return ReferenceResult(states, int(counts[0]), int(counts[1]), embedded, exact)
