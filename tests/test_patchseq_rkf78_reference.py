"""Independent rational method and separately expressed RHS, short units only."""

from __future__ import annotations

import ast
import hashlib
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from patchseq_rkf78_reference import ReferenceFailure, RKF78Driver, exact_export, prepare_build
from test_patchseq_gbs_hh import INITIAL, decode, parameters, rhs_oracle

ROOT = Path(__file__).parents[1]


@pytest.fixture(scope='module')
def reference(tmp_path_factory):
    try:
        return RKF78Driver(tmp_path_factory.mktemp('rkf78-independent-units'))
    except RuntimeError as exc:
        pytest.skip(str(exc))
    except subprocess.CalledProcessError as exc:
        pytest.fail(str(exc.stderr))


def fraction_ast(expression):
    """Parse only the pinned upstream integer/rational coefficient grammar."""
    text = re.sub(r'static_cast<\s*Value\s*>\(\s*(-?\d+)\s*\)', r'(\1)', expression)
    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return Fraction(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -visit(node.operand)
        if isinstance(node, ast.BinOp):
            a, b = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Div):
                return a/b
            if isinstance(node.op, ast.Sub):
                return a-b
        raise AssertionError('unexpected upstream coefficient expression: '+expression)
    return visit(ast.parse(text.strip(), mode='eval').body)


def test_every_rational_table_entry_matches_pinned_method_source(reference):
    header = next((p/'boost/numeric/odeint/stepper/runge_kutta_fehlberg78.hpp'
        for p in map(Path, ['/opt/homebrew/include', '/usr/include', '/usr/local/include'])
        if (p/'boost/numeric/odeint/stepper/runge_kutta_fehlberg78.hpp').is_file()), None)
    if header is None:
        pytest.skip('optional coefficient-source comparison needs the pinned header, not at runtime')
    assert hashlib.sha256(header.read_bytes()).hexdigest() == (
        'a9d39665fa613553ae0aff3b0836650141eb91c48f683b3f4805f7d8bca4ce8a')
    text, table = header.read_text(), reference.table
    for key in [*(f'a{i}' for i in range(1, 13)), 'b', 'db', 'c']:
        block = re.search(r'struct rk78_coefficients_'+key+r'\b.*?\n\};', text, re.DOTALL).group()
        assignments = re.findall(r'\(\*this\)\[(\d+)\]\s*=\s*([^;]+);', block)
        expected = [fraction_ast(value) for index, value in assignments]
        actual = table['a'][int(key[1:])] if key.startswith('a') else table[key]
        assert list(map(Fraction, actual)) == expected


def test_numerical_source_and_build_closure_do_not_call_primary_engine(reference):
    source = (ROOT/'scripts/patchseq_rkf78_reference.cpp').read_text()
    includes = re.findall(r'^#include\s+[<"]([^>"]+)[>"]', source, re.MULTILINE)
    assert all('patchseq' not in name and 'boost' not in name for name in includes)
    assert 'mpfr_pow_ui' in source
    closure = reference.provenance['source_hashes']
    for name in ('patchseq_mpfr_kernel.cpp', 'patchseq_gbs_hh.cpp', 'patchseq_gbs_core.hpp',
                 'patchseq_gbs_guarded_core.hpp', 'patchseq_gbs_mpfr_number.hpp'):
        assert str((ROOT/'scripts'/name).resolve()) not in closure
    assert reference.environment['precision160'] == 160
    assert reference.environment['precision320'] == 320


@pytest.mark.parametrize('bits', [160, 320])
@pytest.mark.parametrize('voltage', [-200., -65., -47., -45., -20., 40., 400.])
def test_independent_rhs_against_high_precision_equation_oracle(reference, bits, voltage):
    mp = pytest.importorskip('mpmath').mp.clone()
    mp.dps = 150
    state = [voltage, .05, .6, .2, .05]
    actual = reference.derivative_exact(parameters(), state, 125., bits)
    expected = rhs_oracle(parameters(), state, 125., mp)
    for text, value in zip(actual, expected, strict=True):
        assert abs(decode(text, mp)-value) <= max(1, abs(value))*mp.power(2, 12-bits)


@pytest.mark.parametrize('bits', [160, 320])
@pytest.mark.parametrize('factor', [1, 2, 4])
def test_passive_closed_form_whole_native_array_and_exact_work_count(reference, bits, factor):
    current = np.r_[np.zeros(5), np.full(7, -50.), np.full(11, 125.), np.zeros(8)]
    actual = reference.integrate(parameters(passive=True), current, 50000., INITIAL, bits, factor)
    mp = pytest.importorskip('mpmath').mp.clone()
    mp.dps = 150
    voltage, expected, h = mp.mpf(-65), [-65.], mp.mpf(1000./50000.)
    for command in current[:-1]:
        steady = mp.mpf(-65)+mp.mpf(float(command))/10
        voltage = steady+(voltage-steady)*mp.exp(-h/10)
        expected.append(float(voltage))
    np.testing.assert_allclose(actual.states[:, 0], expected, atol=2e-12, rtol=0)
    np.testing.assert_array_equal(actual.states[0], INITIAL)
    assert actual.rhs_evaluations == (len(current)-1)*factor*13
    assert actual.accepted_substeps == (len(current)-1)*factor
    assert len(actual.final_exact) == 5
    assert np.all(actual.maximum_embedded_estimate >= 0)


@pytest.mark.parametrize('bits', [160, 320])
def test_observed_high_order_convergence_against_exact_fast_passive_endpoint(reference, bits):
    p = parameters(passive=True)
    p['capacitance_pF'], p['leak_nS'] = 1., 100.
    initial, current = [-64., .05, .6, .2, .05], np.zeros(2)
    mp = pytest.importorskip('mpmath').mp.clone()
    mp.prec = 600
    expected = mp.mpf(-65)+mp.exp(-100*mp.mpf(1000./50000.))
    errors = []
    for factor in (1, 2, 4):
        result = reference.integrate(p, current, 50000., initial, bits, factor)
        errors.append(abs(decode(result.final_exact[0], mp)-expected))
    assert errors[0] > errors[1] > errors[2]
    assert errors[0]/errors[1] > 100
    assert errors[1]/errors[2] > 100


@pytest.mark.parametrize('bits', [160, 320])
def test_active_fixed_grid_refinement_and_full_precision_exact_repeats(reference, bits):
    current = np.full(101, 2000.)
    results = [reference.integrate(parameters(), current, 50000., INITIAL, bits, factor)
               for factor in (1, 2, 4)]
    np.testing.assert_allclose(results[1].states, results[2].states, atol=1e-8, rtol=0)
    repeated = reference.integrate(parameters(), current, 50000., INITIAL, bits, 4)
    np.testing.assert_array_equal(repeated.states.view(np.uint64), results[2].states.view(np.uint64))
    assert repeated.final_exact == results[2].final_exact
    # Coarsest arm is retained as a control, not silently deleted or substituted.
    assert results[0].rhs_evaluations == 13*100


@pytest.mark.parametrize('which', ['rhs', 'steps'])
def test_failure_retains_executed_work_and_never_returns_partial_success(reference, which):
    kwargs = {'maximum_rhs': 1} if which == 'rhs' else {'maximum_substeps': 0}
    with pytest.raises(ReferenceFailure, match='budget') as failure:
        reference.integrate(parameters(), np.zeros(4), 50000., INITIAL, 160, 2, **kwargs)
    assert failure.value.accepted_substeps == 0
    assert failure.value.rhs_evaluations == (1 if which == 'rhs' else 0)


def test_invalid_accepted_gate_and_nonfinite_dynamics_are_not_clipped(reference):
    with pytest.raises(ReferenceFailure, match='gate bound'):
        reference.integrate(parameters(), np.zeros(3), 50000., [-65., 1.5, .6, .2, .05], 160, 1)
    with pytest.raises(ReferenceFailure, match='nonfinite'):
        reference.integrate(parameters(), np.zeros(3), 50000., [1e308, .05, .6, .2, .05], 160, 1)


def test_final_command_has_no_interval_and_threaded_calls_are_independent(reference):
    def run(bits):
        return reference.integrate(parameters(), [0., -50., 125., 0.], 50000., INITIAL, bits, 2)
    serial = [run(160), run(320)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        concurrent = list(pool.map(run, (160, 320)))
    for a, b in zip(serial, concurrent, strict=True):
        np.testing.assert_array_equal(a.states.view(np.uint64), b.states.view(np.uint64))
        assert a.final_exact == b.final_exact
    terminal = reference.integrate(parameters(), [0., -50., 125., 1e308], 50000., INITIAL, 160, 2)
    np.testing.assert_array_equal(terminal.states.view(np.uint64), serial[0].states.view(np.uint64))


def test_exact_build_reuse_and_detached_provenance(reference):
    receipt = reference.provenance
    binary, reused = prepare_build(Path(receipt['binary_path']).parent)
    assert reused == receipt and str(binary) == receipt['binary_path']
    receipt['source_hashes'].clear()
    reference.table['b'].clear()
    reference.environment['precision160'] = 53
    assert reference.provenance['source_hashes'] and len(reference.table['b']) == 13
    assert reference.environment['precision160'] == 160
    reference.verify()


@pytest.mark.parametrize('field,value', [('bits', 53), ('bits', True), ('factor', 3), ('factor', True),
    ('rate', 20000.), ('rate', True), ('rate', float('nan')), ('maximum_rhs', -1),
    ('maximum_substeps', True), ('maximum_rhs', 2**63), ('current', [1.]),
    ('current', [[0., 0.]]), ('current', [0., float('inf')]), ('initial', INITIAL[:4])])
def test_invalid_independent_replay_specification_not_reinterpreted(reference, field, value):
    kwargs = {'parameters': parameters(), 'current': [0., 0.], 'rate': 50000.,
              'initial': INITIAL, 'bits': 160, 'factor': 1, field: value}
    with pytest.raises(ValueError):
        reference.integrate(**kwargs)


@pytest.mark.parametrize('defect', ['missing', 'extra', 'capacitance', 'negative_sodium', 'boolean'])
def test_independent_parameters_require_exact_original_fields(reference, defect):
    p = parameters()
    if defect == 'missing':
        del p['leak_nS']
    elif defect == 'extra':
        p['extra'] = 1
    elif defect == 'capacitance':
        p['capacitance_pF'] = 0
    elif defect == 'negative_sodium':
        p['sodium_nS'] = -1
    else:
        p['capacitance_pF'] = True
    with pytest.raises(ValueError):
        reference.integrate(p, [0., 0.], 50000., INITIAL, 160, 1)


@pytest.mark.parametrize('text', ['', 'nan', 'inf', '1p0', '0p1', '+0p0', '1pp0', '8'*40+'p+0',
    '8'*40+'p00', '0'*40+'p0', 'A'*40+'p0', '8'*40+'p9999999999999999'])
def test_noncanonical_exact_state_rejected_without_rounding_or_clipping(text):
    with pytest.raises(ValueError):
        exact_export(text, 160)


@pytest.mark.parametrize('bits', [160, 320])
def test_independent_exact_state_zero_sign_and_binary64_consistency(bits):
    assert np.signbit(exact_export('-0p0', bits))
    assert not np.signbit(exact_export('0p0', bits))
    text = '8'+'0'*(bits//4-1)+'p'+str(1-bits)
    assert exact_export(text, bits) == 1.
    with pytest.raises(ValueError):
        exact_export(text, 53)
