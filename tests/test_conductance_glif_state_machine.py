"""Independent analytical controls, before any acquired-cell outcomes."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from smart_robustness.validation.conductance_glif_state_machine import simulate_conductance_glif
from smart_robustness.validation.glif_conductance_input import (
    effective_current,
    run_extended_native,
    validate_inputs,
)
from smart_robustness.validation.native_glif_parity import exact_repeat, fixtures
from smart_robustness.validation.native_glif_state_machine import simulate_glif


@pytest.fixture
def passive():
    p = copy.deepcopy(fixtures()[0]["parameters"])
    p.update(dt=0.001, El_reference=-0.065, th_inf=100.0, init_threshold=100.0)
    return p


def run(p, bias, ge, gi, ee=0.0, ei=-0.08):
    return simulate_conductance_glif(p, bias, ge, gi, e_exc=ee, e_inh=ei)


@pytest.mark.parametrize("fixture", fixtures(), ids=lambda f: f["identity"])
def test_exact_current_only_null_all_families_and_cuts(fixture):
    p = copy.deepcopy(fixture["parameters"])
    p["El_reference"] = -0.065
    bias = np.full(100, 4e-10)
    zero = np.zeros(100)
    original, old_state = simulate_glif(p, bias)
    extended, state = run(p, bias, zero, zero)
    assert exact_repeat(original, extended)
    assert old_state == state


def test_closed_form_passive_discrete_recurrence(passive):
    p = passive
    bias, ge, gi = 1e-11, 3e-9, 2e-9
    n = 80
    r, state = run(p, np.full(n, bias), np.full(n, ge), np.full(n, gi))
    c = p["C"] * p["coeffs"]["C"]
    gl = p["coeffs"]["G"] / p["R_input"]
    total = gl + ge + gi
    # Independent closed-form solution to the passive discrete recurrence.
    equilibrium = (
        bias + gl * p["El"] + ge * (0.0 - p["El_reference"])
        + gi * (-0.08 - p["El_reference"])
    ) / total
    expected = equilibrium + (p["init_voltage"] - equilibrium) * (
        1 - p["dt"] * total / c
    ) ** np.arange(1, n + 1)
    np.testing.assert_allclose(r["voltage"], expected, rtol=1e-12, atol=1e-14)
    assert not state["bad_reset_stop"] and not len(r["spike_time_steps"])


def test_coordinate_translation_and_reversal_null():
    shifted = effective_current(2.0, 3.0, 4.0, 0.2 - 0.5, 0.7 - 0.5, -0.1 - 0.5)
    native = effective_current(2.0, 3.0, 4.0, 0.2, 0.7, -0.1)
    assert shifted == pytest.approx(native)
    assert effective_current(2.0, 3.0, 4.0, -0.065, -0.065, -0.065) == 2.0


def test_conductance_changes_passive_euler_slope(passive):
    p = passive
    q = copy.deepcopy(p)
    q["init_voltage"] = 0.001
    zero = np.zeros(1)
    ge, gi = np.full(1, 3e-9), np.full(1, 2e-9)
    a, _ = run(p, zero, ge, gi)
    b, _ = run(q, zero, ge, gi)
    gl = p["coeffs"]["G"] / p["R_input"]
    slope = 1 - (gl + ge[0] + gi[0]) * p["dt"] / (p["C"] * p["coeffs"]["C"])
    assert b["voltage"][0] - a["voltage"][0] == pytest.approx(0.001 * slope)


def test_adaptive_threshold_uses_same_held_input():
    p = copy.deepcopy(fixtures()[-2]["parameters"])
    p["El_reference"] = -0.065
    bias, ge, gi = np.array([1e-11]), np.array([3e-9]), np.array([2e-9])
    held = bias[0] + ge[0] * 0.065 + gi[0] * (-0.015)
    original, _ = simulate_glif(p, np.array([held]))
    extended, _ = run(p, bias, ge, gi)
    for field in original:
        np.testing.assert_allclose(original[field], extended[field], rtol=1e-12, atol=1e-14)


def test_cut_skips_conductance_updates_but_uses_absolute_index():
    p = copy.deepcopy(fixtures()[1]["parameters"])
    p["El_reference"] = -0.065
    bias = np.full(30, 8e-10)
    zero = np.zeros(30)
    null, _ = run(p, bias, zero, zero)
    cut_mask = np.isnan(null["voltage"])
    ge = zero.copy()
    ge[cut_mask] = 1.0
    modified, _ = run(p, bias, ge, zero)
    assert exact_repeat(null, modified)


@pytest.mark.parametrize("fixture", fixtures(), ids=lambda f: f["identity"])
def test_conductance_repeat_is_exact(fixture):
    p = copy.deepcopy(fixture["parameters"])
    p["El_reference"] = -0.065
    bias = np.full(100, 4e-10)
    ge, gi = np.full(100, 2e-9), np.full(100, 1e-9)
    a, state_a = run(p, bias, ge, gi)
    b, state_b = run(p, bias, ge, gi)
    assert exact_repeat(a, b) and state_a == state_b


@pytest.mark.parametrize("bad", [-1.0, np.nan, np.inf])
def test_reject_bad_conductance(passive, bad):
    with pytest.raises(ValueError):
        run(passive, np.zeros(1), np.array([bad]), np.zeros(1))


@pytest.mark.parametrize("dt", [0.0, -1.0, np.inf, np.nan])
def test_reject_bad_timestep(passive, dt):
    passive["dt"] = dt
    with pytest.raises(ValueError, match="timestep"):
        run(passive, np.zeros(1), np.zeros(1), np.zeros(1))


def test_reject_shapes_and_nonfinite_bias_reversals(passive):
    for bias, ge, gi in [([0], [0, 0], [0]), ([[0]], [[0]], [[0]]), ([np.nan], [0], [0])]:
        with pytest.raises(ValueError):
            run(passive, bias, ge, gi)
    with pytest.raises(ValueError):
        run(passive, [0], [0], [0], ee=np.nan)


def test_native_adapter_contract_and_restoration(passive):
    class Probe:
        def __init__(self):
            self.calls = []

        def dynamics(self, *args):
            self.calls.append(args)
            return (1, 2, 3)

        def reset(self, *args):
            return (1, 2, 3, False)

        def run(self, bias):
            self.dynamics(0.001, 0.02, [0], bias[2], 2, [])
            self.reset(1, 2, 3)
            return {"probe": True}

    neuron = Probe()
    originals = neuron.dynamics, neuron.reset
    out, state = run_extended_native(
        neuron, passive, [0, 0, 1e-11], [0, 0, 3e-9], [0, 0, 2e-9], e_exc=0, e_inh=-0.08
    )
    assert out == {"probe": True}
    assert state == {"bad_reset_stop": False, "native_code_called": True}
    assert neuron.calls[0][3] == pytest.approx(1e-11 + 3e-9 * 0.064 + 2e-9 * (-0.016))
    assert (neuron.dynamics, neuron.reset) == originals


def test_native_adapter_restores_after_exception(passive):
    class Probe:
        def dynamics(self):
            pass

        def reset(self):
            pass

        def run(self, bias):
            raise RuntimeError("probe failure")

    neuron = Probe()
    originals = neuron.dynamics, neuron.reset
    with pytest.raises(RuntimeError, match="probe failure"):
        run_extended_native(neuron, passive, [0], [0], [0], e_exc=0, e_inh=-0.08)
    assert (neuron.dynamics, neuron.reset) == originals


def test_empty_input_has_consistent_outputs(passive):
    validate_inputs(passive, [], [], [], e_exc=0, e_inh=-0.08)
    r, _ = run(passive, [], [], [])
    assert r["voltage"].shape == (0,) and r["AScurrents"].shape == (0, 2)
