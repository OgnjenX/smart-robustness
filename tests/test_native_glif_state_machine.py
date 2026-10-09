"""Hand-derived scheduling checks precede every native-reference outcome."""

from __future__ import annotations

import numpy as np
import pytest

from smart_robustness.validation.native_glif_state_machine import simulate_glif


@pytest.fixture
def parameters():
    return {
        "dt": 1.0,
        "C": 1.0,
        "R_input": 1.0,
        "El": 0.0,
        "th_inf": 1.0,
        "init_voltage": 0.0,
        "init_threshold": 1.0,
        "init_AScurrents": [0.0],
        "asc_tau_array": [1.0],
        "asc_amp_array": [0.0],
        "spike_cut_length": 0,
        "AScurrent_dynamics_method": {"name": "none", "params": {}},
        "AScurrent_reset_method": {"name": "none", "params": {}},
        "voltage_dynamics_method": {"name": "linear_forward_euler", "params": {}},
        "voltage_reset_method": {"name": "zero", "params": {}},
        "threshold_dynamics_method": {"name": "inf", "params": {}},
        "threshold_reset_method": {"name": "inf", "params": {}},
    }


def test_strict_crossing_not_threshold_equality(parameters):
    r, state = simulate_glif(parameters, np.array([1.0, 1.0]))
    np.testing.assert_array_equal(r["voltage"], [1.0, 1.0])
    assert not len(r["spike_time_steps"]) and not state["native_code_called"]


def test_cut_nan_window_reset_sample_and_legacy_interpolation(parameters):
    parameters["spike_cut_length"] = 2
    r, _ = simulate_glif(parameters, np.full(7, 2.0))
    np.testing.assert_array_equal(r["spike_time_steps"], [0, 3, 6])
    np.testing.assert_array_equal(
        np.isnan(r["voltage"]), [True, True, False, True, True, False, True]
    )
    np.testing.assert_array_equal(r["interpolated_spike_times"], [0.5, 3.5, 6.5])
    np.testing.assert_array_equal(r["interpolated_spike_voltage"], [3.0, 3.0, 3.0])


def test_voltage_uses_old_not_new_afterspike_current(parameters):
    parameters["init_AScurrents"] = [0.25]
    parameters["AScurrent_dynamics_method"]["name"] = "exp"
    r, _ = simulate_glif(parameters, np.zeros(1))
    assert r["voltage"][0] == 0.25
    assert r["AScurrents"][0, 0] == pytest.approx(0.25 * np.exp(-1))


def test_reset_uses_postintegration_voltage(parameters):
    parameters["voltage_reset_method"] = {"name": "v_before", "params": {"a": 0.25, "b": 0.1}}
    r, _ = simulate_glif(parameters, np.array([2.0]))
    assert r["voltage"][0] == 0.6


def test_bad_reset_tail_preserved(parameters):
    parameters["voltage_reset_method"] = {"name": "v_before", "params": {"a": 1.0, "b": 0.0}}
    r, state = simulate_glif(parameters, np.full(10, 2.0))
    assert state["bad_reset_stop"]
    np.testing.assert_array_equal(r["voltage"][:6], np.full(6, 2.0))
    assert np.isnan(r["voltage"][6:]).all()


def test_unknown_method_not_aliased(parameters):
    parameters["voltage_dynamics_method"]["name"] = "not-implemented"
    with pytest.raises(ValueError, match="unsupported"):
        simulate_glif(parameters, np.zeros(1))


def test_adaptive_voltage_threshold_hand_solution(parameters):
    parameters["threshold_dynamics_method"] = {
        "name": "three_components_exact",
        "params": {"a_spike": 0.1, "b_spike": 1.0, "a_voltage": 0.2, "b_voltage": 2.0},
    }
    r, _ = simulate_glif(parameters, np.array([0.5]))
    expected = 1.0 - 0.1 * np.exp(-1) + 0.05 * np.exp(-2) + 0.05
    assert r["threshold"][0] == pytest.approx(expected)


def test_afterspike_reset_advances_old_currents_through_cut(parameters):
    parameters["spike_cut_length"] = 2
    parameters["init_AScurrents"] = [0.25]
    parameters["asc_amp_array"] = [0.1]
    parameters["coeffs"] = {"asc_amp_array": [2.0]}
    parameters["AScurrent_dynamics_method"]["name"] = "exp"
    parameters["AScurrent_reset_method"] = {"name": "sum", "params": {"r": [0.5]}}
    r, _ = simulate_glif(parameters, np.full(3, 2.0))
    assert r["AScurrents"][2, 0] == pytest.approx(0.2 + 0.125 * np.exp(-3))


def test_nan_voltage_does_not_become_a_false_spike(parameters):
    r, _ = simulate_glif(parameters, np.array([np.nan]))
    assert np.isnan(r["voltage"][0]) and not len(r["spike_time_steps"])


def test_repeat_is_exact_without_state_leakage(parameters):
    first, first_state = simulate_glif(parameters, np.full(7, 2.0))
    second, second_state = simulate_glif(parameters, np.full(7, 2.0))
    assert first_state == second_state
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
