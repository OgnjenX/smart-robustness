from __future__ import annotations

import numpy as np
import pytest

from smart_robustness.validation.inhibitory_target_location import (
    ARMS,
    ARRIVALS_MS,
    PORT_NAME,
    REFERENCE_PORT_CONDUCTANCE_NS,
    cross_arm_gate,
    isolated_parameters,
    numerical_gate,
    replay_gate,
    replay_resource_amplitudes,
    target_port,
)


def test_registered_resource_amplitudes_are_fixed() -> None:
    assert np.array_equal(
        replay_resource_amplitudes(),
        np.array(
            [
                1.0,
                0.5906346234610091,
                0.42305461195209926,
                0.354453157440346,
                0.3263700971830197,
            ]
        ),
    )


def test_replay_gate_has_registered_boundary_and_weighted_peak() -> None:
    assert replay_gate(ARRIVALS_MS[0] - 1e-9) == 0.0
    assert replay_gate(ARRIVALS_MS[0]) == 0.0
    samples = np.arange(ARRIVALS_MS[0], ARRIVALS_MS[0] + 10.0, 0.001)
    peak = max(replay_gate(float(time)) for time in samples)
    assert peak == pytest.approx(2.0, rel=1e-6)


def test_target_ports_conserve_total_conductance() -> None:
    for arm, compartment in ARMS.items():
        port = target_port(arm)
        params = isolated_parameters(arm, "free_membrane")
        cell_compartment = params["cell_spec"].compartment(compartment)
        total_nS = (
            port.conductance_density_mS_cm2
            * cell_compartment.lateral_area_cm2
            * 1e6
        )
        assert port.name == PORT_NAME
        assert port.compartment == compartment
        assert total_nS == pytest.approx(REFERENCE_PORT_CONDUCTANCE_NS, rel=1e-15)


def test_isolated_parameters_remove_all_unregistered_ports() -> None:
    params = isolated_parameters("legacy_proximal", "free_membrane")
    assert len(params["synaptic_ports"]) == 1
    assert params["gap_junction_ports"] == ()
    assert params["external_input_ports"] == ()
    assert params["injection_ports"] == ()
    assert params["voltage_clamps_mV"] == {}
    clamped = isolated_parameters(
        "pv_like_somatic", "all_compartments_clamped_minus55"
    )
    assert clamped["voltage_clamps_mV"] == {
        "soma": -55.0,
        "proximal_dendrite": -55.0,
        "distal_dendrite": -55.0,
    }


def _trace(arm: str, soma: np.ndarray, proximal: np.ndarray, distal: np.ndarray):
    return {
        "time_ms": np.array([0.0, 100.0, 100.1, 100.2]),
        "v_soma_mV": soma,
        "v_proximal_dendrite_mV": proximal,
        "v_distal_dendrite_mV": distal,
        "target_effective_conductance_nS": np.array([0.0, 0.0, 10.0, 5.0]),
        "target_synaptic_current_pA": np.array([0.0, 0.0, -150.0, -75.0]),
    }


def test_cross_arm_clamp_requires_exact_resource_and_current() -> None:
    common = np.array([-55.0, -55.0, -55.0, -55.0])
    traces = {arm: _trace(arm, common, common, common) for arm in ARMS}
    result = cross_arm_gate(
        traces, protocol="all_compartments_clamped_minus55"
    )
    assert result["pass"] is True
    traces["sst_like_distal"]["target_synaptic_current_pA"][3] += 1e-6
    assert (
        cross_arm_gate(traces, protocol="all_compartments_clamped_minus55")[
            "pass"
        ]
        is False
    )


def test_cross_arm_free_gate_detects_location_effect_after_identical_prestate() -> None:
    base = np.array([-73.0, -73.0, -73.0, -73.0])
    traces = {
        "legacy_proximal": _trace("legacy_proximal", base.copy(), base.copy(), base.copy()),
        "pv_like_somatic": _trace("pv_like_somatic", base.copy(), base.copy(), base.copy()),
        "sst_like_distal": _trace("sst_like_distal", base.copy(), base.copy(), base.copy()),
    }
    traces["pv_like_somatic"]["v_soma_mV"][2:] += 0.02
    traces["sst_like_distal"]["v_distal_dendrite_mV"][2:] += 0.03
    result = cross_arm_gate(traces, protocol="free_membrane")
    assert result["pre_arrival_voltage_traces_exact_across_arms"] is True
    assert result["soma_location_effect_detected"] is True
    assert result["target_location_effect_detected"] is True
    assert result["pass"] is True


def test_numerical_gate_uses_registered_tolerances() -> None:
    summary = {
        "peak_effective_conductance_nS": 100.0,
        "integral_effective_conductance_nS_ms": 200.0,
        "peak_absolute_current_pA": 1000.0,
        "integral_absolute_current_pA_ms": 2000.0,
        "voltage": {
            compartment: {"minimum_mV": -73.0, "maximum_mV": -55.0}
            for compartment in ("soma", "proximal_dendrite", "distal_dendrite")
        },
    }
    fine = {
        **summary,
        "voltage": {
            compartment: {"minimum_mV": -73.1, "maximum_mV": -55.1}
            for compartment in ("soma", "proximal_dendrite", "distal_dendrite")
        },
    }
    assert numerical_gate(summary, fine)["pass"] is True
    fine["peak_absolute_current_pA"] = 1020.0
    assert numerical_gate(summary, fine)["pass"] is False


@pytest.mark.parametrize("arm", ["unknown", "PV", "sst"])
def test_unknown_arm_fails_closed(arm: str) -> None:
    with pytest.raises(ValueError):
        target_port(arm)
