from __future__ import annotations

import pytest

brian = pytest.importorskip("brian2")

from smart_robustness.models.compartmental_hh import create_compartmental_hh_population
from smart_robustness.models.modeldb112923 import (
    ahp_ach_layer5_spec,
    ahp_density_to_total_nS,
    figure8_relay_spec,
)


def _params(cell_class: str = "thalamic_relay") -> dict[str, str]:
    params = {
        "cell_class": cell_class,
        "axial_convention": "symmetric_cable",
        "leak_convention": "table3_reversal",
        "voltage_coordinate": "relative_to_table3_leak",
        "nak_rate_convention": "printed_smart",
        "calcium_gate_convention": "reciprocal",
        "gate_initialization_convention": "steady_state_at_initial_voltage",
        "calcium_density_convention": "table3",
        "ahp_convention": "paper_text",
        "specific_capacitance_uF_cm2": 1.0,
        "enable_ahp_ach": cell_class == "layer5_excitatory",
    }
    if cell_class == "layer5_excitatory":
        params["ahp_max_conductance_nS"] = 1.0
        params["ahp_event_weight"] = 1.0
        params["e_ahp_mV"] = -90.0
    return params


@pytest.mark.parametrize("cell_class", ["thalamic_relay", "layer4_excitatory", "layer5_excitatory"])
def test_vectorized_population_runs_and_exposes_published_compartments(cell_class: str) -> None:
    brian.start_scope()
    brian.defaultclock.dt = 0.01 * brian.ms
    population = create_compartmental_hh_population(
        name=f"test_{cell_class}", size=3, params=_params(cell_class), brian=brian
    )
    monitor = brian.StateMonitor(population.group, "v_soma", record=True)
    brian.Network(population.group, monitor).run(0.1 * brian.ms)
    assert population.compartments == tuple(
        compartment.name for compartment in population.cell_spec.compartments
    )
    assert brian.numpy_.isfinite(monitor.v_soma[:]).all()


def test_relay_uses_table3_calcium_density_without_silent_global_override() -> None:
    brian.start_scope()
    population = create_compartmental_hh_population(
        name="relay_table3_ca", size=1, params=_params(), brian=brian
    )
    expected = population.cell_spec.compartment("proximal_dendrite").conductance_nS("ca")
    assert population.group.g_ca_proximal_dendrite[0] / brian.nsiemens == pytest.approx(expected)


def test_population_accepts_an_explicit_source_specific_cell_spec() -> None:
    brian.start_scope()
    params = _params()
    params["cell_spec"] = figure8_relay_spec(leak_density_mS_cm2=0.1)
    population = create_compartmental_hh_population(
        name="figure8_source_cell", size=1, params=params, brian=brian
    )
    assert population.cell_spec.name == "modeldb112923_figure8_relay"
    assert population.group.g_ca_soma[0] / brian.nsiemens > 0


def test_modeldb_calcium_gates_initialize_in_absolute_voltage_coordinate() -> None:
    brian.start_scope()
    params = _params()
    params.update(
        {
            "cell_spec": figure8_relay_spec(leak_density_mS_cm2=0.1),
            "voltage_coordinate": "shifted_67_mV",
            "nak_rate_convention": "standard_traub_miles",
            "calcium_gate_convention": "modeldb_112923",
            "v_init_mV": -80.0,
        }
    )
    population = create_compartmental_hh_population(
        name="figure8_gate_initialization", size=1, params=params, brian=brian
    )
    expected_h = 1 / (1 + brian.exp((-80.0 + 83.5) / 6.3))
    assert population.group.h_ca_soma[0] == pytest.approx(float(expected_h))


def test_zero_gate_initialization_is_an_explicit_audit_alternative() -> None:
    brian.start_scope()
    params = _params()
    params["gate_initialization_convention"] = "zero"
    population = create_compartmental_hh_population(
        name="zero_gate_initialization", size=1, params=params, brian=brian
    )
    assert population.group.m_soma[0] == 0
    assert population.group.h_soma[0] == 0
    assert population.group.n_soma[0] == 0
    assert population.group.m_ca_proximal_dendrite[0] == 0
    assert population.group.h_ca_proximal_dendrite[0] == 0


def test_gate_initialization_convention_is_required() -> None:
    brian.start_scope()
    params = _params()
    del params["gate_initialization_convention"]
    with pytest.raises(KeyError, match="gate_initialization_convention"):
        create_compartmental_hh_population(
            name="missing_gate_initialization", size=1, params=params, brian=brian
        )


def test_exact_voltage_clamp_is_fixed_during_integration() -> None:
    brian.start_scope()
    brian.defaultclock.dt = 0.01 * brian.ms
    params = _params()
    params["voltage_clamps_mV"] = {"proximal_dendrite": -12.0}
    population = create_compartmental_hh_population(
        name="exact_proximal_clamp", size=1, params=params, brian=brian
    )
    brian.Network(population.group).run(0.1 * brian.ms)
    assert population.group.v_proximal_dendrite[0] / brian.mV == pytest.approx(-12.0)
    assert population.compiled.voltage_clamped_compartments == frozenset(
        {"proximal_dendrite"}
    )


def test_spike_event_arms_above_30_and_emits_once_below_zero() -> None:
    brian.start_scope()
    brian.defaultclock.dt = 0.1 * brian.ms
    group = brian.NeuronGroup(
        1,
        "dv_soma/dt=0*volt/second : volt\narmed : 1",
        threshold="armed > 0.5 and v_soma < 0*mV",
        reset="armed = 0",
        events={"arm_spike": "armed < 0.5 and v_soma > 30*mV"},
        method="euler",
    )
    group.run_on_event("arm_spike", "armed=1", when="after_thresholds", order=1)
    spike_monitor = brian.SpikeMonitor(group)
    network = brian.Network(group, spike_monitor)
    group.v_soma = 31 * brian.mV
    network.run(0.1 * brian.ms)
    assert group.armed[0] == 1
    group.v_soma = -1 * brian.mV
    network.run(0.1 * brian.ms)
    assert spike_monitor.count[0] == 1
    network.run(0.2 * brian.ms)
    assert spike_monitor.count[0] == 1


def test_layer5_requires_source_unidentified_ahp_conductance_explicitly() -> None:
    brian.start_scope()
    params = _params("layer5_excitatory")
    del params["ahp_max_conductance_nS"]
    with pytest.raises(ValueError, match="explicit ahp_max_conductance_nS"):
        create_compartmental_hh_population(
            name="layer5_missing_ahp", size=1, params=params, brian=brian
        )


def test_layer5_spike_generates_ahp_and_ach_suppresses_it() -> None:
    brian.start_scope()
    brian.defaultclock.dt = 0.1 * brian.ms
    population = create_compartmental_hh_population(
        name="layer5_modulation", size=1, params=_params("layer5_excitatory"), brian=brian
    )
    group = population.group
    group.v_soma = -1 * brian.mV
    group.armed = 1
    network = brian.Network(group)
    network.run(0.1 * brian.ms)
    assert group.ahp_rise[0] > 0
    assert group.ahp_fall[0] > 0
    network.run(10 * brian.ms)
    unsuppressed = float(group.ahp_gate[0] * (1 - group.ach_gate[0]))
    population.trigger_ach()
    network.run(5.5 * brian.ms)
    suppressed = float(group.ahp_gate[0] * (1 - group.ach_gate[0]))
    assert group.ach_gate[0] > 0.9
    assert suppressed < unsuppressed


def test_modeldb_layer5_spike_uses_serialized_ahp_weight() -> None:
    brian.start_scope()
    params = _params("layer5_excitatory")
    params["ahp_convention"] = "modeldb_112923"
    params["cell_spec"] = ahp_ach_layer5_spec(soma_axial_resistance_kohm_cm=35.0)
    params["ahp_max_conductance_nS"] = ahp_density_to_total_nS(0.1, params["cell_spec"])
    params["ahp_event_weight"] = 4.5
    population = create_compartmental_hh_population(
        name="layer5_modeldb_ahp", size=1, params=params, brian=brian
    )
    group = population.group
    group.v_soma = -1 * brian.mV
    group.armed = 1
    brian.Network(group).run(0.1 * brian.ms)
    assert group.ahp_rise[0] == pytest.approx(1.0)
    assert group.ahp_fall[0] == pytest.approx(1.0)
    assert group.ahp_event_weight[0] == pytest.approx(4.5)


def test_source_specific_ahp_cell_geometry_and_density_conversion() -> None:
    cell = ahp_ach_layer5_spec(soma_axial_resistance_kohm_cm=35.0)
    assert cell.compartment("soma").diameter_mm == pytest.approx(0.1)
    assert cell.compartment("proximal_dendrite").length_mm == pytest.approx(0.1)
    assert cell.compartment("soma").e_leak_mV == -78.0
    assert cell.compartment("proximal_dendrite").axial_resistance_kohm_cm == 35.0
    expected = 0.1 * cell.compartment("soma").lateral_area_cm2 * 1e6
    assert ahp_density_to_total_nS(0.1, cell) == pytest.approx(expected)


def test_specific_capacitance_is_required_and_positive() -> None:
    brian.start_scope()
    params = _params()
    del params["specific_capacitance_uF_cm2"]
    with pytest.raises(KeyError, match="specific_capacitance_uF_cm2"):
        create_compartmental_hh_population(
            name="missing_capacitance", size=1, params=params, brian=brian
        )

    params["specific_capacitance_uF_cm2"] = 0
    with pytest.raises(ValueError, match="must be positive"):
        create_compartmental_hh_population(
            name="invalid_capacitance", size=1, params=params, brian=brian
        )
