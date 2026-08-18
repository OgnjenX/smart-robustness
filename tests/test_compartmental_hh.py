from __future__ import annotations

import pytest

brian = pytest.importorskip("brian2")

from smart_robustness.models.compartmental_hh import create_compartmental_hh_population
from smart_robustness.models.modeldb112923 import (
    ahp_ach_layer5_spec,
    ahp_density_to_total_nS,
    figure8_relay_spec,
    first_order_population_facts,
)


def _params(cell_class: str = "thalamic_relay") -> dict[str, str]:
    params = {
        "cell_class": cell_class,
        "axial_convention": "symmetric_cable",
        "leak_convention": "table3_reversal",
        "voltage_coordinate": "relative_to_table3_leak",
        "nak_rate_convention": "printed_smart",
        "calcium_gate_convention": "reciprocal",
        "calcium_voltage_coordinate": "integrated_voltage",
        "gate_initialization_convention": "steady_state_at_initial_voltage",
        "membrane_initialization_convention": "physical_leak_voltage",
        "spike_event_coordinate": "absolute_physical",
        "spike_event_threshold_mV": 30.0,
        "spike_event_rule": "latched_peak_then_zero",
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


def test_transmembrane_readout_is_negative_net_axial_current() -> None:
    brian.start_scope()
    brian.defaultclock.dt = 0.01 * brian.ms
    population = create_compartmental_hh_population(
        name="current_readout", size=1, params=_params("thalamic_relay"), brian=brian
    )
    population.group.v_soma = -60 * brian.mV
    population.group.v_proximal_dendrite = -50 * brian.mV
    network = brian.Network(population.group)
    network.run(0 * brian.ms)
    assert population.group.i_axial_inward_soma[0] / brian.pA > 0
    assert population.group.i_transmembrane_paper_soma[0] / brian.pA == pytest.approx(
        population.group.i_axial_inward_soma[0] / brian.pA
    )
    assert population.group.i_transmembrane_outward_soma[0] / brian.pA == pytest.approx(
        -population.group.i_axial_inward_soma[0] / brian.pA
    )


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


def test_modeldb_reticular_calcium_gates_initialize_in_absolute_voltage_coordinate() -> None:
    brian.start_scope()
    params = _params()
    trn = next(fact for fact in first_order_population_facts() if fact.canonical_name == "trn")
    params.update(
        {
            "cell_spec": trn.cell,
            "calcium_gate_convention": "modeldb_reticular_112923",
            "e_k_mV": trn.e_k_mV,
            "e_ca_mV": trn.e_ca_mV,
        }
    )
    population = create_compartmental_hh_population(
        name="reticular_gate_initialization", size=1, params=params, brian=brian
    )
    expected_m = 1 / (1 + brian.exp((-52.0 + 69.0) / 7.4))
    expected_h = 1 / (1 + brian.exp((-80.0 + 69.0) / -5.0))
    assert population.group.m_ca_soma[0] == pytest.approx(float(expected_m))
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


def test_spike_event_emits_below_zero_after_preceding_value_above_30() -> None:
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
    group.armed = 0
    spike_monitor = brian.SpikeMonitor(group)
    network = brian.Network(group, spike_monitor)
    group.v_soma = 60 * brian.mV
    network.run(0.1 * brian.ms)
    assert spike_monitor.count[0] == 0
    assert group.armed[0] == 1
    network.run(0.1 * brian.ms)
    assert spike_monitor.count[0] == 0
    group.v_soma = -1 * brian.mV
    network.run(0.1 * brian.ms)
    assert spike_monitor.count[0] == 1
    assert group.armed[0] == 0
    group.v_soma = 60 * brian.mV
    network.run(0.1 * brian.ms)
    assert spike_monitor.count[0] == 1
    group.v_soma = -1 * brian.mV
    network.run(0.1 * brian.ms)
    assert spike_monitor.count[0] == 2


def test_source_spike_event_coordinate_is_relative_to_soma_leak() -> None:
    brian.start_scope()
    brian.defaultclock.dt = 0.01 * brian.ms
    params = _params()
    params["spike_event_coordinate"] = "relative_to_soma_leak"
    params["voltage_clamps_mV"] = {"soma": -20.0}
    population = create_compartmental_hh_population(
        name="relative_spike_coordinate", size=1, params=params, brian=brian
    )
    spike_monitor = brian.SpikeMonitor(population.group)
    network = brian.Network(population.group, spike_monitor)

    # Table 3 relay E_leak=-60 mV, so -20 mV arms at +40 mV in the
    # leak-relative coordinate; release follows after returning below -60 mV.
    network.run(0.01 * brian.ms)
    assert spike_monitor.count[0] == 0
    assert population.group.armed[0] == 1
    population.group.v_soma = -61 * brian.mV
    network.run(0.01 * brian.ms)
    assert spike_monitor.count[0] == 1
    assert population.group.armed[0] == 0


def test_source_spike_event_coordinate_accepts_fixed_67_mv_shift() -> None:
    brian.start_scope()
    brian.defaultclock.dt = 0.01 * brian.ms
    params = _params()
    params["spike_event_coordinate"] = "shifted_67_mV"
    params["voltage_clamps_mV"] = {"soma": -30.0}
    population = create_compartmental_hh_population(
        name="shifted_67_spike_coordinate", size=1, params=params, brian=brian
    )
    spike_monitor = brian.SpikeMonitor(population.group)
    network = brian.Network(population.group, spike_monitor)

    # SMART's fixed +67 mV coordinate maps physical -30 mV to +37 mV,
    # then emits when the physical soma falls below -67 mV.
    network.run(0.01 * brian.ms)
    assert spike_monitor.count[0] == 0
    assert population.group.armed[0] == 1
    population.group.v_soma = -68 * brian.mV
    network.run(0.01 * brian.ms)
    assert spike_monitor.count[0] == 1
    assert population.group.armed[0] == 0


def test_kinness_minus_20_mv_event_threshold_is_explicit() -> None:
    brian.start_scope()
    brian.defaultclock.dt = 0.01 * brian.ms
    params = _params()
    params["spike_event_threshold_mV"] = -20.0
    params["voltage_clamps_mV"] = {"soma": -10.0}
    population = create_compartmental_hh_population(
        name="kinness_minus20_spike_threshold", size=1, params=params, brian=brian
    )
    spike_monitor = brian.SpikeMonitor(population.group)
    network = brian.Network(population.group, spike_monitor)
    network.run(0.01 * brian.ms)
    assert population.group.armed[0] == 1
    population.group.v_soma = -1 * brian.mV
    network.run(0.01 * brian.ms)
    assert spike_monitor.count[0] == 1


def test_literal_previous_sample_rule_uses_only_the_immediately_preceding_voltage() -> None:
    brian.start_scope()
    brian.defaultclock.dt = 0.01 * brian.ms
    params = _params()
    params["spike_event_rule"] = "literal_previous_sample"
    params["spike_event_threshold_mV"] = 30.0
    params["voltage_clamps_mV"] = {"soma": 40.0}
    population = create_compartmental_hh_population(
        name="literal_previous_sample", size=1, params=params, brian=brian
    )
    spike_monitor = brian.SpikeMonitor(population.group)
    network = brian.Network(population.group, spike_monitor)

    network.run(0.01 * brian.ms)
    assert population.group.previous_spike_voltage[0] / brian.mV == pytest.approx(40.0)
    population.group.v_soma = -1 * brian.mV
    network.run(0.01 * brian.ms)
    assert spike_monitor.count[0] == 1
    network.run(0.01 * brian.ms)
    assert spike_monitor.count[0] == 1


def test_literal_previous_sample_initializes_fixed_shift_coordinate() -> None:
    brian.start_scope()
    params = _params()
    params["spike_event_rule"] = "literal_previous_sample"
    params["spike_event_coordinate"] = "shifted_67_mV"
    params["v_init_mV"] = -60.0
    population = create_compartmental_hh_population(
        name="literal_previous_sample_shifted_67", size=1, params=params, brian=brian
    )
    assert population.group.previous_spike_voltage[0] / brian.mV == pytest.approx(7.0)


def test_spike_event_rule_is_required_and_not_an_implicit_simulator_default() -> None:
    brian.start_scope()
    params = _params()
    del params["spike_event_rule"]
    with pytest.raises(KeyError, match="spike_event_rule"):
        create_compartmental_hh_population(
            name="missing_spike_event_rule", size=1, params=params, brian=brian
        )


def test_membrane_initialization_coordinate_is_required_and_explicit() -> None:
    brian.start_scope()
    params = _params()
    del params["membrane_initialization_convention"]
    with pytest.raises(KeyError, match="membrane_initialization_convention"):
        create_compartmental_hh_population(
            name="missing_membrane_initialization", size=1, params=params, brian=brian
        )

    params = _params()
    params["membrane_initialization_convention"] = "kinness_internal_zero"
    population = create_compartmental_hh_population(
        name="internal_zero_initialization", size=1, params=params, brian=brian
    )
    assert population.group.v_soma[0] / brian.mV == pytest.approx(0.0)


def test_calcium_voltage_coordinate_is_required() -> None:
    brian.start_scope()
    params = _params()
    del params["calcium_voltage_coordinate"]
    with pytest.raises(KeyError, match="calcium_voltage_coordinate"):
        create_compartmental_hh_population(
            name="missing_calcium_voltage_coordinate", size=1, params=params, brian=brian
        )


def test_hysteretic_rule_does_not_rearm_between_zero_and_negative_threshold() -> None:
    brian.start_scope()
    brian.defaultclock.dt = 0.01 * brian.ms
    params = _params()
    params["spike_event_rule"] = "hysteretic_threshold_then_zero"
    params["spike_event_threshold_mV"] = -20.0
    params["voltage_clamps_mV"] = {"soma": -10.0}
    population = create_compartmental_hh_population(
        name="hysteretic_spike_detector", size=1, params=params, brian=brian
    )
    spike_monitor = brian.SpikeMonitor(population.group)
    network = brian.Network(population.group, spike_monitor)

    network.run(0.01 * brian.ms)
    assert population.group.armed[0] == 1
    population.group.v_soma = -1 * brian.mV
    network.run(0.01 * brian.ms)
    assert spike_monitor.count[0] == 1
    assert population.group.armed[0] == -1
    network.run(0.02 * brian.ms)
    assert spike_monitor.count[0] == 1
    population.group.v_soma = -21 * brian.mV
    network.run(0.01 * brian.ms)
    assert population.group.armed[0] == 0


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
    params = _params("layer5_excitatory")
    params["voltage_clamps_mV"] = {"soma": 60.0}
    population = create_compartmental_hh_population(
        name="layer5_modulation", size=1, params=params, brian=brian
    )
    group = population.group
    group.v_soma = 60 * brian.mV
    group.armed = 0
    network = brian.Network(group)
    network.run(0.1 * brian.ms)
    group.v_soma = -1 * brian.mV
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
    params["voltage_clamps_mV"] = {"soma": 60.0}
    population = create_compartmental_hh_population(
        name="layer5_modeldb_ahp", size=1, params=params, brian=brian
    )
    group = population.group
    group.v_soma = 60 * brian.mV
    group.armed = 0
    network = brian.Network(group)
    network.run(0.1 * brian.ms)
    group.v_soma = -1 * brian.mV
    network.run(0.1 * brian.ms)
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
