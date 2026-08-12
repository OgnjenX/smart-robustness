from __future__ import annotations

import numpy as np
import pytest

brian = pytest.importorskip("brian2")

from smart_robustness.classic_sector import (
    CalciumKineticsConvention,
    FirstOrderRuntimeConventions,
    IntrinsicCellConvention,
    ZeroSensitivityInputConvention,
    build_first_order_chemical_sector,
    build_first_order_intrinsic_sector,
    build_first_order_voltage_clamp_sector,
    figure6_runtime_conventions,
    first_order_population_parameters,
)
from smart_robustness.models.modeldb112923 import first_order_population_facts


def test_protocol_voltage_clamp_preserves_sector_and_pins_selected_relay_dendrites() -> None:
    brian.start_scope()
    brian.prefs.codegen.target = "numpy"
    brian.defaultclock.dt = 0.01 * brian.ms
    selected = (38, 39, 40, 41, 42)
    sector = build_first_order_voltage_clamp_sector(
        clamped_relay_indices=selected,
        conventions=figure6_runtime_conventions(),
        brian=brian,
    )
    assert sector.cell_count == 812
    assert sector.compartment_count == 1950
    assert len(sector.projections) == 53
    relay = sector.populations["thalamic_relay"].group
    sector.network.run(0.02 * brian.ms)
    assert np.asarray(relay.v_proximal_dendrite[list(selected)] / brian.mV) == pytest.approx(-12)
    assert float(relay.v_proximal_dendrite[0] / brian.mV) != pytest.approx(-12)


def test_first_order_intrinsic_sector_builds_all_source_cells() -> None:
    brian.start_scope()
    brian.defaultclock.dt = 0.01 * brian.ms
    sector = build_first_order_intrinsic_sector(brian=brian)
    assert len(sector.populations) == 12
    assert sector.cell_count == 812
    assert sector.compartment_count == 1950
    assert int(sector.populations["thalamic_relay"].group.N) == 81
    assert int(sector.populations["thalamic_nonspecific"].group.N) == 1
    assert (
        sum(len(population.compiled.synaptic_ports) for population in sector.populations.values())
        == 51
    )
    assert (
        sum(
            len(population.compiled.external_input_ports)
            for population in sector.populations.values()
        )
        == 10
    )
    assert (
        sum(len(population.compiled.injection_ports) for population in sector.populations.values())
        == 1
    )
    assert (
        sum(
            len(population.compiled.gap_junction_ports)
            for population in sector.populations.values()
        )
        == 4
    )
    sector.network.run(0 * brian.ms)


def test_classic_sector_uses_kinness_rest_relative_nak_voltage() -> None:
    brian.start_scope()
    sector = build_first_order_intrinsic_sector(brian=brian)
    assert {
        population.compiled.voltage_coordinate.value
        for population in sector.populations.values()
    } == {"relative_to_table3_leak"}


def test_runtime_convention_fingerprint_is_stable_and_sensitive() -> None:
    classic = FirstOrderRuntimeConventions()
    assert classic.fingerprint == FirstOrderRuntimeConventions().fingerprint
    zero = FirstOrderRuntimeConventions(gate_initialization_convention="zero")
    assert zero.fingerprint != classic.fingerprint
    paper_cells = FirstOrderRuntimeConventions(intrinsic_cell_convention="paper_table3")
    assert paper_cells.fingerprint != classic.fingerprint


def test_intrinsic_cell_source_is_explicit_and_does_not_mix_trn_densities() -> None:
    trn = next(fact for fact in first_order_population_facts() if fact.canonical_name == "trn")
    executable = first_order_population_parameters(
        trn,
        conventions=FirstOrderRuntimeConventions(
            intrinsic_cell_convention=IntrinsicCellConvention.MODELDB_112923.value
        ),
    )["cell_spec"]
    paper = first_order_population_parameters(
        trn,
        conventions=FirstOrderRuntimeConventions(
            intrinsic_cell_convention=IntrinsicCellConvention.PAPER_TABLE3.value
        ),
    )["cell_spec"]

    assert executable.soma.g_ca_mS_cm2 == 100
    assert executable.compartment("proximal_dendrite").g_ca_mS_cm2 == 100
    assert executable.soma.g_k_mS_cm2 == 80
    assert paper.soma.g_ca_mS_cm2 is None
    assert paper.compartment("proximal_dendrite").g_ca_mS_cm2 == 10
    assert paper.soma.g_k_mS_cm2 == 100


def test_calcium_kinetics_source_is_independent_from_intrinsic_cell_source() -> None:
    trn = next(fact for fact in first_order_population_facts() if fact.canonical_name == "trn")
    executable = first_order_population_parameters(
        trn,
        conventions=FirstOrderRuntimeConventions(
            intrinsic_cell_convention="paper_table3",
            calcium_kinetics_convention=CalciumKineticsConvention.MODELDB_112923.value,
        ),
    )
    paper = first_order_population_parameters(
        trn,
        conventions=FirstOrderRuntimeConventions(
            intrinsic_cell_convention="paper_table3",
            calcium_kinetics_convention=CalciumKineticsConvention.PAPER_2008.value,
            calcium_gate_convention="reciprocal",
        ),
    )

    assert executable["calcium_gate_convention"] == "modeldb_reticular_112923"
    assert paper["calcium_gate_convention"] == "reciprocal"


def test_figure6_profile_names_the_source_constrained_runtime() -> None:
    conventions = figure6_runtime_conventions()
    assert conventions.gate_initialization_convention == "steady_state_at_initial_voltage"
    assert conventions.intrinsic_cell_convention == "modeldb_112923"
    assert conventions.calcium_kinetics_convention == "modeldb_112923"
    assert conventions.zero_sensitivity_input_convention == "omit_all_zero"
    assert conventions.spike_event_coordinate == "absolute_physical"
    assert conventions.spike_event_threshold_mV == 30.0
    assert conventions.specific_capacitance_uF_cm2 == 1.0
    assert conventions.gaussian_weight_convention == "source_peak"
    assert conventions.modifiable_weight_initialization == "figure6_pathway_specific"
    assert conventions.gaussian_learning_bounds_convention == "figure6_pathway_specific"
    assert conventions.projection_source_convention == "modeldb_as_serialized"


def test_cross_checked_profile_resolves_ampa_23_source_without_mutating_catalog() -> None:
    brian.start_scope()
    literal = build_first_order_chemical_sector(brian=brian)
    literal_projection = literal.projections["modeldb112923.projection.022"]
    assert literal.populations["layer4_inhibitory_v1"].group is literal_projection.source

    brian.start_scope()
    corrected = build_first_order_chemical_sector(
        conventions=FirstOrderRuntimeConventions(
            projection_source_convention="paper_supplement_cross_checked"
        ),
        brian=brian,
    )
    corrected_projection = corrected.projections["modeldb112923.projection.022"]
    assert corrected.populations["layer23_excitatory_v1"].group is corrected_projection.source
    assert len(corrected_projection) == 81


def test_all_zero_input_channels_have_an_explicit_legacy_convention() -> None:
    brian.start_scope()
    conventions = FirstOrderRuntimeConventions(
        zero_sensitivity_input_convention=ZeroSensitivityInputConvention.OMIT_ALL_ZERO.value
    )
    sector = build_first_order_intrinsic_sector(conventions=conventions, brian=brian)
    ports = tuple(
        port
        for population in sector.populations.values()
        for port in population.compiled.external_input_ports
    )
    assert ports
    assert all(any(port.sensitivities_mV) for port in ports)
    assert "modeldb112923.external.001" not in {port.record_id for port in ports}


def test_full_runtime_profile_reaches_every_population() -> None:
    brian.start_scope()
    conventions = FirstOrderRuntimeConventions(
        gate_initialization_convention="zero",
        specific_capacitance_uF_cm2=2.0,
    )
    sector = build_first_order_intrinsic_sector(conventions=conventions, brian=brian)
    relay = sector.populations["thalamic_relay"]
    assert np.allclose(relay.group.m_soma[:], 0)
    assert relay.group.C_soma[0] / brian.pfarad == pytest.approx(
        2.0 * relay.cell_spec.soma.lateral_area_cm2 * 1e6
    )


def test_runtime_profile_and_legacy_gate_override_are_mutually_exclusive() -> None:
    brian.start_scope()
    with pytest.raises(ValueError, match="not both"):
        build_first_order_intrinsic_sector(
            conventions=FirstOrderRuntimeConventions(),
            gate_initialization_convention="zero",
            brian=brian,
        )


def test_network_layer5_and_layer6ii_use_serialized_fast_ahp() -> None:
    brian.start_scope()
    sector = build_first_order_intrinsic_sector(brian=brian)
    for name in ("layer5_excitatory_v1", "layer6ii_excitatory_v1"):
        population = sector.populations[name]
        assert population.compiled.ahp_ach_enabled
        assert population.compiled.ahp_convention.value == "smart_network_112923"
        assert "dahp_fall/dt=-ahp_fall/(20.0*ms)" in population.compiled.equations


def test_full_sector_gate_initialization_is_selectable() -> None:
    brian.start_scope()
    sector = build_first_order_intrinsic_sector(
        gate_initialization_convention="zero", brian=brian
    )
    relay = sector.populations["thalamic_relay"].group
    assert np.allclose(relay.m_soma[:], 0)
    assert np.allclose(relay.h_soma[:], 0)
    assert np.allclose(relay.n_soma[:], 0)
    assert np.allclose(relay.m_ca_proximal_dendrite[:], 0)
    assert np.allclose(relay.h_ca_proximal_dendrite[:], 0)


def test_source_populations_expose_presynaptic_transmitter_depletion() -> None:
    brian.start_scope()
    sector = build_first_order_intrinsic_sector(brian=brian)
    for name in (
        "layer5_excitatory_v1",
        "layer6ii_excitatory_v1",
        "layer6i_excitatory_v1",
    ):
        population = sector.populations[name]
        assert population.compiled.depletion_enabled
        assert np.allclose(population.group.transmitter[:], 1.0)
    assert not sector.populations["layer4_excitatory_v1"].compiled.depletion_enabled


def test_bottom_up_gate_is_explicitly_addressable_by_source_record() -> None:
    brian.start_scope()
    sector = build_first_order_intrinsic_sector(brian=brian)
    relay = sector.populations["thalamic_relay"]
    record_id = "modeldb112923.external.002"
    relay.set_external_input(record_id, "green", 64, indices=[40])
    port = next(port for port in relay.compiled.external_input_ports if port.record_id == record_id)
    assert getattr(relay.group, f"{port.name}_input_green")[40] == pytest.approx(64)
    assert getattr(relay.group, f"{port.name}_input_green")[0] == pytest.approx(0.0)
    with pytest.raises(ValueError, match="between zero and 255"):
        relay.set_external_input(record_id, "green", 256)


def test_direct_current_gate_is_explicitly_addressable_by_source_record() -> None:
    brian.start_scope()
    sector = build_first_order_intrinsic_sector(brian=brian)
    relay = sector.populations["thalamic_relay"]
    record_id = "modeldb112923.external.000"
    relay.set_external_injection(record_id, "red", 122, indices=[40])
    port = relay.compiled.injection_ports[0]
    values = np.asarray(getattr(relay.group, f"{port.name}_input_red")[:])
    assert values[40] == 122
    assert np.count_nonzero(values) == 1
    assert f"i_{port.name}=" in relay.compiled.equations
    with pytest.raises(ValueError, match="between zero and 255"):
        relay.set_external_injection(record_id, "red", -1)
