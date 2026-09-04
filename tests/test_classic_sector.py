from __future__ import annotations

import numpy as np
import pytest

brian = pytest.importorskip("brian2")

from smart_robustness.classic_sector import (
    CalciumKineticsConvention,
    FirstOrderRuntimeConventions,
    IntrinsicCellConvention,
    NonspecificDistalGabaSourceConvention,
    TrnCalciumSourceConvention,
    TrnDendriticCalciumDensityConvention,
    TrnPotassiumConvention,
    ZeroSensitivityInputConvention,
    _resolved_projection_record,
    _ring_kernel_convention_for_record,
    _ring_peak_radius_scale_for_record,
    build_first_order_chemical_sector,
    build_first_order_intrinsic_sector,
    build_first_order_voltage_clamp_sector,
    figure6_runtime_conventions,
    first_order_population_parameters,
)
from smart_robustness.modeldb_projections import MODELDB_FIRST_ORDER
from smart_robustness.models.modeldb112923 import first_order_population_facts


def test_methods_dual_and_override_is_scoped_to_adaptive_corticothalamic_records() -> None:
    conventions = FirstOrderRuntimeConventions(
        top_down_learning_rule_convention="paper_methods_dual_and"
    )
    resolved = {
        record.id: _resolved_projection_record(record, conventions=conventions)
        for record in MODELDB_FIRST_ORDER.projections
    }
    assert resolved["modeldb112923.projection.005"].learning_rule == "Dual AND gated"
    assert resolved["modeldb112923.projection.007"].learning_rule == "Dual AND gated"
    assert resolved["modeldb112923.projection.035"].learning_rule == "Postsynaptically gated"


def test_corticoreticular_ampa_delay_override_is_projection_specific() -> None:
    conventions = FirstOrderRuntimeConventions(corticoreticular_ampa_delay_ms=2.0)
    resolved = {
        record.id: _resolved_projection_record(record, conventions=conventions)
        for record in MODELDB_FIRST_ORDER.projections
    }
    assert resolved["modeldb112923.projection.012"].delay_ms == 2.0
    assert resolved["modeldb112923.projection.009"].delay_ms == 3.0
    assert resolved["modeldb112923.projection.003"].delay_ms == 2.0


def test_corticoreticular_ampa_delay_override_must_be_positive() -> None:
    record = MODELDB_FIRST_ORDER.by_id("modeldb112923.projection.012")
    with pytest.raises(ValueError, match="AMPA delay must be positive"):
        _resolved_projection_record(
            record,
            conventions=FirstOrderRuntimeConventions(
                corticoreticular_ampa_delay_ms=0.0
            ),
        )


def test_nonspecific_distal_gaba_source_alternative_is_projection_specific() -> None:
    serialized = MODELDB_FIRST_ORDER.by_id("modeldb112923.projection.049")
    conventions = FirstOrderRuntimeConventions(
        nonspecific_distal_gaba_source_convention=(
            NonspecificDistalGabaSourceConvention.PAPER_SUPPLEMENT_1P5_1_7
        )
    )
    resolved = {
        projection_id: _resolved_projection_record(
            MODELDB_FIRST_ORDER.by_id(projection_id), conventions=conventions
        )
        for projection_id in (
            "modeldb112923.projection.047",
            "modeldb112923.projection.048",
            "modeldb112923.projection.049",
        )
    }
    assert serialized.channel_conductance_mS_cm2 == 1.461
    assert serialized.fall_ms == 4.0
    assert resolved["modeldb112923.projection.049"].channel_conductance_mS_cm2 == 1.5
    assert resolved["modeldb112923.projection.049"].fall_ms == 7.0
    assert resolved["modeldb112923.projection.047"].fall_ms == 4.0
    assert resolved["modeldb112923.projection.048"].fall_ms == 4.0


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
    literal_event = FirstOrderRuntimeConventions(spike_event_rule="literal_previous_sample")
    assert literal_event.fingerprint != classic.fingerprint
    internal_voltage = FirstOrderRuntimeConventions(
        membrane_initialization_convention="kinness_internal_zero",
        calcium_voltage_coordinate="internal_zero_plus_serialized_leak",
    )
    assert internal_voltage.fingerprint != classic.fingerprint
    trn_internal_event = FirstOrderRuntimeConventions(
        trn_spike_event_coordinate="relative_to_soma_leak"
    )
    assert trn_internal_event.fingerprint != classic.fingerprint
    trn_axial_scale = FirstOrderRuntimeConventions(
        trn_soma_proximal_axial_conductance_scale=2.0
    )
    assert trn_axial_scale.fingerprint != classic.fingerprint
    trn_event_offset = FirstOrderRuntimeConventions(
        trn_spike_event_voltage_offset_mV=60.0
    )
    assert trn_event_offset.fingerprint != classic.fingerprint
    trn_event_blend = FirstOrderRuntimeConventions(
        trn_spike_event_proximal_blend_fraction=0.5
    )
    assert trn_event_blend.fingerprint != classic.fingerprint
    nonspecific_event_blend = FirstOrderRuntimeConventions(
        nonspecific_spike_event_proximal_blend_fraction=0.5
    )
    assert nonspecific_event_blend.fingerprint != classic.fingerprint
    nonspecific_split_event = FirstOrderRuntimeConventions(
        nonspecific_spike_event_proximal_blend_fraction=0.5,
        nonspecific_spike_event_release_proximal_blend_fraction=0.0,
    )
    assert nonspecific_split_event.fingerprint != nonspecific_event_blend.fingerprint
    radial_annulus = FirstOrderRuntimeConventions(
        ring_kernel_convention="radial_annulus"
    )
    assert radial_annulus.fingerprint != classic.fingerprint
    targeted_annulus = FirstOrderRuntimeConventions(
        corticoreticular_ring_kernel_convention="radial_annulus"
    )
    assert targeted_annulus.fingerprint != classic.fingerprint
    targeted_annulus_radius = FirstOrderRuntimeConventions(
        corticoreticular_ring_kernel_convention="radial_annulus",
        corticoreticular_ring_peak_radius_scale=2.0,
    )
    assert targeted_annulus_radius.fingerprint != targeted_annulus.fingerprint
    targeted_ampa_delay = FirstOrderRuntimeConventions(
        corticoreticular_ampa_delay_ms=2.0
    )
    assert targeted_ampa_delay.fingerprint != classic.fingerprint
    paper_nonspecific_distal_gaba = FirstOrderRuntimeConventions(
        nonspecific_distal_gaba_source_convention="paper_supplement_1p5_1_7"
    )
    assert paper_nonspecific_distal_gaba.fingerprint != classic.fingerprint


def test_corticoreticular_ring_override_is_projection_specific() -> None:
    conventions = FirstOrderRuntimeConventions(
        corticoreticular_ring_kernel_convention="radial_annulus"
    )
    assert (
        _ring_kernel_convention_for_record(
            "modeldb112923.projection.009", conventions=conventions
        )
        == "radial_annulus"
    )
    assert (
        _ring_kernel_convention_for_record(
            "modeldb112923.projection.012", conventions=conventions
        )
        == "radial_annulus"
    )
    assert (
        _ring_kernel_convention_for_record(
            "modeldb112923.projection.011", conventions=conventions
        )
        == "center_excluded_gaussian"
    )


def test_corticoreticular_ring_radius_override_is_projection_specific() -> None:
    conventions = FirstOrderRuntimeConventions(
        corticoreticular_ring_kernel_convention="radial_annulus",
        corticoreticular_ring_peak_radius_scale=2.0,
    )
    assert _ring_peak_radius_scale_for_record(
        "modeldb112923.projection.009", conventions=conventions
    ) == pytest.approx(2.0)
    assert _ring_peak_radius_scale_for_record(
        "modeldb112923.projection.012", conventions=conventions
    ) == pytest.approx(2.0)
    assert _ring_peak_radius_scale_for_record(
        "modeldb112923.projection.011", conventions=conventions
    ) == pytest.approx(1.0)


def test_corticoreticular_ring_radius_requires_annular_kernel() -> None:
    conventions = FirstOrderRuntimeConventions(
        corticoreticular_ring_peak_radius_scale=2.0
    )
    with pytest.raises(ValueError, match="requires radial_annulus"):
        _ring_peak_radius_scale_for_record(
            "modeldb112923.projection.009", conventions=conventions
        )


def test_trn_event_coordinate_can_follow_kinness_without_changing_relay() -> None:
    conventions = FirstOrderRuntimeConventions(
        trn_spike_event_coordinate="relative_to_soma_leak",
        trn_spike_event_threshold_mV=30.0,
    )
    facts = {fact.canonical_name: fact for fact in first_order_population_facts()}
    relay = first_order_population_parameters(
        facts["thalamic_relay"], conventions=conventions
    )
    trn = first_order_population_parameters(facts["trn"], conventions=conventions)

    assert relay["spike_event_coordinate"] == "absolute_physical"
    assert relay["spike_event_threshold_mV"] == 30.0
    assert trn["spike_event_coordinate"] == "relative_to_soma_leak"
    assert trn["spike_event_threshold_mV"] == 30.0


def test_trn_event_release_voltage_is_scoped_and_validated() -> None:
    facts = {fact.canonical_name: fact for fact in first_order_population_facts()}
    conventions = FirstOrderRuntimeConventions(trn_spike_event_release_mV=-40.0)
    relay = first_order_population_parameters(
        facts["thalamic_relay"], conventions=conventions
    )
    trn = first_order_population_parameters(facts["trn"], conventions=conventions)

    assert relay["spike_event_release_mV"] == 0.0
    assert trn["spike_event_release_mV"] == -40.0
    assert conventions.fingerprint != FirstOrderRuntimeConventions().fingerprint

    with pytest.raises(ValueError, match="release voltage must be finite"):
        first_order_population_parameters(
            facts["trn"],
            conventions=FirstOrderRuntimeConventions(
                trn_spike_event_release_mV=float("nan")
            ),
        )


def test_trn_numeric_event_offset_is_scoped_and_validated() -> None:
    conventions = FirstOrderRuntimeConventions(
        trn_spike_event_voltage_offset_mV=60.0
    )
    facts = {fact.canonical_name: fact for fact in first_order_population_facts()}
    relay = first_order_population_parameters(
        facts["thalamic_relay"], conventions=conventions
    )
    trn = first_order_population_parameters(facts["trn"], conventions=conventions)

    assert "spike_event_voltage_offset_mV" not in relay
    assert trn["spike_event_voltage_offset_mV"] == 60.0

    with pytest.raises(ValueError, match="offset must be finite"):
        first_order_population_parameters(
            facts["trn"],
            conventions=FirstOrderRuntimeConventions(
                trn_spike_event_voltage_offset_mV=float("nan")
            ),
        )


def test_trn_proximal_event_blend_is_scoped_and_validated() -> None:
    conventions = FirstOrderRuntimeConventions(
        trn_spike_event_proximal_blend_fraction=0.5
    )
    facts = {fact.canonical_name: fact for fact in first_order_population_facts()}
    relay = first_order_population_parameters(
        facts["thalamic_relay"], conventions=conventions
    )
    trn = first_order_population_parameters(facts["trn"], conventions=conventions)

    assert "spike_event_proximal_blend_fraction" not in relay
    assert trn["spike_event_proximal_blend_fraction"] == 0.5

    with pytest.raises(ValueError, match="blend must be finite"):
        first_order_population_parameters(
            facts["trn"],
            conventions=FirstOrderRuntimeConventions(
                trn_spike_event_proximal_blend_fraction=float("nan")
            ),
        )


def test_nonspecific_proximal_event_blend_is_scoped_and_validated() -> None:
    conventions = FirstOrderRuntimeConventions(
        nonspecific_spike_event_proximal_blend_fraction=0.5
    )
    facts = {fact.canonical_name: fact for fact in first_order_population_facts()}
    relay = first_order_population_parameters(
        facts["thalamic_relay"], conventions=conventions
    )
    trn = first_order_population_parameters(facts["trn"], conventions=conventions)
    nonspecific = first_order_population_parameters(
        facts["thalamic_nonspecific"], conventions=conventions
    )

    assert "spike_event_proximal_blend_fraction" not in relay
    assert "spike_event_proximal_blend_fraction" not in trn
    assert nonspecific["spike_event_proximal_blend_fraction"] == 0.5

    split = first_order_population_parameters(
        facts["thalamic_nonspecific"],
        conventions=FirstOrderRuntimeConventions(
            nonspecific_spike_event_proximal_blend_fraction=0.5,
            nonspecific_spike_event_release_proximal_blend_fraction=0.0,
        ),
    )
    assert split["spike_event_proximal_blend_fraction"] == 0.5
    assert split["spike_event_release_proximal_blend_fraction"] == 0.0

    with pytest.raises(ValueError, match="nonspecific spike-event proximal blend"):
        first_order_population_parameters(
            facts["thalamic_nonspecific"],
            conventions=FirstOrderRuntimeConventions(
                nonspecific_spike_event_proximal_blend_fraction=1.01
            ),
        )
    with pytest.raises(ValueError, match="release blend requires"):
        first_order_population_parameters(
            facts["thalamic_nonspecific"],
            conventions=FirstOrderRuntimeConventions(
                nonspecific_spike_event_release_proximal_blend_fraction=0.0
            ),
        )
    with pytest.raises(ValueError, match="between zero and one"):
        first_order_population_parameters(
            facts["trn"],
            conventions=FirstOrderRuntimeConventions(
                trn_spike_event_proximal_blend_fraction=1.01
            ),
        )


def test_intrinsic_cell_source_is_explicit_and_does_not_mix_trn_densities() -> None:
    trn = next(fact for fact in first_order_population_facts() if fact.canonical_name == "trn")
    executable = first_order_population_parameters(
        trn,
        conventions=FirstOrderRuntimeConventions(
            intrinsic_cell_convention=IntrinsicCellConvention.MODELDB_112923.value
        ),
    )["cell_spec"]
    paper_parameters = first_order_population_parameters(
        trn,
        conventions=FirstOrderRuntimeConventions(
            intrinsic_cell_convention=IntrinsicCellConvention.PAPER_TABLE3.value
        ),
    )
    paper = paper_parameters["cell_spec"]

    assert executable.soma.g_ca_mS_cm2 == 100
    assert executable.compartment("proximal_dendrite").g_ca_mS_cm2 == 100
    assert executable.soma.g_k_mS_cm2 == 80
    assert paper.soma.g_ca_mS_cm2 is None
    assert paper.compartment("proximal_dendrite").g_ca_mS_cm2 == 10
    assert paper.soma.g_k_mS_cm2 == 100
    assert paper_parameters["e_na_mV"] == 50
    assert paper_parameters["e_k_mV"] == -90
    assert paper_parameters["e_ca_mV"] == 180


def test_source_hybrid_uses_archived_relay_and_table3_for_other_cells() -> None:
    facts = {fact.canonical_name: fact for fact in first_order_population_facts()}
    conventions = FirstOrderRuntimeConventions(
        intrinsic_cell_convention=(
            IntrinsicCellConvention.MODELDB_RELAY_PAPER_TABLE3_OTHERS.value
        )
    )
    relay = first_order_population_parameters(
        facts["thalamic_relay"], conventions=conventions
    )
    trn = first_order_population_parameters(facts["trn"], conventions=conventions)

    assert relay["cell_spec"].name == "modeldb112923_relay"
    assert relay["cell_spec"].compartment("proximal_dendrite").g_ca_mS_cm2 == 0.1
    assert trn["cell_spec"].name == "trn"
    assert trn["cell_spec"].compartment("proximal_dendrite").g_ca_mS_cm2 == 10
    assert relay["e_k_mV"] == facts["thalamic_relay"].e_k_mV
    assert trn["e_k_mV"] == -90


@pytest.mark.parametrize(
    ("potassium", "expected_density", "expected_reversal"),
    (
        (TrnPotassiumConvention.SELECTED_SOURCE, 100.0, -90.0),
        (TrnPotassiumConvention.MODELDB_DENSITY, 80.0, -90.0),
        (TrnPotassiumConvention.MODELDB_REVERSAL, 100.0, -100.0),
        (TrnPotassiumConvention.MODELDB_DENSITY_AND_REVERSAL, 80.0, -100.0),
    ),
)
def test_trn_potassium_source_conflicts_are_independently_selectable(
    potassium: TrnPotassiumConvention,
    expected_density: float,
    expected_reversal: float,
) -> None:
    facts = {fact.canonical_name: fact for fact in first_order_population_facts()}
    conventions = FirstOrderRuntimeConventions(
        intrinsic_cell_convention=(
            IntrinsicCellConvention.MODELDB_RELAY_PAPER_TABLE3_OTHERS.value
        ),
        trn_potassium_convention=potassium.value,
    )

    trn = first_order_population_parameters(facts["trn"], conventions=conventions)
    relay = first_order_population_parameters(
        facts["thalamic_relay"], conventions=conventions
    )

    assert trn["cell_spec"].soma.g_k_mS_cm2 == expected_density
    assert trn["e_k_mV"] == expected_reversal
    assert relay["cell_spec"].soma.g_k_mS_cm2 == 100.0
    assert relay["e_k_mV"] == -100.0


def test_trn_calibrated_potassium_density_is_local_and_explicit() -> None:
    facts = {fact.canonical_name: fact for fact in first_order_population_facts()}
    conventions = FirstOrderRuntimeConventions(
        intrinsic_cell_convention=(
            IntrinsicCellConvention.MODELDB_RELAY_PAPER_TABLE3_OTHERS.value
        ),
        trn_soma_potassium_density_mS_cm2=40.0,
    )
    trn = first_order_population_parameters(facts["trn"], conventions=conventions)
    relay = first_order_population_parameters(
        facts["thalamic_relay"], conventions=conventions
    )
    assert trn["cell_spec"].soma.g_k_mS_cm2 == 40.0
    assert relay["cell_spec"].soma.g_k_mS_cm2 == 100.0
    assert "calibrated_k_40" in trn["cell_spec"].name


def test_trn_calibrated_sodium_density_is_local_and_explicit() -> None:
    facts = {fact.canonical_name: fact for fact in first_order_population_facts()}
    conventions = FirstOrderRuntimeConventions(
        intrinsic_cell_convention=(
            IntrinsicCellConvention.MODELDB_RELAY_PAPER_TABLE3_OTHERS.value
        ),
        trn_soma_sodium_density_mS_cm2=150.0,
    )
    trn = first_order_population_parameters(facts["trn"], conventions=conventions)
    relay = first_order_population_parameters(
        facts["thalamic_relay"], conventions=conventions
    )
    assert trn["cell_spec"].soma.g_na_mS_cm2 == 150.0
    assert relay["cell_spec"].soma.g_na_mS_cm2 == 100.0
    assert "calibrated_na_150" in trn["cell_spec"].name


@pytest.mark.parametrize("density", [float("nan"), 0.0, -1.0])
def test_trn_calibrated_sodium_density_must_be_positive(density: float) -> None:
    facts = {fact.canonical_name: fact for fact in first_order_population_facts()}
    conventions = FirstOrderRuntimeConventions(
        trn_soma_sodium_density_mS_cm2=density
    )
    with pytest.raises(ValueError, match="finite and positive"):
        first_order_population_parameters(facts["trn"], conventions=conventions)


@pytest.mark.parametrize("density", [float("nan"), 0.0, -1.0])
def test_trn_calibrated_potassium_density_must_be_positive(density: float) -> None:
    facts = {fact.canonical_name: fact for fact in first_order_population_facts()}
    conventions = FirstOrderRuntimeConventions(
        trn_soma_potassium_density_mS_cm2=density
    )
    with pytest.raises(ValueError, match="finite and positive"):
        first_order_population_parameters(facts["trn"], conventions=conventions)


def test_trn_calibrated_potassium_density_rejects_source_density_override() -> None:
    facts = {fact.canonical_name: fact for fact in first_order_population_facts()}
    conventions = FirstOrderRuntimeConventions(
        trn_potassium_convention=TrnPotassiumConvention.MODELDB_DENSITY.value,
        trn_soma_potassium_density_mS_cm2=40.0,
    )
    with pytest.raises(ValueError, match="cannot both override"):
        first_order_population_parameters(facts["trn"], conventions=conventions)


@pytest.mark.parametrize(
    ("calcium", "expected_soma_density", "expected_reversal"),
    (
        (TrnCalciumSourceConvention.SELECTED_SOURCE, None, 180.0),
        (TrnCalciumSourceConvention.MODELDB_SOMA_CHANNEL, 100.0, 180.0),
        (TrnCalciumSourceConvention.MODELDB_REVERSAL, None, 120.0),
        (
            TrnCalciumSourceConvention.MODELDB_SOMA_CHANNEL_AND_REVERSAL,
            100.0,
            120.0,
        ),
    ),
)
def test_trn_calcium_source_conflicts_are_independently_selectable(
    calcium: TrnCalciumSourceConvention,
    expected_soma_density: float | None,
    expected_reversal: float,
) -> None:
    facts = {fact.canonical_name: fact for fact in first_order_population_facts()}
    conventions = FirstOrderRuntimeConventions(
        intrinsic_cell_convention=(
            IntrinsicCellConvention.MODELDB_RELAY_PAPER_TABLE3_OTHERS.value
        ),
        trn_calcium_source_convention=calcium.value,
    )

    trn = first_order_population_parameters(facts["trn"], conventions=conventions)
    relay = first_order_population_parameters(
        facts["thalamic_relay"], conventions=conventions
    )

    assert trn["cell_spec"].soma.g_ca_mS_cm2 == expected_soma_density
    assert trn["e_ca_mV"] == expected_reversal
    assert relay["cell_spec"].soma.g_ca_mS_cm2 == 0.1
    assert relay["e_ca_mV"] == facts["thalamic_relay"].e_ca_mV


def test_trn_dendritic_calcium_density_conflict_is_explicit_and_scoped() -> None:
    facts = {fact.canonical_name: fact for fact in first_order_population_facts()}
    conventions = FirstOrderRuntimeConventions(
        intrinsic_cell_convention=(
            IntrinsicCellConvention.MODELDB_RELAY_PAPER_TABLE3_OTHERS.value
        ),
        trn_dendritic_calcium_density_convention=(
            TrnDendriticCalciumDensityConvention.MODELDB_100.value
        ),
    )

    trn = first_order_population_parameters(facts["trn"], conventions=conventions)
    relay = first_order_population_parameters(
        facts["thalamic_relay"], conventions=conventions
    )

    assert trn["cell_spec"].soma.g_ca_mS_cm2 is None
    assert trn["cell_spec"].compartment("proximal_dendrite").g_ca_mS_cm2 == 100.0
    assert trn["cell_spec"].compartment("distal_dendrite").g_ca_mS_cm2 == 100.0
    assert relay["cell_spec"].compartment("proximal_dendrite").g_ca_mS_cm2 == 0.1


def test_behavior_calibrated_trn_dendritic_density_is_explicit_and_validated() -> None:
    facts = {fact.canonical_name: fact for fact in first_order_population_facts()}
    conventions = FirstOrderRuntimeConventions(
        intrinsic_cell_convention=(
            IntrinsicCellConvention.MODELDB_RELAY_PAPER_TABLE3_OTHERS.value
        ),
        trn_dendritic_calcium_density_mS_cm2=40.0,
    )
    trn = first_order_population_parameters(facts["trn"], conventions=conventions)

    assert trn["cell_spec"].compartment("proximal_dendrite").g_ca_mS_cm2 == 40.0
    assert trn["cell_spec"].compartment("distal_dendrite").g_ca_mS_cm2 == 40.0

    with pytest.raises(ValueError, match="finite and positive"):
        first_order_population_parameters(
            facts["trn"],
            conventions=FirstOrderRuntimeConventions(
                trn_dendritic_calcium_density_mS_cm2=0.0
            ),
        )
    with pytest.raises(ValueError, match="cannot both override"):
        first_order_population_parameters(
            facts["trn"],
            conventions=FirstOrderRuntimeConventions(
                trn_dendritic_calcium_density_convention="modeldb_100",
                trn_dendritic_calcium_density_mS_cm2=40.0,
            ),
        )


def test_trn_soma_proximal_axial_scale_is_local_and_validated() -> None:
    facts = {fact.canonical_name: fact for fact in first_order_population_facts()}
    conventions = FirstOrderRuntimeConventions(
        trn_soma_proximal_axial_conductance_scale=2.0
    )
    trn = first_order_population_parameters(facts["trn"], conventions=conventions)
    relay = first_order_population_parameters(
        facts["thalamic_relay"], conventions=conventions
    )

    assert trn["axial_edge_conductance_scales"] == (2.0, 1.0)
    assert "axial_edge_conductance_scales" not in relay

    with pytest.raises(ValueError, match="finite and positive"):
        first_order_population_parameters(
            facts["trn"],
            conventions=FirstOrderRuntimeConventions(
                trn_soma_proximal_axial_conductance_scale=0.0
            ),
        )


def test_category_source_hybrid_also_uses_archived_layer6ii() -> None:
    facts = {fact.canonical_name: fact for fact in first_order_population_facts()}
    conventions = FirstOrderRuntimeConventions(
        intrinsic_cell_convention=(
            IntrinsicCellConvention.MODELDB_RELAY_LAYER6II_PAPER_TABLE3_OTHERS.value
        )
    )
    layer6ii = first_order_population_parameters(
        facts["layer6ii_excitatory_v1"], conventions=conventions
    )["cell_spec"]
    layer23 = first_order_population_parameters(
        facts["layer23_excitatory_v1"], conventions=conventions
    )["cell_spec"]

    assert layer6ii.name == "modeldb112923_layer6ii"
    assert layer6ii.compartment("proximal_dendrite").diameter_mm == 0.008
    assert layer23.name == "layer23_excitatory"


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


def test_nonspecific_calcium_kinetics_can_be_selected_without_changing_trn() -> None:
    facts = {fact.canonical_name: fact for fact in first_order_population_facts()}
    conventions = FirstOrderRuntimeConventions(
        calcium_kinetics_convention=CalciumKineticsConvention.MODELDB_112923.value,
        nonspecific_calcium_kinetics_convention=(
            CalciumKineticsConvention.PAPER_2008.value
        ),
    )

    nonspecific = first_order_population_parameters(
        facts["thalamic_nonspecific"], conventions=conventions
    )
    trn = first_order_population_parameters(facts["trn"], conventions=conventions)

    assert nonspecific["calcium_gate_convention"] == "reciprocal"
    assert trn["calcium_gate_convention"] == "modeldb_reticular_112923"


def test_nonspecific_axial_source_can_be_selected_without_changing_trn() -> None:
    facts = {fact.canonical_name: fact for fact in first_order_population_facts()}
    conventions = FirstOrderRuntimeConventions(
        axial_convention="modeldb_relay_kinness_paper_others",
        nonspecific_axial_convention="kinness_serialized_edge",
    )

    nonspecific = first_order_population_parameters(
        facts["thalamic_nonspecific"], conventions=conventions
    )
    trn = first_order_population_parameters(facts["trn"], conventions=conventions)
    layer4 = first_order_population_parameters(
        facts["layer4_excitatory_v1"], conventions=conventions
    )

    assert nonspecific["axial_convention"] == "kinness_serialized_edge"
    assert trn["axial_convention"] == "paper_literal"
    assert layer4["axial_convention"] == "paper_literal"


def test_nonspecific_intrinsic_source_can_be_selected_without_changing_trn() -> None:
    facts = {fact.canonical_name: fact for fact in first_order_population_facts()}
    conventions = FirstOrderRuntimeConventions(
        intrinsic_cell_convention=(
            IntrinsicCellConvention.MODELDB_RELAY_PAPER_TABLE3_OTHERS.value
        ),
        nonspecific_intrinsic_cell_convention=(
            IntrinsicCellConvention.MODELDB_112923.value
        ),
    )

    nonspecific = first_order_population_parameters(
        facts["thalamic_nonspecific"], conventions=conventions
    )["cell_spec"]
    trn = first_order_population_parameters(facts["trn"], conventions=conventions)[
        "cell_spec"
    ]

    assert nonspecific.name == "modeldb112923_intralaminar"
    assert nonspecific.soma.g_na_mS_cm2 == 50.0
    assert nonspecific.soma.g_k_mS_cm2 == 30.0
    assert trn.name == "trn"
    assert trn.soma.g_na_mS_cm2 == 100.0
    assert trn.soma.g_k_mS_cm2 == 100.0


def test_nonspecific_event_threshold_can_be_selected_without_changing_trn() -> None:
    facts = {fact.canonical_name: fact for fact in first_order_population_facts()}
    conventions = FirstOrderRuntimeConventions(
        nonspecific_spike_event_threshold_mV=-20.0
    )

    nonspecific = first_order_population_parameters(
        facts["thalamic_nonspecific"], conventions=conventions
    )
    trn = first_order_population_parameters(facts["trn"], conventions=conventions)
    relay = first_order_population_parameters(
        facts["thalamic_relay"], conventions=conventions
    )

    assert nonspecific["spike_event_threshold_mV"] == -20.0
    assert trn["spike_event_threshold_mV"] == 30.0
    assert relay["spike_event_threshold_mV"] == 30.0

    with pytest.raises(ValueError, match="nonspecific spike-event threshold"):
        first_order_population_parameters(
            facts["thalamic_nonspecific"],
            conventions=FirstOrderRuntimeConventions(
                nonspecific_spike_event_threshold_mV=float("nan")
            ),
        )


def test_figure6_profile_names_the_source_constrained_runtime() -> None:
    conventions = figure6_runtime_conventions()
    assert conventions.gate_initialization_convention == "steady_state_at_initial_voltage"
    assert conventions.intrinsic_cell_convention == "modeldb_112923"
    assert conventions.calcium_kinetics_convention == "modeldb_112923"
    assert conventions.zero_sensitivity_input_convention == "omit_all_zero"
    assert conventions.spike_event_coordinate == "absolute_physical"
    assert conventions.spike_event_threshold_mV == 30.0
    assert conventions.postsynaptic_learning_coordinate == "absolute_physical"
    assert conventions.postsynaptic_learning_threshold_mV == 30.0
    assert conventions.spike_event_rule == "latched_peak_then_zero"
    assert conventions.membrane_initialization_convention == "physical_leak_voltage"
    assert conventions.calcium_voltage_coordinate == "integrated_voltage"
    assert conventions.specific_capacitance_uF_cm2 == 1.0
    assert conventions.gaussian_weight_convention == "source_peak"
    assert conventions.gaussian_spread_convention == "standard_deviation"
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
    assert np.allclose(getattr(relay.group, f"{port.name}_input_source_count")[:], 1)
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
