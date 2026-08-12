import pytest

from smart_robustness.modeldb_projections import MODELDB_FULL
from smart_robustness.models.modeldb112923 import (
    AHP_ACH_SHA256,
    ARCHIVE_SHA256,
    CA_REBOUND_SHA256,
    FIGURE8_SOURCE_FACTS,
    FIRST_ORDER_POPULATIONS,
    FIRST_ORDER_PROJECTION_COUNT,
    SECOND_ORDER_POPULATIONS,
    SMART_NML_SHA256,
    figure8_relay_spec,
    first_order_population_facts,
    first_order_structural_counts,
    full_network_structural_counts,
    second_order_population_facts,
)


def test_official_backup_hashes_are_pinned() -> None:
    for digest in (ARCHIVE_SHA256, SMART_NML_SHA256, CA_REBOUND_SHA256, AHP_ACH_SHA256):
        assert len(digest) == 64
        int(digest, 16)


def test_figure8_relay_requires_unserialized_leak_candidate() -> None:
    with pytest.raises(ValueError, match="explicit positive candidate"):
        figure8_relay_spec(leak_density_mS_cm2=0)
    relay = figure8_relay_spec(leak_density_mS_cm2=0.1)
    assert relay.soma.diameter_mm == pytest.approx(0.2)
    assert relay.soma.g_na_mS_cm2 == pytest.approx(50)
    assert all(c.g_ca_mS_cm2 == 250 for c in relay.compartments)
    assert FIGURE8_SOURCE_FACTS.missing_leak_density


def test_first_order_structural_counts_match_smart_nml_audit() -> None:
    assert len(FIRST_ORDER_POPULATIONS) == 12
    assert first_order_structural_counts() == (812, 1950)
    assert FIRST_ORDER_PROJECTION_COUNT == 56


def test_first_order_source_cells_cover_all_populations_and_compartments() -> None:
    facts = first_order_population_facts()
    assert len(facts) == 12
    assert sum(f.shape[0] * f.shape[1] for f in facts) == 812
    assert sum(len(f.cell.compartments) * f.shape[0] * f.shape[1] for f in facts) == 1950
    relay = facts[0]
    assert relay.source_name == "Relay"
    assert relay.cell.soma.diameter_mm == 0.05
    assert relay.cell.compartment("distal_dendrite").axial_resistance_kohm_cm == 8.2
    trn = facts[1]
    assert trn.calcium_gate_convention == "modeldb_reticular_112923"
    layer5 = facts[2]
    assert layer5.ahp_density_mS_cm2 == 0.4
    assert (layer5.ahp_rise_ms, layer5.ahp_fall_ms) == (5, 20)
    assert (layer5.depletion_epsilon, layer5.depletion_recovery_ms) == (0.5, 100)
    layer6i = facts[4]
    assert (layer6i.depletion_epsilon, layer6i.depletion_recovery_ms) == (1.0, 400)


def test_second_order_source_cells_cover_pulvinar_v2_and_preserve_differences() -> None:
    facts = second_order_population_facts()
    assert len(SECOND_ORDER_POPULATIONS) == len(facts) == 12
    assert full_network_structural_counts() == (1624, 3900)
    assert {fact.canonical_name for fact in facts} == {
        "trn_v2", "thalamic_relay_v2", "layer6ii_excitatory_v2",
        "layer5_excitatory_v2", "layer6i_excitatory_v2", "layer23_excitatory_v2",
        "layer4_excitatory_v2", "layer23_inhibitory_v2", "thalamic_interneuron_v2",
        "thalamic_nonspecific_v2", "thalamic_matrix_v2", "layer4_inhibitory_v2",
    }
    layer5 = next(fact for fact in facts if fact.canonical_name == "layer5_excitatory_v2")
    assert layer5.cell.soma.e_leak_mV == -72
    assert layer5.cell.compartment("proximal_dendrite").axial_resistance_kohm_cm == 5


def _channel_by_token(compartment: dict, token: str) -> dict | None:
    return next(
        (
            channel
            for channel in compartment["intrinsic_channels"]
            if token.lower() in channel["name"].lower()
        ),
        None,
    )


def _assert_optional_approx(actual: float | None, expected: float | None) -> None:
    if expected is None:
        assert actual is None
    else:
        assert actual == pytest.approx(expected)


def test_all_24_runtime_cell_specs_match_extracted_smart_nml_intrinsics() -> None:
    """Prove the runtime transcription against independently extracted XML facts."""

    runtime = {
        fact.canonical_name: fact
        for fact in first_order_population_facts() + second_order_population_facts()
    }
    extracted = {
        population["canonical_name"]: population
        for population in MODELDB_FULL.intrinsic_populations
    }
    assert runtime.keys() == extracted.keys()
    for canonical_name, fact in runtime.items():
        source_compartments = extracted[canonical_name]["compartments"]
        assert len(fact.cell.compartments) == len(source_compartments)
        for compartment, source in zip(fact.cell.compartments, source_compartments, strict=True):
            assert compartment.name == source["name"]
            assert compartment.diameter_mm == pytest.approx(source["diameter_cm"] * 10)
            assert compartment.length_mm == pytest.approx(source["length_cm"] * 10)
            assert compartment.e_leak_mV == pytest.approx(source["leak_reversal_mV"])
            assert compartment.g_leak_mS_cm2 == pytest.approx(
                source["leak_conductance_mS_cm2"]
            )
            if source["axial_resistance_kohm_cm"] is not None:
                assert compartment.axial_resistance_kohm_cm == pytest.approx(
                    source["axial_resistance_kohm_cm"]
                )
            for token, runtime_density in (
                ("Na (", compartment.g_na_mS_cm2),
                ("K_dr", compartment.g_k_mS_cm2),
                ("Ca++", compartment.g_ca_mS_cm2),
            ):
                channel = _channel_by_token(source, token)
                source_density = None if channel is None else channel["conductance_mS_cm2"]
                _assert_optional_approx(runtime_density, source_density)
                if channel is not None:
                    expected_reversal = {
                        "Na (": fact.e_na_mV,
                        "K_dr": fact.e_k_mV,
                        "Ca++": fact.e_ca_mV,
                    }[token]
                    assert expected_reversal == pytest.approx(channel["reversal_mV"])
        soma = source_compartments[0]
        _assert_optional_approx(fact.depletion_epsilon, soma["depletion_rate"])
        _assert_optional_approx(fact.depletion_recovery_ms, soma["recovery_ms"])
        ahp = _channel_by_token(soma, "AHP")
        if ahp is None:
            assert fact.ahp_density_mS_cm2 is None
        else:
            gate = ahp["gates"][0]["attributes"]
            assert fact.ahp_density_mS_cm2 == pytest.approx(ahp["conductance_mS_cm2"])
            assert fact.ahp_reversal_mV == pytest.approx(ahp["reversal_mV"])
            assert fact.ahp_rise_ms == pytest.approx(float(gate["tau_r"]))
            assert fact.ahp_fall_ms == pytest.approx(float(gate["tau_f"]))


def test_v2_intrinsic_xml_is_homologous_except_for_layer5_leak_and_axial_values() -> None:
    extracted = {
        population["canonical_name"]: population
        for population in MODELDB_FULL.intrinsic_populations
    }
    pairs = (
        ("trn", "trn_v2"),
        ("thalamic_relay", "thalamic_relay_v2"),
        ("layer6ii_excitatory_v1", "layer6ii_excitatory_v2"),
        ("layer6i_excitatory_v1", "layer6i_excitatory_v2"),
        ("layer23_excitatory_v1", "layer23_excitatory_v2"),
        ("layer4_excitatory_v1", "layer4_excitatory_v2"),
        ("layer23_inhibitory_v1", "layer23_inhibitory_v2"),
        ("thalamic_interneuron", "thalamic_interneuron_v2"),
        ("thalamic_nonspecific", "thalamic_nonspecific_v2"),
        ("thalamic_matrix", "thalamic_matrix_v2"),
        ("layer4_inhibitory_v1", "layer4_inhibitory_v2"),
    )
    for first, second in pairs:
        assert extracted[first]["compartments"] == extracted[second]["compartments"]

    v1 = extracted["layer5_excitatory_v1"]["compartments"]
    v2 = extracted["layer5_excitatory_v2"]["compartments"]
    for v1_compartment, v2_compartment in zip(v1, v2, strict=True):
        differing = {
            key
            for key in v1_compartment
            if v1_compartment[key] != v2_compartment[key]
        }
        expected = {"leak_reversal_mV"}
        if v1_compartment["axial_resistance_kohm_cm"] is not None:
            expected.add("axial_resistance_kohm_cm")
        assert differing == expected
