import math

import pytest

from smart_robustness.models.table3 import TABLE3_CELLS, get_cell_spec

EXPECTED_COMPARTMENT_COUNTS = {
    "thalamic_relay": 3,
    "thalamic_matrix": 3,
    "thalamic_interneuron": 2,
    "trn": 3,
    "thalamic_nonspecific": 3,
    "layer4_excitatory": 2,
    "layer4_inhibitory": 2,
    "layer23_excitatory": 2,
    "layer23_inhibitory": 2,
    "layer5_excitatory": 3,
    "layer6i_excitatory": 2,
    "layer6ii_excitatory": 3,
}


def test_all_table3_cell_classes_are_transcribed() -> None:
    assert {name: len(spec.compartments) for name, spec in TABLE3_CELLS.items()} == (
        EXPECTED_COMPARTMENT_COUNTS
    )


def test_layer23_soma_density_conversion_matches_published_geometry() -> None:
    soma = get_cell_spec("layer23_excitatory").soma
    expected_area = math.pi * 0.05 * 0.05 / 100
    assert soma.lateral_area_cm2 == pytest.approx(expected_area)
    assert soma.capacitance_pF() == pytest.approx(78.5398163)
    assert soma.conductance_nS("na") == pytest.approx(3926.990817)
    assert soma.conductance_nS("k") == pytest.approx(2356.19449)
    assert soma.conductance_nS("leak") == pytest.approx(3.9269908)


def test_thalamic_calcium_is_compartment_specific() -> None:
    relay = get_cell_spec("thalamic_relay")
    assert relay.soma.g_ca_mS_cm2 is None
    assert relay.compartment("proximal_dendrite").g_ca_mS_cm2 == 10
    assert relay.compartment("distal_dendrite").g_ca_mS_cm2 == 10
    assert get_cell_spec("thalamic_nonspecific").compartment("proximal_dendrite").g_ca_mS_cm2 == 250


def test_layer5_distal_dendrite_has_active_na_and_k_channels() -> None:
    distal = get_cell_spec("layer5_excitatory").compartment("distal_dendrite")
    assert distal.g_na_mS_cm2 == 50
    assert distal.g_k_mS_cm2 == 30


def test_missing_channel_converts_to_zero_total_conductance() -> None:
    dendrite = get_cell_spec("layer4_excitatory").compartment("proximal_dendrite")
    assert dendrite.g_na_mS_cm2 is None
    assert dendrite.conductance_nS("na") == 0


def test_unknown_cell_and_channel_fail_loudly() -> None:
    with pytest.raises(ValueError, match="unknown SMART Table 3 cell"):
        get_cell_spec("not_a_cell")
    with pytest.raises(ValueError, match="unknown channel"):
        get_cell_spec("trn").soma.conductance_nS("chloride")
