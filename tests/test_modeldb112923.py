import pytest

from smart_robustness.models.modeldb112923 import (
    AHP_ACH_SHA256,
    ARCHIVE_SHA256,
    CA_REBOUND_SHA256,
    FIGURE8_SOURCE_FACTS,
    FIRST_ORDER_POPULATIONS,
    FIRST_ORDER_PROJECTION_COUNT,
    SMART_NML_SHA256,
    figure8_relay_spec,
    first_order_population_facts,
    first_order_structural_counts,
)


def test_official_backup_hashes_are_pinned() -> None:
    for digest in (ARCHIVE_SHA256, SMART_NML_SHA256, CA_REBOUND_SHA256, AHP_ACH_SHA256):
        assert len(digest) == 64
        int(digest, 16)


def test_figure8_relay_requires_unserialized_leak_candidate() -> None:
    with pytest.raises(ValueError, match="explicit positive candidate"):
        figure8_relay_spec(leak_density_mS_cm2=0)
    relay = figure8_relay_spec(leak_density_mS_cm2=0.1)
    assert relay.soma.diameter_mm == pytest.approx(0.02)
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
    assert relay.cell.soma.diameter_mm == 0.005
    assert relay.cell.compartment("distal_dendrite").axial_resistance_kohm_cm == 8.2
    layer5 = facts[2]
    assert layer5.ahp_density_mS_cm2 == 0.4
    assert (layer5.ahp_rise_ms, layer5.ahp_fall_ms) == (5, 20)
