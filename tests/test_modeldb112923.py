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
