from __future__ import annotations

import pytest

from smart_robustness.models.inhibitory_routing import (
    REGISTERED_INHIBITORY_PROJECTION_IDS,
    InhibitoryRoutingMode,
    cortical_inhibitory_projection_ids,
    make_inhibitory_routing_sector_builder,
)


def test_source_inventory_is_exact() -> None:
    assert cortical_inhibitory_projection_ids() == REGISTERED_INHIBITORY_PROJECTION_IDS


def test_legacy_wrapper_returns_sector_without_mutation() -> None:
    marker = object()
    sector = type(
        "Sector",
        (),
        {"projections": {projection_id: marker for projection_id in REGISTERED_INHIBITORY_PROJECTION_IDS}},
    )()
    calls = []

    def base_builder(**kwargs):
        calls.append(kwargs)
        return sector

    builder = make_inhibitory_routing_sector_builder(
        mode=InhibitoryRoutingMode.LEGACY_AGGREGATE,
        base_builder=base_builder,
    )
    assert builder(example=3) is sector
    assert calls == [{"example": 3}]
    assert all(value is marker for value in sector.projections.values())


def test_legacy_wrapper_rejects_positional_arguments() -> None:
    builder = make_inhibitory_routing_sector_builder(
        mode="legacy_aggregate",
        base_builder=lambda **_: None,
    )
    with pytest.raises(TypeError, match="keyword arguments"):
        builder(1)


def test_legacy_wrapper_fails_closed_on_missing_projection() -> None:
    sector = type("Sector", (), {"projections": {}})()
    builder = make_inhibitory_routing_sector_builder(
        mode="legacy_aggregate",
        base_builder=lambda **_: sector,
    )
    with pytest.raises(RuntimeError, match="lacks registered inhibitory projections"):
        builder()


def test_unregistered_mode_is_rejected() -> None:
    with pytest.raises(ValueError):
        make_inhibitory_routing_sector_builder(
            mode="pv_like",
            base_builder=lambda **_: None,
        )
