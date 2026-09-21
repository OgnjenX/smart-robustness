"""Conservative scaffold for staged cortical inhibitory-routing studies."""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from typing import Any

from ..modeldb_projections import MODELDB_FIRST_ORDER


class InhibitoryRoutingMode(StrEnum):
    """Registered modes admitted before biological routing is implemented."""

    LEGACY_AGGREGATE = "legacy_aggregate"


REGISTERED_CORTICAL_INHIBITORY_POPULATIONS = frozenset(
    {"layer23_inhibitory_v1", "layer4_inhibitory_v1"}
)
REGISTERED_INHIBITORY_PROJECTION_IDS = frozenset(
    {
        "modeldb112923.projection.022",
        "modeldb112923.projection.026",
        "modeldb112923.projection.027",
        "modeldb112923.projection.028",
        "modeldb112923.projection.029",
        "modeldb112923.projection.030",
        "modeldb112923.projection.031",
        "modeldb112923.projection.036",
        "modeldb112923.projection.039",
        "modeldb112923.projection.040",
        "modeldb112923.projection.041",
    }
)


def cortical_inhibitory_projection_ids() -> frozenset[str]:
    """Return the source-derived projection inventory touching cortical INs."""

    observed = {
        record.id
        for record in MODELDB_FIRST_ORDER.projections
        if record.source_population in REGISTERED_CORTICAL_INHIBITORY_POPULATIONS
        or record.target_population in REGISTERED_CORTICAL_INHIBITORY_POPULATIONS
    }
    if observed != REGISTERED_INHIBITORY_PROJECTION_IDS:
        missing = sorted(REGISTERED_INHIBITORY_PROJECTION_IDS - observed)
        unexpected = sorted(observed - REGISTERED_INHIBITORY_PROJECTION_IDS)
        raise RuntimeError(
            "first-order inhibitory projection inventory drifted: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return frozenset(observed)


def make_inhibitory_routing_sector_builder(
    *,
    mode: InhibitoryRoutingMode | str,
    base_builder: Callable[..., Any],
) -> Callable[..., Any]:
    """Wrap a sector builder without changing the legacy aggregate network.

    This deliberately supports only the exact-null mode. Future biological
    routing modes require separate preregistration, implementation and sealing.
    """

    selected = InhibitoryRoutingMode(mode)
    if selected is not InhibitoryRoutingMode.LEGACY_AGGREGATE:
        raise ValueError(f"unregistered inhibitory routing mode: {selected}")
    expected = cortical_inhibitory_projection_ids()

    def builder(*args: Any, **kwargs: Any):
        if args:
            raise TypeError("the inhibitory-routing builder accepts keyword arguments only")
        sector = base_builder(**kwargs)
        missing = expected - set(sector.projections)
        if missing:
            raise RuntimeError(
                "legacy aggregate sector lacks registered inhibitory projections: "
                f"{sorted(missing)}"
            )
        return sector

    return builder
