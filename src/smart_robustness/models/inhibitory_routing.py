"""Conservative scaffold for staged cortical inhibitory-routing studies."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
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
PROJECTION036_ID = "modeldb112923.projection.036"
LAYER4_EXCITATORY_SUFFIX = "layer4_excitatory_v1"


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


def retarget_projection036_to_soma(params: dict[str, Any]) -> dict[str, Any]:
    """Copy L4 pyramidal parameters and change only projection 036's target."""

    ports = params.get("synaptic_ports")
    if not isinstance(ports, tuple):
        raise TypeError("layer-4 excitatory synaptic_ports must be a tuple")
    matches = [index for index, port in enumerate(ports) if port.record_id == PROJECTION036_ID]
    if len(matches) != 1:
        raise RuntimeError("projection 036 must occur exactly once")
    index = matches[0]
    port = ports[index]
    if port.compartment != "proximal_dendrite":
        raise RuntimeError("projection 036 no longer has its registered proximal target")
    cell = params.get("cell_spec")
    if cell is None:
        raise RuntimeError("layer-4 excitatory parameters require an explicit cell spec")
    soma = cell.compartment("soma")
    proximal = cell.compartment("proximal_dendrite")
    if soma.lateral_area_cm2 != proximal.lateral_area_cm2:
        raise RuntimeError("projection 036 somatic retargeting no longer conserves area exactly")
    transformed_ports = list(ports)
    transformed_ports[index] = replace(port, compartment="soma")
    transformed = dict(params)
    transformed["synaptic_ports"] = tuple(transformed_ports)
    return transformed


def make_layer4_projection036_somatic_population_factory(
    *, base_factory: Callable[..., Any]
) -> Callable[..., Any]:
    """Return the preregistered one-factor L4 PV-like routing transform."""

    def factory(*args: Any, **kwargs: Any):
        if args:
            raise TypeError("the PV-like population factory accepts keyword arguments only")
        name = kwargs.get("name")
        params = kwargs.get("params")
        if not isinstance(name, str) or not isinstance(params, dict):
            raise TypeError("population name and parameter mapping are required")
        if name.endswith(LAYER4_EXCITATORY_SUFFIX):
            kwargs = dict(kwargs)
            kwargs["params"] = retarget_projection036_to_soma(params)
        return base_factory(**kwargs)

    return factory
