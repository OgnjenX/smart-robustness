"""Typed axial-coupling specifications for multicompartment SMART cells.

Grossberg and Versace (2008) describe an axial term using the geometry of the
compartment receiving the current, while Table 3 labels the associated value
as axial resistance.  ADR 0001 therefore keeps two interpretations available:

``paper_literal``
    Apply the closest dimensionally consistent reading of Equation 2 to each
    receiving compartment independently.  Directional conductances can differ.

``symmetric_cable``
    Join the axial resistances of two half-compartments in series.  The edge has
    one reciprocal conductance and therefore conserves total axial current.

``kinness_2008``
    Implement Equation 7 of the contemporaneous KInNeSS framework paper.  It
    retains KInNeSS's directional, receiving-compartment conductance density
    and includes the geometry of both compartments.

No convention is selected by default.  The classic SMART profile must make and
record that decision only after the validation described in ADR 0001.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise

from .table3 import TABLE3_CELLS, CellSpec, CompartmentSpec

MM_TO_CM = 0.1
NS_PER_MILLISIEMENS = 1_000_000.0


class AxialConvention(StrEnum):
    """Named interpretations retained by ADR 0001."""

    PAPER_LITERAL = "paper_literal"
    SYMMETRIC_CABLE = "symmetric_cable"
    KINNESS_2008 = "kinness_2008"
    KINNESS_SERIALIZED_EDGE = "kinness_serialized_edge"


@dataclass(frozen=True, slots=True)
class AxialEndpointSpec:
    """Validated Table 3 geometry and derived values for one edge endpoint."""

    compartment_name: str
    diameter_cm: float
    length_cm: float
    axial_resistivity_kohm_cm: float
    lateral_area_cm2: float
    half_resistance_kohm: float


@dataclass(frozen=True, slots=True)
class AxialEdgeSpec:
    """Axial coupling between two adjacent, ordered cell compartments.

    ``near`` and ``far`` follow the compartment order in Table 3 (for example,
    soma then proximal dendrite).  Conductance densities are suitable for the
    compartmental current-density equations, while total conductances make the
    conservation properties of an edge explicit.
    """

    cell_name: str
    near: AxialEndpointSpec
    far: AxialEndpointSpec
    convention: AxialConvention
    conductance_into_near_nS: float
    conductance_into_far_nS: float
    conductance_density_into_near_mS_cm2: float
    conductance_density_into_far_mS_cm2: float

    def currents_pA(self, near_voltage_mV: float, far_voltage_mV: float) -> tuple[float, float]:
        """Return currents entering ``near`` and ``far`` in pA.

        The conversion is direct because ``nS * mV = pA``.  Positive current
        enters an endpoint.  Symmetric-cable currents are equal and opposite;
        paper-literal currents need not be because its directional densities
        are derived independently from the receiving compartments.
        """

        _require_finite(near_voltage_mV, "near voltage")
        _require_finite(far_voltage_mV, "far voltage")
        voltage_far_minus_near_mV = far_voltage_mV - near_voltage_mV
        return (
            self.conductance_into_near_nS * voltage_far_minus_near_mV,
            -self.conductance_into_far_nS * voltage_far_minus_near_mV,
        )


def _require_finite(value: float, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number, got {value!r}")


def _require_positive(value: float, label: str) -> None:
    _require_finite(value, label)
    if value <= 0:
        raise ValueError(f"{label} must be positive, got {value!r}")


def _coerce_convention(convention: AxialConvention | str) -> AxialConvention:
    try:
        return AxialConvention(convention)
    except (TypeError, ValueError) as error:
        choices = tuple(item.value for item in AxialConvention)
        raise ValueError(
            f"unknown axial convention {convention!r}; expected one of {choices}"
        ) from error


def _endpoint(
    compartment: CompartmentSpec,
    context: str,
    *,
    axial_resistance_kohm_cm: float | None = None,
) -> AxialEndpointSpec:
    if not isinstance(compartment.name, str) or not compartment.name.strip():
        raise ValueError(f"{context}: compartment name must be non-empty")

    _require_positive(compartment.diameter_mm, f"{context} diameter_mm")
    _require_positive(compartment.length_mm, f"{context} length_mm")
    axial_resistance = (
        compartment.axial_resistance_kohm_cm
        if axial_resistance_kohm_cm is None
        else axial_resistance_kohm_cm
    )
    _require_positive(axial_resistance, f"{context} axial_resistance_kohm_cm")

    diameter_cm = compartment.diameter_mm * MM_TO_CM
    length_cm = compartment.length_mm * MM_TO_CM
    lateral_area_cm2 = math.pi * diameter_cm * length_cm
    cross_section_cm2 = math.pi * (diameter_cm / 2.0) ** 2
    half_resistance_kohm = axial_resistance * (length_cm / 2.0) / cross_section_cm2

    _require_positive(lateral_area_cm2, f"{context} lateral area")
    _require_positive(half_resistance_kohm, f"{context} half resistance")
    return AxialEndpointSpec(
        compartment_name=compartment.name,
        diameter_cm=diameter_cm,
        length_cm=length_cm,
        axial_resistivity_kohm_cm=axial_resistance,
        lateral_area_cm2=lateral_area_cm2,
        half_resistance_kohm=half_resistance_kohm,
    )


def _paper_literal_density_mS_cm2(endpoint: AxialEndpointSpec) -> float:
    """Equation 2 density: D / (4 rho L^2), with all lengths in cm."""

    density = endpoint.diameter_cm / (
        4.0 * endpoint.axial_resistivity_kohm_cm * endpoint.length_cm**2
    )
    _require_positive(density, f"{endpoint.compartment_name} paper-literal density")
    return density


def _total_nS(density_mS_cm2: float, area_cm2: float) -> float:
    conductance_nS = density_mS_cm2 * area_cm2 * NS_PER_MILLISIEMENS
    _require_positive(conductance_nS, "axial total conductance")
    return conductance_nS


def _paper_literal_edge(
    cell_name: str,
    near: AxialEndpointSpec,
    far: AxialEndpointSpec,
) -> AxialEdgeSpec:
    near_density = _paper_literal_density_mS_cm2(near)
    far_density = _paper_literal_density_mS_cm2(far)
    return AxialEdgeSpec(
        cell_name=cell_name,
        near=near,
        far=far,
        convention=AxialConvention.PAPER_LITERAL,
        conductance_into_near_nS=_total_nS(near_density, near.lateral_area_cm2),
        conductance_into_far_nS=_total_nS(far_density, far.lateral_area_cm2),
        conductance_density_into_near_mS_cm2=near_density,
        conductance_density_into_far_mS_cm2=far_density,
    )


def _kinness_density_mS_cm2(
    receiving: AxialEndpointSpec,
    neighboring: AxialEndpointSpec,
) -> float:
    """KInNeSS (Versace et al., 2008) Equation 7.

    The paper denotes the pair-specific axial conductance by ``g_A``.  SMART
    Table 3 instead serializes an axial resistivity for each compartment, so
    the directional implementation uses the reciprocal resistivity of the
    receiving endpoint.  This choice is explicit and testable.
    """

    axial_conductivity_mS_per_cm = 1.0 / receiving.axial_resistivity_kohm_cm
    geometry = (
        receiving.diameter_cm**2 / receiving.length_cm
        + neighboring.diameter_cm**2 / neighboring.length_cm
    )
    density = (
        axial_conductivity_mS_per_cm
        * geometry
        / (8.0 * receiving.diameter_cm * receiving.length_cm)
    )
    _require_positive(density, f"{receiving.compartment_name} KInNeSS density")
    return density


def _kinness_edge(
    cell_name: str,
    near: AxialEndpointSpec,
    far: AxialEndpointSpec,
) -> AxialEdgeSpec:
    near_density = _kinness_density_mS_cm2(near, far)
    far_density = _kinness_density_mS_cm2(far, near)
    return AxialEdgeSpec(
        cell_name=cell_name,
        near=near,
        far=far,
        convention=AxialConvention.KINNESS_2008,
        conductance_into_near_nS=_total_nS(near_density, near.lateral_area_cm2),
        conductance_into_far_nS=_total_nS(far_density, far.lateral_area_cm2),
        conductance_density_into_near_mS_cm2=near_density,
        conductance_density_into_far_mS_cm2=far_density,
    )


def _symmetric_cable_edge(
    cell_name: str,
    near: AxialEndpointSpec,
    far: AxialEndpointSpec,
) -> AxialEdgeSpec:
    edge_resistance_kohm = near.half_resistance_kohm + far.half_resistance_kohm
    _require_positive(edge_resistance_kohm, f"{cell_name} edge resistance")

    # 1 / kOhm is mS; multiplying by 1e6 converts mS to nS.
    edge_conductance_nS = NS_PER_MILLISIEMENS / edge_resistance_kohm
    _require_positive(edge_conductance_nS, f"{cell_name} edge conductance")
    near_density = edge_conductance_nS / NS_PER_MILLISIEMENS / near.lateral_area_cm2
    far_density = edge_conductance_nS / NS_PER_MILLISIEMENS / far.lateral_area_cm2
    _require_positive(near_density, f"{near.compartment_name} edge density")
    _require_positive(far_density, f"{far.compartment_name} edge density")

    return AxialEdgeSpec(
        cell_name=cell_name,
        near=near,
        far=far,
        convention=AxialConvention.SYMMETRIC_CABLE,
        conductance_into_near_nS=edge_conductance_nS,
        conductance_into_far_nS=edge_conductance_nS,
        conductance_density_into_near_mS_cm2=near_density,
        conductance_density_into_far_mS_cm2=far_density,
    )


def build_axial_edges(
    cell: CellSpec,
    convention: AxialConvention | str,
) -> tuple[AxialEdgeSpec, ...]:
    """Build ordered adjacent-compartment edges for one SMART cell class."""

    resolved_convention = _coerce_convention(convention)
    if not isinstance(cell.name, str) or not cell.name.strip():
        raise ValueError("cell name must be non-empty")
    if len(cell.compartments) < 2:
        raise ValueError(f"{cell.name}: axial coupling requires at least two compartments")

    names = tuple(compartment.name for compartment in cell.compartments)
    if len(names) != len(set(names)):
        raise ValueError(f"{cell.name}: compartment names must be unique")
    if resolved_convention is AxialConvention.KINNESS_SERIALIZED_EDGE:
        edges: list[AxialEdgeSpec] = []
        for near_compartment, far_compartment in pairwise(cell.compartments):
            # KInNeSS serializes ``inpResistance`` on the child compartment,
            # representing the parent-child connection rather than a root
            # membrane property. Use that one value in both current directions.
            resistance = far_compartment.axial_resistance_kohm_cm
            near = _endpoint(
                near_compartment,
                f"{cell.name}.{near_compartment.name or '<unnamed>'}",
                axial_resistance_kohm_cm=resistance,
            )
            far = _endpoint(
                far_compartment,
                f"{cell.name}.{far_compartment.name or '<unnamed>'}",
                axial_resistance_kohm_cm=resistance,
            )
            edges.append(_kinness_edge(cell.name, near, far))
        return tuple(edges)
    endpoints = tuple(
        _endpoint(compartment, f"{cell.name}.{compartment.name or '<unnamed>'}")
        for compartment in cell.compartments
    )

    edges: list[AxialEdgeSpec] = []
    for near, far in pairwise(endpoints):
        if resolved_convention is AxialConvention.PAPER_LITERAL:
            edges.append(_paper_literal_edge(cell.name, near, far))
        elif resolved_convention is AxialConvention.SYMMETRIC_CABLE:
            edges.append(_symmetric_cable_edge(cell.name, near, far))
        else:
            edges.append(_kinness_edge(cell.name, near, far))
    return tuple(edges)


def build_table3_axial_edges(
    convention: AxialConvention | str,
) -> tuple[AxialEdgeSpec, ...]:
    """Build all adjacent edges for all 12 Table 3 SMART cell classes."""

    resolved_convention = _coerce_convention(convention)
    return tuple(
        edge
        for cell in TABLE3_CELLS.values()
        for edge in build_axial_edges(cell, resolved_convention)
    )
