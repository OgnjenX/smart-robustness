"""Conserved one-compartment classic-HH intervention for cortical SMART cells."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, TypeVar

from ..synapses import kinness_gap_total_conductance_nS
from .adapter import SmartPopulationAdapter, validate_smart_population_adapter
from .compartmental_hh import create_compartmental_hh_population
from .ports import (
    ExternalInputPortSpec,
    GapJunctionPortSpec,
    InjectionPortSpec,
    SynapticPortSpec,
)
from .selective import CORTICAL_CELL_CLASSES
from .table3 import CellSpec, CompartmentSpec

POINT_HH_INTERVENTION = "cortical_point_hh_conserved_v1"
_PortWithConductance = TypeVar(
    "_PortWithConductance",
    SynapticPortSpec,
    ExternalInputPortSpec,
)


def _total_density_weighted(
    cell: CellSpec,
    attribute: str,
) -> tuple[float, bool]:
    total = 0.0
    present = False
    for compartment in cell.compartments:
        value = getattr(compartment, attribute)
        if value is not None:
            present = True
            total += float(value) * compartment.lateral_area_cm2
    return total, present


def collapse_cell_spec_to_conserved_point(cell: CellSpec) -> CellSpec:
    """Collapse a multicompartment cell while conserving registered totals."""

    if len(cell.compartments) < 2:
        raise ValueError("point-HH collapse requires a multicompartment source cell")
    total_area_cm2 = sum(
        compartment.lateral_area_cm2 for compartment in cell.compartments
    )
    if not math.isfinite(total_area_cm2) or total_area_cm2 <= 0:
        raise ValueError("source cell must have a finite positive membrane area")

    total_leak_density_area, _ = _total_density_weighted(cell, "g_leak_mS_cm2")
    if total_leak_density_area <= 0:
        raise ValueError("source cell must have positive total leak conductance")
    leak_reversal_mV = sum(
        compartment.g_leak_mS_cm2
        * compartment.lateral_area_cm2
        * compartment.e_leak_mV
        for compartment in cell.compartments
    ) / total_leak_density_area

    def collapsed_density(attribute: str) -> float | None:
        total, present = _total_density_weighted(cell, attribute)
        return total / total_area_cm2 if present else None

    diameter_mm = cell.soma.diameter_mm
    if not math.isfinite(diameter_mm) or diameter_mm <= 0:
        raise ValueError("source soma must have a finite positive diameter")
    length_mm = total_area_cm2 * 100.0 / (math.pi * diameter_mm)
    point = CompartmentSpec(
        name="soma",
        diameter_mm=diameter_mm,
        length_mm=length_mm,
        axial_resistance_kohm_cm=cell.soma.axial_resistance_kohm_cm,
        e_leak_mV=leak_reversal_mV,
        g_leak_mS_cm2=total_leak_density_area / total_area_cm2,
        g_na_mS_cm2=collapsed_density("g_na_mS_cm2"),
        g_k_mS_cm2=collapsed_density("g_k_mS_cm2"),
        g_ca_mS_cm2=collapsed_density("g_ca_mS_cm2"),
    )
    return CellSpec(name=f"{cell.name}_point_conserved", compartments=(point,))


def _retarget_conductance_port(
    port: _PortWithConductance,
    *,
    source_cell: CellSpec,
    point_cell: CellSpec,
) -> _PortWithConductance:
    source_area = source_cell.compartment(port.compartment).lateral_area_cm2
    point_area = point_cell.soma.lateral_area_cm2
    return replace(
        port,
        compartment="soma",
        conductance_density_mS_cm2=(
            port.conductance_density_mS_cm2 * source_area / point_area
        ),
    )


def _retarget_gap_port(
    port: GapJunctionPortSpec,
    *,
    source_cell: CellSpec,
    point_cell: CellSpec,
) -> GapJunctionPortSpec:
    source = source_cell.compartment(port.compartment)
    old_total_nS = kinness_gap_total_conductance_nS(
        port.conductance_density_mS_cm2,
        diameter_mm=source.diameter_mm,
        length_mm=source.length_mm,
    )
    point_unit_total_nS = kinness_gap_total_conductance_nS(
        1.0,
        diameter_mm=point_cell.soma.diameter_mm,
        length_mm=point_cell.soma.length_mm,
    )
    return replace(
        port,
        compartment="soma",
        conductance_density_mS_cm2=old_total_nS / point_unit_total_nS,
    )


def _retarget_injection_port(
    port: InjectionPortSpec,
    *,
    source_cell: CellSpec,
    point_cell: CellSpec,
) -> InjectionPortSpec:
    source_area = source_cell.compartment(port.compartment).lateral_area_cm2
    point_area = point_cell.soma.lateral_area_cm2
    scale = source_area / point_area
    return replace(
        port,
        compartment="soma",
        sensitivities_pA_cm2=tuple(
            sensitivity * scale for sensitivity in port.sensitivities_pA_cm2
        ),
    )


def conserved_cortical_point_hh_parameters(
    params: dict[str, Any],
) -> dict[str, Any]:
    """Apply the preregistered no-fit cortical point-HH transformation."""

    cell_class = str(params.get("cell_class", ""))
    if cell_class not in CORTICAL_CELL_CLASSES:
        raise ValueError(
            "the conserved point-HH intervention applies only to registered "
            f"cortical classes, got {cell_class!r}"
        )
    if params.get("somatic_spike_model", "classic_hh") != "classic_hh":
        raise ValueError("point-HH collapse requires the classic HH spike model")
    for controlled in (
        "point_hh_intervention",
        "disabled_nak_compartments",
        "axial_edge_conductance_scales",
    ):
        if controlled in params:
            raise ValueError(f"{controlled} is controlled by the point-HH intervention")
    if params.get("voltage_clamps_mV"):
        raise ValueError("point-HH collapse does not permit compartment voltage clamps")

    source_cell = params.get("cell_spec")
    if not isinstance(source_cell, CellSpec):
        raise TypeError("point-HH collapse requires an explicit source CellSpec")
    point_cell = collapse_cell_spec_to_conserved_point(source_cell)
    transformed = dict(params)
    transformed["cell_spec"] = point_cell
    transformed["point_hh_intervention"] = POINT_HH_INTERVENTION
    transformed["synaptic_ports"] = tuple(
        _retarget_conductance_port(
            port,
            source_cell=source_cell,
            point_cell=point_cell,
        )
        for port in params.get("synaptic_ports", ())
    )
    transformed["external_input_ports"] = tuple(
        _retarget_conductance_port(
            port,
            source_cell=source_cell,
            point_cell=point_cell,
        )
        for port in params.get("external_input_ports", ())
    )
    transformed["gap_junction_ports"] = tuple(
        _retarget_gap_port(
            port,
            source_cell=source_cell,
            point_cell=point_cell,
        )
        for port in params.get("gap_junction_ports", ())
    )
    transformed["injection_ports"] = tuple(
        _retarget_injection_port(
            port,
            source_cell=source_cell,
            point_cell=point_cell,
        )
        for port in params.get("injection_ports", ())
    )
    return transformed


def create_conserved_cortical_point_hh_population(
    *,
    name: str,
    size: int,
    params: dict[str, Any],
    brian=None,
) -> SmartPopulationAdapter:
    """Create one preregistered conserved cortical point-HH population."""

    population = create_compartmental_hh_population(
        name=name,
        size=size,
        params=conserved_cortical_point_hh_parameters(params),
        brian=brian,
    )
    validate_smart_population_adapter(population)
    return population
