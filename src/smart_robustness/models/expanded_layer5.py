"""Registered branched layer-5 morphology intervention."""

from __future__ import annotations

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
from .table3 import CellSpec

LAYER5_CELL_CLASS = "layer5_excitatory_v1"
EXPANDED_LAYER5_INTERVENTION = "expanded_layer5_branched_v1"
EXPANDED_LAYER5_TOPOLOGY = (
    ("soma", "basal_dendrite"),
    ("soma", "proximal_apical_dendrite"),
    ("proximal_apical_dendrite", "distal_tuft"),
)
_PortWithConductance = TypeVar(
    "_PortWithConductance",
    SynapticPortSpec,
    ExternalInputPortSpec,
)


def expand_layer5_cell_spec(source: CellSpec) -> CellSpec:
    """Apply the preregistered equal-area proximal split."""

    names = tuple(compartment.name for compartment in source.compartments)
    expected = ("soma", "proximal_dendrite", "distal_dendrite")
    if names != expected:
        raise ValueError(
            f"expanded layer-5 source compartments must be {expected}, got {names}"
        )
    proximal = source.compartment("proximal_dendrite")
    if any(
        value is not None
        for value in (
            proximal.g_na_mS_cm2,
            proximal.g_k_mS_cm2,
            proximal.g_ca_mS_cm2,
        )
    ):
        raise ValueError("registered proximal split requires a passive source compartment")
    basal = replace(
        proximal,
        name="basal_dendrite",
        length_mm=proximal.length_mm * 0.5,
    )
    proximal_apical = replace(
        proximal,
        name="proximal_apical_dendrite",
        length_mm=proximal.length_mm * 0.5,
    )
    distal_tuft = replace(source.compartment("distal_dendrite"), name="distal_tuft")
    return CellSpec(
        name=f"{source.name}_expanded_branched",
        compartments=(source.soma, basal, proximal_apical, distal_tuft),
    )


def _target_name(compartment: str) -> str:
    mapping = {
        "soma": "soma",
        "proximal_dendrite": "basal_dendrite",
        "distal_dendrite": "distal_tuft",
    }
    try:
        return mapping[compartment]
    except KeyError as exc:
        raise ValueError(
            f"expanded layer-5 intervention has no mapping for {compartment!r}"
        ) from exc


def _retarget_conductance_port(
    port: _PortWithConductance,
    *,
    source: CellSpec,
    expanded: CellSpec,
) -> _PortWithConductance:
    target = _target_name(port.compartment)
    old_area = source.compartment(port.compartment).lateral_area_cm2
    new_area = expanded.compartment(target).lateral_area_cm2
    return replace(
        port,
        compartment=target,
        conductance_density_mS_cm2=(
            port.conductance_density_mS_cm2 * old_area / new_area
        ),
    )


def _retarget_gap_port(
    port: GapJunctionPortSpec,
    *,
    source: CellSpec,
    expanded: CellSpec,
) -> GapJunctionPortSpec:
    target = _target_name(port.compartment)
    old_compartment = source.compartment(port.compartment)
    new_compartment = expanded.compartment(target)
    old_total_nS = kinness_gap_total_conductance_nS(
        port.conductance_density_mS_cm2,
        diameter_mm=old_compartment.diameter_mm,
        length_mm=old_compartment.length_mm,
    )
    new_unit_total_nS = kinness_gap_total_conductance_nS(
        1.0,
        diameter_mm=new_compartment.diameter_mm,
        length_mm=new_compartment.length_mm,
    )
    return replace(
        port,
        compartment=target,
        conductance_density_mS_cm2=old_total_nS / new_unit_total_nS,
    )


def _retarget_injection_port(
    port: InjectionPortSpec,
    *,
    source: CellSpec,
    expanded: CellSpec,
) -> InjectionPortSpec:
    target = _target_name(port.compartment)
    old_area = source.compartment(port.compartment).lateral_area_cm2
    new_area = expanded.compartment(target).lateral_area_cm2
    scale = old_area / new_area
    return replace(
        port,
        compartment=target,
        sensitivities_pA_cm2=tuple(
            sensitivity * scale for sensitivity in port.sensitivities_pA_cm2
        ),
    )


def expanded_layer5_parameters(params: dict[str, Any]) -> dict[str, Any]:
    """Apply the no-fit expanded layer-5 transformation."""

    cell_class = str(params.get("cell_class", ""))
    if cell_class != LAYER5_CELL_CLASS:
        raise ValueError(
            "expanded layer-5 intervention applies only to "
            f"{LAYER5_CELL_CLASS}, got {cell_class!r}"
        )
    if params.get("somatic_spike_model", "classic_hh") != "classic_hh":
        raise ValueError("expanded layer-5 intervention requires classic HH")
    for controlled in (
        "expanded_layer5_intervention",
        "axial_topology_pairs",
        "axial_edge_conductance_scales",
        "disabled_nak_compartments",
    ):
        if controlled in params:
            raise ValueError(
                f"{controlled} is controlled by the expanded layer-5 intervention"
            )
    if params.get("voltage_clamps_mV"):
        raise ValueError("expanded layer-5 intervention does not permit voltage clamps")
    source = params.get("cell_spec")
    if not isinstance(source, CellSpec):
        raise TypeError("expanded layer-5 intervention requires an explicit CellSpec")
    expanded = expand_layer5_cell_spec(source)
    transformed = dict(params)
    transformed["cell_spec"] = expanded
    transformed["expanded_layer5_intervention"] = EXPANDED_LAYER5_INTERVENTION
    transformed["axial_topology_pairs"] = EXPANDED_LAYER5_TOPOLOGY
    transformed["synaptic_ports"] = tuple(
        _retarget_conductance_port(port, source=source, expanded=expanded)
        for port in params.get("synaptic_ports", ())
    )
    transformed["external_input_ports"] = tuple(
        _retarget_conductance_port(port, source=source, expanded=expanded)
        for port in params.get("external_input_ports", ())
    )
    transformed["gap_junction_ports"] = tuple(
        _retarget_gap_port(port, source=source, expanded=expanded)
        for port in params.get("gap_junction_ports", ())
    )
    transformed["injection_ports"] = tuple(
        _retarget_injection_port(port, source=source, expanded=expanded)
        for port in params.get("injection_ports", ())
    )
    return transformed


def create_expanded_layer5_population(
    *,
    name: str,
    size: int,
    params: dict[str, Any],
    brian=None,
) -> SmartPopulationAdapter:
    """Create the registered expanded branched layer-5 population."""

    population = create_compartmental_hh_population(
        name=name,
        size=size,
        params=expanded_layer5_parameters(params),
        brian=brian,
    )
    validate_smart_population_adapter(population)
    return population
