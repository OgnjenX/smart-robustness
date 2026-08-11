"""Source-backed assembly of one first-order 9x9 SMART sector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models.compartmental_hh import CompartmentalPopulation, create_compartmental_hh_population
from .models.modeldb112923 import FirstOrderPopulationFacts, first_order_population_facts
from .models.ports import external_ports_for_target, gap_ports_for_target, ports_for_target
from .projections import SUPPLEMENTARY_TABLE3, ConnectionKind
from .synapses import connect_classic_projection, connect_gap_junction


@dataclass(slots=True)
class FirstOrderSector:
    network: Any
    populations: dict[str, CompartmentalPopulation]
    facts: tuple[FirstOrderPopulationFacts, ...]
    projections: dict[str, Any]

    @property
    def cell_count(self) -> int:
        return sum(int(population.group.N) for population in self.populations.values())

    @property
    def compartment_count(self) -> int:
        return sum(
            int(population.group.N) * len(population.compartments)
            for population in self.populations.values()
        )


def _population_parameters(facts: FirstOrderPopulationFacts) -> dict[str, Any]:
    has_ahp = facts.ahp_density_mS_cm2 is not None
    parameters: dict[str, Any] = {
        "cell_spec": facts.cell,
        "cell_class": facts.canonical_name,
        "axial_convention": "kinness_serialized_edge",
        "leak_convention": "table3_reversal",
        "voltage_coordinate": "shifted_67_mV",
        "nak_rate_convention": "standard_traub_miles",
        "calcium_gate_convention": "modeldb_112923",
        "calcium_density_convention": "table3",
        "ahp_convention": "smart_network_112923" if has_ahp else "modeldb_112923",
        "specific_capacitance_uF_cm2": 1.0,
        "enable_ahp_ach": has_ahp,
        "e_na_mV": facts.e_na_mV,
        "e_k_mV": facts.e_k_mV,
        "e_ca_mV": facts.e_ca_mV,
        "method": "rk4",
        "synaptic_ports": ports_for_target(SUPPLEMENTARY_TABLE3.records, facts.canonical_name),
        "gap_junction_ports": gap_ports_for_target(
            SUPPLEMENTARY_TABLE3.records, facts.canonical_name
        ),
        "external_input_ports": external_ports_for_target(
            SUPPLEMENTARY_TABLE3.records, facts.canonical_name
        ),
    }
    if has_ahp:
        soma = facts.cell.soma
        parameters.update(
            {
                "ahp_max_conductance_nS": (
                    float(facts.ahp_density_mS_cm2) * soma.lateral_area_cm2 * 1e6
                ),
                "ahp_event_weight": 1.0,
                "e_ahp_mV": facts.ahp_reversal_mV,
            }
        )
    return parameters


def build_first_order_intrinsic_sector(*, brian=None) -> FirstOrderSector:
    """Instantiate all 812 cells and 1,950 compartments before connectivity.

    This milestone intentionally assembles intrinsic populations only. Chemical
    synapses, gap junctions, external inputs, plasticity, and depletion are
    separate audited components and are not implied by this constructor.
    """

    if brian is None:
        import brian2 as brian

    facts = first_order_population_facts()
    populations: dict[str, CompartmentalPopulation] = {}
    for population_facts in facts:
        width, height = population_facts.shape
        populations[population_facts.canonical_name] = create_compartmental_hh_population(
            name=f"smart_v1_{population_facts.canonical_name}",
            size=width * height,
            params=_population_parameters(population_facts),
            brian=brian,
        )
    network = brian.Network(*(population.group for population in populations.values()))
    return FirstOrderSector(network=network, populations=populations, facts=facts, projections={})


def build_first_order_chemical_sector(*, brian=None) -> FirstOrderSector:
    """Instantiate the first-order cells and every in-scope chemical projection."""

    if brian is None:
        import brian2 as brian

    sector = build_first_order_intrinsic_sector(brian=brian)
    facts_by_name = {fact.canonical_name: fact for fact in sector.facts}
    projections: dict[str, Any] = {}
    for record in SUPPLEMENTARY_TABLE3.records:
        if record.kind is not ConnectionKind.CHEMICAL:
            continue
        if record.source.population not in sector.populations:
            # The V2->V1 feedback projection belongs to the higher-order loop.
            continue
        if record.target.population not in sector.populations:
            continue
        projections[record.id] = connect_classic_projection(
            record,
            pre=sector.populations[record.source.population],
            post=sector.populations[record.target.population],
            source_shape=facts_by_name[record.source.population].shape,
            target_shape=facts_by_name[record.target.population].shape,
            brian=brian,
        )
    sector.projections = projections
    sector.network.add(*projections.values())
    return sector


def build_first_order_connected_sector(*, brian=None) -> FirstOrderSector:
    """Build chemical and electrical connectivity; external inputs remain separate."""

    if brian is None:
        import brian2 as brian

    sector = build_first_order_chemical_sector(brian=brian)
    facts_by_name = {fact.canonical_name: fact for fact in sector.facts}
    electrical: dict[str, Any] = {}
    for record in SUPPLEMENTARY_TABLE3.records:
        if record.kind is not ConnectionKind.GAP_JUNCTION:
            continue
        electrical[record.id] = connect_gap_junction(
            record,
            pre=sector.populations[record.source.population],
            post=sector.populations[record.target.population],
            source_shape=facts_by_name[record.source.population].shape,
            target_shape=facts_by_name[record.target.population].shape,
            brian=brian,
        )
    sector.projections.update(electrical)
    sector.network.add(*electrical.values())
    return sector
