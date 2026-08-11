"""Source-backed assembly of one first-order 9x9 SMART sector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models.compartmental_hh import CompartmentalPopulation, create_compartmental_hh_population
from .models.modeldb112923 import FirstOrderPopulationFacts, first_order_population_facts


@dataclass(slots=True)
class FirstOrderSector:
    network: Any
    populations: dict[str, CompartmentalPopulation]
    facts: tuple[FirstOrderPopulationFacts, ...]

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
    return FirstOrderSector(network=network, populations=populations, facts=facts)
