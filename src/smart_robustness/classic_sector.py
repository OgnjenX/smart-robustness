"""Source-backed assembly of one first-order 9x9 SMART sector."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from .modeldb_projections import MODELDB_FIRST_ORDER
from .models.compartmental_hh import CompartmentalPopulation, create_compartmental_hh_population
from .models.modeldb112923 import FirstOrderPopulationFacts, first_order_population_facts
from .models.ports import (
    modeldb_external_ports_for_target,
    modeldb_gap_ports_for_target,
    modeldb_injection_ports_for_target,
    modeldb_ports_for_target,
)
from .synapses import connect_modeldb_gap_junction, connect_modeldb_projection


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


class ZeroSensitivityInputConvention(StrEnum):
    """Legacy treatment of input gates whose four source sensitivities are zero."""

    FRAMEWORK_RESTING_LEAK = "framework_resting_leak"
    OMIT_ALL_ZERO = "omit_all_zero"


@dataclass(frozen=True, slots=True)
class FirstOrderRuntimeConventions:
    """Complete executable convention profile for one classic-sector run."""

    axial_convention: str = "kinness_serialized_edge"
    leak_convention: str = "table3_reversal"
    voltage_coordinate: str = "relative_to_table3_leak"
    nak_rate_convention: str = "standard_traub_miles"
    calcium_gate_convention: str = "modeldb_112923"
    gate_initialization_convention: str = "steady_state_at_initial_voltage"
    calcium_density_convention: str = "table3"
    specific_capacitance_uF_cm2: float = 1.0
    integration_method: str = "rk4"
    zero_sensitivity_input_convention: str = "framework_resting_leak"
    spike_event_coordinate: str = "absolute_physical"
    modifiable_weight_initialization: str = "source_serialized_weight"
    gaussian_weight_convention: str = "normalized_density"
    gaussian_learning_bounds_convention: str = "projection_level"

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def figure6_runtime_conventions() -> FirstOrderRuntimeConventions:
    """Return the source-constrained profile used by Figure 6 validation.

    KInNeSS does not serialize gate initial values. Its zero-initialized
    candidate avoids a non-physiological synchronized TRN startup spike.
    Input channels with four zero sensitivities are inert declarations, not
    permanent conductances. The remaining values are recovered source values
    or conventions independently constrained by the published Figure 6 maps.
    """

    return FirstOrderRuntimeConventions(
        gate_initialization_convention="zero",
        zero_sensitivity_input_convention="omit_all_zero",
    )


def first_order_population_parameters(
    facts: FirstOrderPopulationFacts,
    *,
    conventions: FirstOrderRuntimeConventions,
) -> dict[str, Any]:
    has_ahp = facts.ahp_density_mS_cm2 is not None
    zero_input_convention = ZeroSensitivityInputConvention(
        conventions.zero_sensitivity_input_convention
    )
    external_input_ports = modeldb_external_ports_for_target(
        MODELDB_FIRST_ORDER.external_channels, facts.canonical_name
    )
    if zero_input_convention is ZeroSensitivityInputConvention.OMIT_ALL_ZERO:
        external_input_ports = tuple(
            port for port in external_input_ports if any(port.sensitivities_mV)
        )
    parameters: dict[str, Any] = {
        "cell_spec": facts.cell,
        "cell_class": facts.canonical_name,
        "axial_convention": conventions.axial_convention,
        "leak_convention": conventions.leak_convention,
        "voltage_coordinate": conventions.voltage_coordinate,
        "nak_rate_convention": conventions.nak_rate_convention,
        "calcium_gate_convention": conventions.calcium_gate_convention,
        "gate_initialization_convention": conventions.gate_initialization_convention,
        "calcium_density_convention": conventions.calcium_density_convention,
        "spike_event_coordinate": conventions.spike_event_coordinate,
        "ahp_convention": "smart_network_112923" if has_ahp else "modeldb_112923",
        "specific_capacitance_uF_cm2": conventions.specific_capacitance_uF_cm2,
        "enable_ahp_ach": has_ahp,
        "e_na_mV": facts.e_na_mV,
        "e_k_mV": facts.e_k_mV,
        "e_ca_mV": facts.e_ca_mV,
        "method": conventions.integration_method,
        "synaptic_ports": modeldb_ports_for_target(
            MODELDB_FIRST_ORDER.projections, facts.canonical_name
        ),
        "gap_junction_ports": modeldb_gap_ports_for_target(
            MODELDB_FIRST_ORDER.projections, facts.canonical_name
        ),
        "external_input_ports": external_input_ports,
        "injection_ports": modeldb_injection_ports_for_target(
            MODELDB_FIRST_ORDER.external_channels, facts.canonical_name
        ),
        "depletion_epsilon": facts.depletion_epsilon,
        "depletion_recovery_ms": facts.depletion_recovery_ms,
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


def build_first_order_intrinsic_sector(
    *,
    conventions: FirstOrderRuntimeConventions | None = None,
    gate_initialization_convention: str | None = None,
    brian=None,
) -> FirstOrderSector:
    """Instantiate all 812 cells and 1,950 compartments before connectivity.

    This milestone intentionally assembles intrinsic populations only. Chemical
    synapses, gap junctions, external inputs, plasticity, and depletion are
    separate audited components and are not implied by this constructor.
    """

    if brian is None:
        import brian2 as brian

    if conventions is not None and gate_initialization_convention is not None:
        raise ValueError("pass conventions or gate_initialization_convention, not both")
    conventions = conventions or FirstOrderRuntimeConventions(
        gate_initialization_convention=(
            gate_initialization_convention or "steady_state_at_initial_voltage"
        )
    )

    facts = first_order_population_facts()
    populations: dict[str, CompartmentalPopulation] = {}
    for population_facts in facts:
        width, height = population_facts.shape
        populations[population_facts.canonical_name] = create_compartmental_hh_population(
            name=f"smart_v1_{population_facts.canonical_name}",
            size=width * height,
            params=first_order_population_parameters(
                population_facts,
                conventions=conventions,
            ),
            brian=brian,
        )
    network = brian.Network(*(population.group for population in populations.values()))
    return FirstOrderSector(network=network, populations=populations, facts=facts, projections={})


def build_first_order_chemical_sector(
    *,
    conventions: FirstOrderRuntimeConventions | None = None,
    gate_initialization_convention: str | None = None,
    brian=None,
) -> FirstOrderSector:
    """Instantiate the first-order cells and every in-scope chemical projection."""

    if brian is None:
        import brian2 as brian

    resolved_conventions = conventions or FirstOrderRuntimeConventions(
        gate_initialization_convention=(
            gate_initialization_convention or "steady_state_at_initial_voltage"
        )
    )
    sector = build_first_order_intrinsic_sector(
        conventions=resolved_conventions,
        brian=brian,
    )
    facts_by_name = {fact.canonical_name: fact for fact in sector.facts}
    projections: dict[str, Any] = {}
    for record in MODELDB_FIRST_ORDER.projections:
        if record.kind != "chemical":
            continue
        if record.source_population not in sector.populations:
            # The V2->V1 feedback projection belongs to the higher-order loop.
            continue
        if record.target_population not in sector.populations:
            continue
        projections[record.id] = connect_modeldb_projection(
            record,
            pre=sector.populations[record.source_population],
            post=sector.populations[record.target_population],
            source_shape=facts_by_name[record.source_population].shape,
            target_shape=facts_by_name[record.target_population].shape,
            modifiable_weight_initialization=(
                resolved_conventions.modifiable_weight_initialization
            ),
            gaussian_weight_convention=resolved_conventions.gaussian_weight_convention,
            gaussian_learning_bounds_convention=(
                resolved_conventions.gaussian_learning_bounds_convention
            ),
            brian=brian,
        )
    sector.projections = projections
    sector.network.add(*projections.values())
    return sector


def build_first_order_connected_sector(
    *,
    conventions: FirstOrderRuntimeConventions | None = None,
    gate_initialization_convention: str | None = None,
    brian=None,
) -> FirstOrderSector:
    """Build chemical and electrical connectivity; external inputs remain separate."""

    if brian is None:
        import brian2 as brian

    sector = build_first_order_chemical_sector(
        conventions=conventions,
        gate_initialization_convention=gate_initialization_convention,
        brian=brian,
    )
    facts_by_name = {fact.canonical_name: fact for fact in sector.facts}
    electrical: dict[str, Any] = {}
    for record in MODELDB_FIRST_ORDER.projections:
        if record.kind != "gap_junction":
            continue
        if record.source_population not in sector.populations:
            continue
        electrical[record.id] = connect_modeldb_gap_junction(
            record,
            pre=sector.populations[record.source_population],
            post=sector.populations[record.target_population],
            source_shape=facts_by_name[record.source_population].shape,
            target_shape=facts_by_name[record.target_population].shape,
            brian=brian,
        )
    sector.projections.update(electrical)
    sector.network.add(*electrical.values())
    return sector
