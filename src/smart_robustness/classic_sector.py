"""Source-backed assembly of one first-order 9x9 SMART sector."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from .modeldb_projections import MODELDB_FIRST_ORDER, MODELDB_FULL
from .models.compartmental_hh import CompartmentalPopulation, create_compartmental_hh_population
from .models.modeldb112923 import (
    FirstOrderPopulationFacts,
    first_order_population_facts,
    second_order_population_facts,
)
from .models.ports import (
    modeldb_external_ports_for_target,
    modeldb_gap_ports_for_target,
    modeldb_injection_ports_for_target,
    modeldb_ports_for_target,
)
from .partition import population_parts
from .synapses import connect_modeldb_gap_junction, connect_modeldb_projection


@dataclass(slots=True)
class FirstOrderSector:
    network: Any
    populations: dict[str, Any]
    facts: tuple[FirstOrderPopulationFacts, ...]
    projections: dict[str, Any]

    @property
    def cell_count(self) -> int:
        return sum(
            sum(int(part.group.N) for _, part in population_parts(population))
            for population in self.populations.values()
        )

    @property
    def compartment_count(self) -> int:
        return sum(
            sum(int(part.group.N) for _, part in population_parts(population)) * len(population.compartments)
            for population in self.populations.values()
        )


class ZeroSensitivityInputConvention(StrEnum):
    """Legacy treatment of input gates whose four source sensitivities are zero."""

    FRAMEWORK_RESTING_LEAK = "framework_resting_leak"
    OMIT_ALL_ZERO = "omit_all_zero"


class ProjectionSourceConvention(StrEnum):
    """Resolution of executable-source labels contradicted by the supplement."""

    MODELDB_AS_SERIALIZED = "modeldb_as_serialized"
    PAPER_SUPPLEMENT_CROSS_CHECKED = "paper_supplement_cross_checked"


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
    spike_event_threshold_mV: float = 30.0
    modifiable_weight_initialization: str = "source_serialized_weight"
    gaussian_weight_convention: str = "normalized_density"
    gaussian_learning_bounds_convention: str = "projection_level"
    projection_source_convention: str = "modeldb_as_serialized"

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
        modifiable_weight_initialization="figure6_pathway_specific",
        gaussian_learning_bounds_convention="figure6_pathway_specific",
        projection_source_convention="modeldb_as_serialized",
    )


def _resolved_projection_record(record, *, conventions: FirstOrderRuntimeConventions):
    source_convention = ProjectionSourceConvention(conventions.projection_source_convention)
    if (
        source_convention is ProjectionSourceConvention.PAPER_SUPPLEMENT_CROSS_CHECKED
        and record.id == "modeldb112923.projection.022"
    ):
        # SMART.nml points an excitatory channel named "AMPA 2/3" at
        # Layer_4_INT with a Gaussian rule. Supplementary Table 3 independently
        # specifies layer-2/3 excitatory -> layer-6II distal, one-to-one, as
        # does the paper's category pathway. Preserve the literal record under
        # MODELDB_AS_SERIALIZED and resolve it only in the cross-checked profile.
        from dataclasses import replace

        return replace(
            record,
            source_population="layer23_excitatory_v1",
            method="connectFromOne",
            kernel=None,
        )
    return record


def first_order_population_parameters(
    facts: FirstOrderPopulationFacts,
    *,
    conventions: FirstOrderRuntimeConventions,
    catalog=MODELDB_FIRST_ORDER,
) -> dict[str, Any]:
    has_ahp = facts.ahp_density_mS_cm2 is not None
    zero_input_convention = ZeroSensitivityInputConvention(
        conventions.zero_sensitivity_input_convention
    )
    external_input_ports = modeldb_external_ports_for_target(
        catalog.external_channels, facts.canonical_name
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
        "spike_event_threshold_mV": conventions.spike_event_threshold_mV,
        "ahp_convention": "smart_network_112923" if has_ahp else "modeldb_112923",
        "specific_capacitance_uF_cm2": conventions.specific_capacitance_uF_cm2,
        "enable_ahp_ach": has_ahp,
        "e_na_mV": facts.e_na_mV,
        "e_k_mV": facts.e_k_mV,
        "e_ca_mV": facts.e_ca_mV,
        "method": conventions.integration_method,
        "synaptic_ports": modeldb_ports_for_target(
            catalog.projections, facts.canonical_name
        ),
        "gap_junction_ports": modeldb_gap_ports_for_target(
            catalog.projections, facts.canonical_name
        ),
        "external_input_ports": external_input_ports,
        "injection_ports": modeldb_injection_ports_for_target(
            catalog.external_channels, facts.canonical_name
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


def build_full_smart_network(
    *,
    conventions: FirstOrderRuntimeConventions | None = None,
    projection_ids: frozenset[str] | None = None,
    brian=None,
) -> FirstOrderSector:
    """Assemble source-backed V1-pulvinar-V2 cells and selected projections.

    ``projection_ids`` is a diagnostic selector: ``None`` means the complete
    archived network, while an explicit set must contain only catalog IDs.
    Population definitions and receptor ports remain unchanged so projection
    blocks can be isolated without constructing a different neuron model.
    """

    if brian is None:
        import brian2 as brian
    conventions = conventions or FirstOrderRuntimeConventions()
    known_projection_ids = {record.id for record in MODELDB_FULL.projections}
    if projection_ids is not None:
        unknown = projection_ids - known_projection_ids
        if unknown:
            raise ValueError(f"unknown full-network projection IDs: {sorted(unknown)}")
    facts = first_order_population_facts() + second_order_population_facts()
    facts_by_name = {fact.canonical_name: fact for fact in facts}
    populations: dict[str, CompartmentalPopulation] = {}
    for fact in facts:
        width, height = fact.shape
        populations[fact.canonical_name] = create_compartmental_hh_population(
            name=f"smart_full_{fact.canonical_name}",
            size=width * height,
            params=first_order_population_parameters(
                fact, conventions=conventions, catalog=MODELDB_FULL
            ),
            brian=brian,
        )
    network = brian.Network(*(population.group for population in populations.values()))
    projections: dict[str, Any] = {}
    for record in MODELDB_FULL.projections:
        if projection_ids is not None and record.id not in projection_ids:
            continue
        kwargs = {
            "record": record,
            "pre": populations[record.source_population],
            "post": populations[record.target_population],
            "source_shape": facts_by_name[record.source_population].shape,
            "target_shape": facts_by_name[record.target_population].shape,
            "brian": brian,
        }
        if record.kind == "chemical":
            projection = connect_modeldb_projection(
                **kwargs,
                modifiable_weight_initialization=conventions.modifiable_weight_initialization,
                gaussian_weight_convention=conventions.gaussian_weight_convention,
                gaussian_learning_bounds_convention=(
                    conventions.gaussian_learning_bounds_convention
                ),
                spike_event_coordinate=conventions.spike_event_coordinate,
                spike_event_threshold_mV=conventions.spike_event_threshold_mV,
            )
        else:
            projection = connect_modeldb_gap_junction(**kwargs)
        projections[record.id] = projection
        network.add(projection)
    return FirstOrderSector(network, populations, facts, projections)


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
        record = _resolved_projection_record(record, conventions=resolved_conventions)
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
            spike_event_coordinate=resolved_conventions.spike_event_coordinate,
            spike_event_threshold_mV=resolved_conventions.spike_event_threshold_mV,
            brian=brian,
        )
    sector.projections = projections
    sector.network.add(*projections.values())
    return sector


def build_first_order_voltage_clamp_sector(
    *,
    clamped_relay_indices: tuple[int, ...],
    holding_mV: float = -12.0,
    compartment: str = "proximal_dendrite",
    conventions: FirstOrderRuntimeConventions | None = None,
    brian=None,
) -> FirstOrderSector:
    """Build a connected sector with a discrete exact relay voltage clamp.

    Brian2 does not allow split presynaptic groups to sum into one receptor
    state. Keeping the original group therefore preserves the archived SMART
    topology, while start/end operators pin the selected dendrites at every
    integration boundary without introducing a masked ``0 * NaN`` ODE.
    """

    if brian is None:
        import brian2 as brian
    conventions = conventions or FirstOrderRuntimeConventions()
    relay_fact = next(
        fact for fact in first_order_population_facts() if fact.canonical_name == "thalamic_relay"
    )
    relay_compartments = relay_fact.cell.compartments
    valid_compartments = {item.name for item in relay_compartments}
    if compartment not in valid_compartments:
        raise ValueError(
            f"unknown relay clamp compartment {compartment!r}; expected {sorted(valid_compartments)}"
        )
    if not clamped_relay_indices or len(set(clamped_relay_indices)) != len(clamped_relay_indices):
        raise ValueError("clamped relay indices must be nonempty and unique")
    if any(index < 0 or index >= 81 for index in clamped_relay_indices):
        raise ValueError("clamped relay index outside 9x9 sheet")
    sector = build_first_order_connected_sector(conventions=conventions, brian=brian)
    relay = sector.populations["thalamic_relay"].group
    indices = np.asarray(clamped_relay_indices, dtype=int)

    voltage = getattr(relay, f"v_{compartment}")

    def pin_relay_compartments() -> None:
        voltage[indices] = holding_mV * brian.mV

    clamp_start = brian.NetworkOperation(
        pin_relay_compartments, when="start", name="smart_relay_voltage_clamp_start"
    )
    clamp_end = brian.NetworkOperation(
        pin_relay_compartments, when="end", name="smart_relay_voltage_clamp_end"
    )
    pin_relay_compartments()
    sector.network.add(clamp_start, clamp_end)
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
