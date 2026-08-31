"""Source-backed assembly of one first-order 9x9 SMART sector."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from typing import Any

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
from .models.table3 import CellSpec, get_cell_spec
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


class IntrinsicCellConvention(StrEnum):
    """Source selected for dimensions and passive/intrinsic cell parameters."""

    MODELDB_112923 = "modeldb_112923"
    PAPER_TABLE3 = "paper_table3"
    MODELDB_RELAY_PAPER_TABLE3_OTHERS = "modeldb_relay_paper_table3_others"
    MODELDB_RELAY_LAYER6II_PAPER_TABLE3_OTHERS = (
        "modeldb_relay_layer6ii_paper_table3_others"
    )


class TrnPotassiumConvention(StrEnum):
    """Discrete Table-3/SMART.nml conflicts in the TRN potassium system."""

    SELECTED_SOURCE = "selected_source"
    MODELDB_DENSITY = "modeldb_density"
    MODELDB_REVERSAL = "modeldb_reversal"
    MODELDB_DENSITY_AND_REVERSAL = "modeldb_density_and_reversal"


class TrnCalciumSourceConvention(StrEnum):
    """Discrete Table-3/SMART.nml conflicts in the TRN calcium system."""

    SELECTED_SOURCE = "selected_source"
    MODELDB_SOMA_CHANNEL = "modeldb_soma_channel"
    MODELDB_REVERSAL = "modeldb_reversal"
    MODELDB_SOMA_CHANNEL_AND_REVERSAL = "modeldb_soma_channel_and_reversal"


class TrnDendriticCalciumDensityConvention(StrEnum):
    """Table-3 versus SMART.nml density on existing TRN dendritic T channels."""

    SELECTED_SOURCE = "selected_source"
    MODELDB_100 = "modeldb_100"


class CalciumKineticsConvention(StrEnum):
    """Source selected for population-specific T-current gate equations."""

    MODELDB_112923 = "modeldb_112923"
    PAPER_2008 = "paper_2008"


@dataclass(frozen=True, slots=True)
class FirstOrderRuntimeConventions:
    """Complete executable convention profile for one classic-sector run."""

    axial_convention: str = "kinness_serialized_edge"
    intrinsic_cell_convention: str = "modeldb_112923"
    leak_convention: str = "table3_reversal"
    voltage_coordinate: str = "relative_to_table3_leak"
    nak_rate_convention: str = "standard_traub_miles"
    calcium_kinetics_convention: str = "modeldb_112923"
    calcium_gate_convention: str = "modeldb_112923"
    calcium_voltage_coordinate: str = "integrated_voltage"
    gate_initialization_convention: str = "steady_state_at_initial_voltage"
    membrane_initialization_convention: str = "physical_leak_voltage"
    calcium_density_convention: str = "table3"
    specific_capacitance_uF_cm2: float = 1.0
    integration_method: str = "rk4"
    zero_sensitivity_input_convention: str = "framework_resting_leak"
    spike_event_coordinate: str = "absolute_physical"
    spike_event_threshold_mV: float = 30.0
    trn_spike_event_coordinate: str | None = None
    trn_spike_event_threshold_mV: float | None = None
    trn_spike_event_release_mV: float | None = None
    trn_spike_event_voltage_offset_mV: float | None = None
    trn_spike_event_proximal_blend_fraction: float | None = None
    nonspecific_spike_event_proximal_blend_fraction: float | None = None
    trn_potassium_convention: str = "selected_source"
    trn_soma_sodium_density_mS_cm2: float | None = None
    trn_soma_potassium_density_mS_cm2: float | None = None
    trn_calcium_source_convention: str = "selected_source"
    trn_dendritic_calcium_density_convention: str = "selected_source"
    trn_dendritic_calcium_density_mS_cm2: float | None = None
    trn_soma_proximal_axial_conductance_scale: float = 1.0
    postsynaptic_learning_coordinate: str = "absolute_physical"
    postsynaptic_learning_threshold_mV: float = 30.0
    top_down_learning_rule_convention: str = "serialized_presynaptic"
    postsynaptic_learning_timestamp: str = "emitted_event"
    postsynaptic_signal_convention: str = "paper_equation6_literal"
    spike_event_rule: str = "latched_peak_then_zero"
    modifiable_weight_initialization: str = "source_serialized_weight"
    gaussian_weight_convention: str = "source_peak"
    gaussian_spread_convention: str = "standard_deviation"
    ring_kernel_convention: str = "center_excluded_gaussian"
    corticoreticular_ring_kernel_convention: str | None = None
    gaussian_learning_bounds_convention: str = "projection_level"
    postsynaptic_depression_scale_convention: str = "local_learning_bounds"
    projection_source_convention: str = "modeldb_as_serialized"
    convergent_external_input_convention: str = "sum_independent_currents"

    @property
    def fingerprint(self) -> str:
        values = asdict(self)
        if values["postsynaptic_depression_scale_convention"] == "local_learning_bounds":
            # Preserve the identity of every historical profile. This optional
            # discriminator is serialized only when it differs from the
            # pre-existing executable behavior.
            values.pop("postsynaptic_depression_scale_convention")
        if values["trn_spike_event_coordinate"] is None:
            values.pop("trn_spike_event_coordinate")
        if values["trn_spike_event_threshold_mV"] is None:
            values.pop("trn_spike_event_threshold_mV")
        if values["trn_spike_event_release_mV"] is None:
            values.pop("trn_spike_event_release_mV")
        if values["trn_spike_event_voltage_offset_mV"] is None:
            values.pop("trn_spike_event_voltage_offset_mV")
        if values["trn_spike_event_proximal_blend_fraction"] is None:
            values.pop("trn_spike_event_proximal_blend_fraction")
        if values["nonspecific_spike_event_proximal_blend_fraction"] is None:
            values.pop("nonspecific_spike_event_proximal_blend_fraction")
        if values["trn_potassium_convention"] == "selected_source":
            values.pop("trn_potassium_convention")
        if values["trn_soma_sodium_density_mS_cm2"] is None:
            values.pop("trn_soma_sodium_density_mS_cm2")
        if values["trn_soma_potassium_density_mS_cm2"] is None:
            values.pop("trn_soma_potassium_density_mS_cm2")
        if values["trn_calcium_source_convention"] == "selected_source":
            values.pop("trn_calcium_source_convention")
        if values["trn_dendritic_calcium_density_convention"] == "selected_source":
            values.pop("trn_dendritic_calcium_density_convention")
        if values["trn_dendritic_calcium_density_mS_cm2"] is None:
            values.pop("trn_dendritic_calcium_density_mS_cm2")
        if values["trn_soma_proximal_axial_conductance_scale"] == 1.0:
            values.pop("trn_soma_proximal_axial_conductance_scale")
        if values["postsynaptic_learning_threshold_mV"] == values["spike_event_threshold_mV"]:
            # Historical profiles used one value for both roles. Preserve their
            # fingerprints while allowing Equation 6 to be audited separately.
            values.pop("postsynaptic_learning_threshold_mV")
        if values["postsynaptic_learning_coordinate"] == values["spike_event_coordinate"]:
            values.pop("postsynaptic_learning_coordinate")
        if values["top_down_learning_rule_convention"] == "serialized_presynaptic":
            values.pop("top_down_learning_rule_convention")
        if values["postsynaptic_learning_timestamp"] == "emitted_event":
            values.pop("postsynaptic_learning_timestamp")
        if values["postsynaptic_signal_convention"] == "paper_equation6_literal":
            values.pop("postsynaptic_signal_convention")
        if values["ring_kernel_convention"] == "center_excluded_gaussian":
            # Preserve every historical runtime fingerprint while making the
            # newly registered alternative geometry independently traceable.
            values.pop("ring_kernel_convention")
        if values["corticoreticular_ring_kernel_convention"] is None:
            values.pop("corticoreticular_ring_kernel_convention")
        payload = json.dumps(values, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def figure6_runtime_conventions() -> FirstOrderRuntimeConventions:
    """Return the source-constrained profile used by Figure 6 validation.

    SANNDRA's archived gate revision history states that ``TGate.init()``
    resets voltage-gated currents to their resting-potential state.  The
    current executable interpretation therefore initializes every ionic gate
    at its equilibrium occupancy at the compartment's Table 3 initialization
    voltage. A literal
    zero-gate state remains available only as an explicit audit alternative.
    Input channels with four zero sensitivities are inert declarations, not
    permanent conductances. The remaining values are recovered source values
    or conventions independently constrained by the published Figure 6 maps.
    """

    return FirstOrderRuntimeConventions(
        gate_initialization_convention="steady_state_at_initial_voltage",
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
        record = replace(
            record,
            source_population="layer23_excitatory_v1",
            method="connectFromOne",
            kernel=None,
        )
    if conventions.top_down_learning_rule_convention == "paper_methods_dual_and":
        if record.id in {"modeldb112923.projection.005", "modeldb112923.projection.007"}:
            attributes = dict(record.projection_attributes)
            attributes["learningRule"] = "Dual AND gated"
            record = replace(record, projection_attributes=attributes)
    elif conventions.top_down_learning_rule_convention != "serialized_presynaptic":
        raise ValueError(
            "unsupported top-down learning-rule convention "
            f"{conventions.top_down_learning_rule_convention!r}"
        )
    return record


def _ring_kernel_convention_for_record(
    record_id: str, *, conventions: FirstOrderRuntimeConventions
) -> str:
    if (
        record_id
        in {"modeldb112923.projection.009", "modeldb112923.projection.012"}
        and conventions.corticoreticular_ring_kernel_convention is not None
    ):
        return conventions.corticoreticular_ring_kernel_convention
    return conventions.ring_kernel_convention


_TABLE3_CELL_BY_CANONICAL_STEM = {
    "thalamic_relay": "thalamic_relay",
    "trn": "trn",
    "layer5_excitatory": "layer5_excitatory",
    "layer6ii_excitatory": "layer6ii_excitatory",
    "layer6i_excitatory": "layer6i_excitatory",
    "layer4_inhibitory": "layer4_inhibitory",
    "layer23_excitatory": "layer23_excitatory",
    "layer4_excitatory": "layer4_excitatory",
    "layer23_inhibitory": "layer23_inhibitory",
    "thalamic_interneuron": "thalamic_interneuron",
    "thalamic_nonspecific": "thalamic_nonspecific",
    "thalamic_matrix": "thalamic_matrix",
}


def _table3_cell_for_population(canonical_name: str) -> CellSpec:
    """Resolve V1/V2 executable population names to the shared Table 3 class."""

    stem = canonical_name.removesuffix("_v1").removesuffix("_v2")
    try:
        table3_name = _TABLE3_CELL_BY_CANONICAL_STEM[stem]
    except KeyError as exc:  # pragma: no cover - catalog tests enumerate all populations
        raise ValueError(f"no Table 3 cell mapping for {canonical_name!r}") from exc
    return get_cell_spec(table3_name)


def resolved_intrinsic_cell(
    facts: FirstOrderPopulationFacts,
    *,
    conventions: FirstOrderRuntimeConventions,
) -> CellSpec:
    """Return an unmixed paper or recovered-executable intrinsic cell spec."""

    selected = _resolved_intrinsic_source(facts, conventions=conventions)
    if selected is IntrinsicCellConvention.MODELDB_112923:
        cell = facts.cell
    else:
        cell = _table3_cell_for_population(facts.canonical_name)
    potassium = TrnPotassiumConvention(conventions.trn_potassium_convention)
    calibrated_sodium_density = conventions.trn_soma_sodium_density_mS_cm2
    if calibrated_sodium_density is not None and (
        not math.isfinite(calibrated_sodium_density)
        or calibrated_sodium_density <= 0
    ):
        raise ValueError("TRN soma sodium density must be finite and positive")
    calibrated_potassium_density = conventions.trn_soma_potassium_density_mS_cm2
    if calibrated_potassium_density is not None:
        if (
            not math.isfinite(calibrated_potassium_density)
            or calibrated_potassium_density <= 0
        ):
            raise ValueError("TRN soma potassium density must be finite and positive")
        if potassium in {
            TrnPotassiumConvention.MODELDB_DENSITY,
            TrnPotassiumConvention.MODELDB_DENSITY_AND_REVERSAL,
        }:
            raise ValueError(
                "TRN potassium source density and calibrated density cannot both "
                "override the selected cell"
            )
    calcium = TrnCalciumSourceConvention(conventions.trn_calcium_source_convention)
    dendritic_calcium = TrnDendriticCalciumDensityConvention(
        conventions.trn_dendritic_calcium_density_convention
    )
    calibrated_density = conventions.trn_dendritic_calcium_density_mS_cm2
    if calibrated_density is not None:
        if not math.isfinite(calibrated_density) or calibrated_density <= 0:
            raise ValueError("TRN dendritic calcium density must be finite and positive")
        if dendritic_calcium is not TrnDendriticCalciumDensityConvention.SELECTED_SOURCE:
            raise ValueError(
                "TRN dendritic calcium source convention and calibrated density "
                "cannot both override the selected cell"
            )
    if facts.canonical_name != "trn":
        return cell
    compartments = list(cell.compartments)
    suffixes = []
    if calibrated_sodium_density is not None:
        compartments[0] = replace(
            compartments[0], g_na_mS_cm2=calibrated_sodium_density
        )
        suffixes.append(f"calibrated_na_{calibrated_sodium_density:g}")
    if potassium in {
        TrnPotassiumConvention.MODELDB_DENSITY,
        TrnPotassiumConvention.MODELDB_DENSITY_AND_REVERSAL,
    }:
        compartments[0] = replace(
            compartments[0], g_k_mS_cm2=facts.cell.soma.g_k_mS_cm2
        )
        suffixes.append("modeldb_k_density")
    if calibrated_potassium_density is not None:
        compartments[0] = replace(
            compartments[0], g_k_mS_cm2=calibrated_potassium_density
        )
        suffixes.append(f"calibrated_k_{calibrated_potassium_density:g}")
    if calcium in {
        TrnCalciumSourceConvention.MODELDB_SOMA_CHANNEL,
        TrnCalciumSourceConvention.MODELDB_SOMA_CHANNEL_AND_REVERSAL,
    }:
        compartments[0] = replace(
            compartments[0], g_ca_mS_cm2=facts.cell.soma.g_ca_mS_cm2
        )
        suffixes.append("modeldb_soma_calcium")
    if dendritic_calcium is TrnDendriticCalciumDensityConvention.MODELDB_100:
        archived = {item.name: item for item in facts.cell.compartments}
        compartments[1:] = [
            replace(item, g_ca_mS_cm2=archived[item.name].g_ca_mS_cm2)
            if item.g_ca_mS_cm2 is not None
            else item
            for item in compartments[1:]
        ]
        suffixes.append("modeldb_dendritic_calcium")
    elif calibrated_density is not None:
        compartments[1:] = [
            replace(item, g_ca_mS_cm2=calibrated_density)
            if item.g_ca_mS_cm2 is not None
            else item
            for item in compartments[1:]
        ]
        suffixes.append(f"calibrated_dendritic_calcium_{calibrated_density:g}")
    if not suffixes:
        return cell
    return replace(
        cell,
        name=f"{cell.name}_{'_'.join(suffixes)}",
        compartments=tuple(compartments),
    )


def _resolved_intrinsic_source(
    facts: FirstOrderPopulationFacts,
    *,
    conventions: FirstOrderRuntimeConventions,
) -> IntrinsicCellConvention:
    """Resolve a declared source profile to one population's intrinsic source."""

    selected = IntrinsicCellConvention(conventions.intrinsic_cell_convention)
    if selected in {
        IntrinsicCellConvention.MODELDB_RELAY_PAPER_TABLE3_OTHERS,
        IntrinsicCellConvention.MODELDB_RELAY_LAYER6II_PAPER_TABLE3_OTHERS,
    }:
        archived_populations = {"thalamic_relay"}
        if selected is IntrinsicCellConvention.MODELDB_RELAY_LAYER6II_PAPER_TABLE3_OTHERS:
            archived_populations.add("layer6ii_excitatory_v1")
        return (
            IntrinsicCellConvention.MODELDB_112923
            if facts.canonical_name in archived_populations
            else IntrinsicCellConvention.PAPER_TABLE3
        )
    return selected


def first_order_population_parameters(
    facts: FirstOrderPopulationFacts,
    *,
    conventions: FirstOrderRuntimeConventions,
    catalog=MODELDB_FIRST_ORDER,
) -> dict[str, Any]:
    has_ahp = facts.ahp_density_mS_cm2 is not None
    intrinsic_source = _resolved_intrinsic_source(facts, conventions=conventions)
    intrinsic_cell = resolved_intrinsic_cell(facts, conventions=conventions)
    axial_convention = conventions.axial_convention
    if axial_convention == "modeldb_relay_kinness_paper_others":
        axial_convention = (
            "kinness_serialized_edge"
            if facts.canonical_name == "thalamic_relay"
            else "paper_literal"
        )
    elif axial_convention == "modeldb_relay_trn_kinness_paper_others":
        axial_convention = (
            "kinness_serialized_edge"
            if facts.canonical_name in {"thalamic_relay", "trn"}
            else "paper_literal"
        )
    elif axial_convention == "kinness_thalamus_paper_cortex":
        axial_convention = (
            "paper_literal"
            if facts.canonical_name.startswith("layer")
            else "kinness_serialized_edge"
        )
    calcium_kinetics = CalciumKineticsConvention(conventions.calcium_kinetics_convention)
    calcium_gate_convention = conventions.calcium_gate_convention
    if (
        calcium_kinetics is CalciumKineticsConvention.MODELDB_112923
        and facts.calcium_gate_convention is not None
    ):
        calcium_gate_convention = facts.calcium_gate_convention
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
    calcium_density_convention = conventions.calcium_density_convention
    if calcium_density_convention == "trn_methods_global_250_others_table3":
        calcium_density_convention = (
            "methods_global_250" if facts.canonical_name == "trn" else "table3"
        )
    spike_event_coordinate = conventions.spike_event_coordinate
    spike_event_threshold_mV = conventions.spike_event_threshold_mV
    spike_event_release_mV = 0.0
    spike_event_voltage_offset_mV = None
    spike_event_proximal_blend_fraction = None
    if facts.canonical_name == "trn":
        if conventions.trn_spike_event_coordinate is not None:
            spike_event_coordinate = conventions.trn_spike_event_coordinate
        if conventions.trn_spike_event_threshold_mV is not None:
            spike_event_threshold_mV = conventions.trn_spike_event_threshold_mV
        if conventions.trn_spike_event_release_mV is not None:
            spike_event_release_mV = conventions.trn_spike_event_release_mV
        if not math.isfinite(spike_event_release_mV):
            raise ValueError("TRN spike-event release voltage must be finite")
        spike_event_voltage_offset_mV = conventions.trn_spike_event_voltage_offset_mV
        if spike_event_voltage_offset_mV is not None and not math.isfinite(
            spike_event_voltage_offset_mV
        ):
            raise ValueError("TRN spike-event voltage offset must be finite")
        spike_event_proximal_blend_fraction = (
            conventions.trn_spike_event_proximal_blend_fraction
        )
        if spike_event_proximal_blend_fraction is not None and (
            not math.isfinite(spike_event_proximal_blend_fraction)
            or not 0.0 <= spike_event_proximal_blend_fraction <= 1.0
        ):
            raise ValueError(
                "TRN spike-event proximal blend must be finite and between zero and one"
            )
    if facts.canonical_name == "thalamic_nonspecific":
        spike_event_proximal_blend_fraction = (
            conventions.nonspecific_spike_event_proximal_blend_fraction
        )
        if spike_event_proximal_blend_fraction is not None and (
            not math.isfinite(spike_event_proximal_blend_fraction)
            or not 0.0 <= spike_event_proximal_blend_fraction <= 1.0
        ):
            raise ValueError(
                "nonspecific spike-event proximal blend must be finite and "
                "between zero and one"
            )
    trn_axial_scale = conventions.trn_soma_proximal_axial_conductance_scale
    if not math.isfinite(trn_axial_scale) or trn_axial_scale <= 0:
        raise ValueError("TRN soma-proximal axial conductance scale must be finite and positive")
    parameters: dict[str, Any] = {
        "cell_spec": intrinsic_cell,
        "cell_class": facts.canonical_name,
        "axial_convention": axial_convention,
        "leak_convention": conventions.leak_convention,
        "voltage_coordinate": conventions.voltage_coordinate,
        "nak_rate_convention": conventions.nak_rate_convention,
        "calcium_gate_convention": calcium_gate_convention,
        "calcium_voltage_coordinate": conventions.calcium_voltage_coordinate,
        "gate_initialization_convention": conventions.gate_initialization_convention,
        "membrane_initialization_convention": conventions.membrane_initialization_convention,
        "calcium_density_convention": calcium_density_convention,
        "spike_event_coordinate": spike_event_coordinate,
        "spike_event_threshold_mV": spike_event_threshold_mV,
        "spike_event_release_mV": spike_event_release_mV,
        "spike_event_rule": conventions.spike_event_rule,
        "ahp_convention": "smart_network_112923" if has_ahp else "modeldb_112923",
        "specific_capacitance_uF_cm2": conventions.specific_capacitance_uF_cm2,
        "enable_ahp_ach": has_ahp,
        "e_na_mV": 50.0 if intrinsic_source is IntrinsicCellConvention.PAPER_TABLE3 else facts.e_na_mV,
        "e_k_mV": (
            facts.e_k_mV
            if facts.canonical_name == "trn"
            and TrnPotassiumConvention(conventions.trn_potassium_convention)
            in {
                TrnPotassiumConvention.MODELDB_REVERSAL,
                TrnPotassiumConvention.MODELDB_DENSITY_AND_REVERSAL,
            }
            else (
                -90.0
                if intrinsic_source is IntrinsicCellConvention.PAPER_TABLE3
                else facts.e_k_mV
            )
        ),
        "e_ca_mV": (
            facts.e_ca_mV
            if facts.canonical_name == "trn"
            and TrnCalciumSourceConvention(conventions.trn_calcium_source_convention)
            in {
                TrnCalciumSourceConvention.MODELDB_REVERSAL,
                TrnCalciumSourceConvention.MODELDB_SOMA_CHANNEL_AND_REVERSAL,
            }
            else (
                180.0
                if intrinsic_source is IntrinsicCellConvention.PAPER_TABLE3
                else facts.e_ca_mV
            )
        ),
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
    if facts.canonical_name == "trn":
        parameters["axial_edge_conductance_scales"] = (trn_axial_scale, 1.0)
        if spike_event_voltage_offset_mV is not None:
            parameters["spike_event_voltage_offset_mV"] = spike_event_voltage_offset_mV
        if spike_event_proximal_blend_fraction is not None:
            parameters["spike_event_proximal_blend_fraction"] = (
                spike_event_proximal_blend_fraction
            )
    elif (
        facts.canonical_name == "thalamic_nonspecific"
        and spike_event_proximal_blend_fraction is not None
    ):
        parameters["spike_event_proximal_blend_fraction"] = (
            spike_event_proximal_blend_fraction
        )
    if has_ahp:
        soma = intrinsic_cell.soma
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
                gaussian_spread_convention=conventions.gaussian_spread_convention,
                ring_kernel_convention=_ring_kernel_convention_for_record(
                    record.id, conventions=conventions
                ),
                gaussian_learning_bounds_convention=(
                    conventions.gaussian_learning_bounds_convention
                ),
                spike_event_coordinate=conventions.spike_event_coordinate,
                spike_event_threshold_mV=conventions.spike_event_threshold_mV,
                postsynaptic_learning_threshold_mV=(
                    conventions.postsynaptic_learning_threshold_mV
                ),
                postsynaptic_learning_coordinate=(
                    conventions.postsynaptic_learning_coordinate
                ),
                postsynaptic_learning_timestamp=(
                    conventions.postsynaptic_learning_timestamp
                ),
                postsynaptic_signal_convention=(
                    conventions.postsynaptic_signal_convention
                ),
                postsynaptic_depression_scale_convention=(
                    conventions.postsynaptic_depression_scale_convention
                ),
            )
        else:
            projection = connect_modeldb_gap_junction(
                **kwargs,
                gaussian_weight_convention=conventions.gaussian_weight_convention,
                gaussian_spread_convention=conventions.gaussian_spread_convention,
                ring_kernel_convention=conventions.ring_kernel_convention,
            )
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
    instrument_learning_terms: bool = False,
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
            gaussian_spread_convention=resolved_conventions.gaussian_spread_convention,
            ring_kernel_convention=_ring_kernel_convention_for_record(
                record.id, conventions=resolved_conventions
            ),
            gaussian_learning_bounds_convention=(
                resolved_conventions.gaussian_learning_bounds_convention
            ),
            spike_event_coordinate=resolved_conventions.spike_event_coordinate,
            spike_event_threshold_mV=resolved_conventions.spike_event_threshold_mV,
            postsynaptic_learning_threshold_mV=(
                resolved_conventions.postsynaptic_learning_threshold_mV
            ),
            postsynaptic_learning_coordinate=(
                resolved_conventions.postsynaptic_learning_coordinate
            ),
            postsynaptic_learning_timestamp=(
                resolved_conventions.postsynaptic_learning_timestamp
            ),
            postsynaptic_signal_convention=(
                resolved_conventions.postsynaptic_signal_convention
            ),
            postsynaptic_depression_scale_convention=(
                resolved_conventions.postsynaptic_depression_scale_convention
            ),
            instrument_learning_terms=instrument_learning_terms,
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
    # A generated per-neuron expression works in both Brian runtime and C++
    # standalone. Python callbacks and NumPy-index assignments cannot be
    # serialized reliably by the standalone device before its first build.
    mask = "+".join(f"int(i == {index})" for index in clamped_relay_indices)
    voltage_name = f"v_{compartment}"
    pin_code = (
        f"{voltage_name} = {voltage_name}"
        f" + ({mask})*({float(holding_mV)!r}*mV-{voltage_name})"
    )
    clamp_start = relay.run_regularly(
        pin_code, when="start", name="smart_relay_voltage_clamp_start"
    )
    clamp_end = relay.run_regularly(
        pin_code, when="end", name="smart_relay_voltage_clamp_end"
    )
    sector.network.add(clamp_start, clamp_end)
    return sector


def build_first_order_connected_sector(
    *,
    conventions: FirstOrderRuntimeConventions | None = None,
    gate_initialization_convention: str | None = None,
    instrument_learning_terms: bool = False,
    brian=None,
) -> FirstOrderSector:
    """Build chemical and electrical connectivity; external inputs remain separate."""

    if brian is None:
        import brian2 as brian

    if conventions is not None and gate_initialization_convention is not None:
        raise ValueError("pass conventions or gate_initialization_convention, not both")
    resolved_conventions = conventions or FirstOrderRuntimeConventions(
        gate_initialization_convention=(
            gate_initialization_convention or "steady_state_at_initial_voltage"
        )
    )
    sector = build_first_order_chemical_sector(
        conventions=resolved_conventions,
        instrument_learning_terms=instrument_learning_terms,
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
            gaussian_weight_convention=resolved_conventions.gaussian_weight_convention,
            gaussian_spread_convention=resolved_conventions.gaussian_spread_convention,
            ring_kernel_convention=resolved_conventions.ring_kernel_convention,
            brian=brian,
        )
    sector.projections.update(electrical)
    sector.network.add(*electrical.values())
    return sector
