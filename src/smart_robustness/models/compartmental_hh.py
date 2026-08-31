"""Executable vectorized multicompartment SMART populations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from .axial import AxialConvention, build_axial_edges
from .currents import (
    E_CA_MV,
    E_K_MV,
    E_NA_MV,
    G_CA_MSIEMENS_CM2,
    AHPConvention,
    NaKRateConvention,
    TTypeGateConvention,
    t_type_h_inf,
    t_type_m_inf,
    traub_miles_rates,
)
from .equation_builder import (
    CalciumDensityConvention,
    CalciumVoltageCoordinate,
    CompiledCellEquations,
    LeakConvention,
    VoltageCoordinate,
    compile_cell_equations,
)
from .ports import ExternalInputPortSpec, GapJunctionPortSpec, InjectionPortSpec, SynapticPortSpec
from .table3 import CellSpec, get_cell_spec


class GateInitializationConvention(StrEnum):
    """Named alternatives for a KInNeSS default absent from the SMART archive."""

    STEADY_STATE_AT_INITIAL_VOLTAGE = "steady_state_at_initial_voltage"
    ZERO = "zero"


class MembraneInitializationConvention(StrEnum):
    """Coordinate used for the serialized compartment resting state."""

    PHYSICAL_LEAK_VOLTAGE = "physical_leak_voltage"
    KINNESS_INTERNAL_ZERO = "kinness_internal_zero"


class SpikeEventCoordinate(StrEnum):
    """Voltage coordinate used by SMART's two-stage spike event rule."""

    ABSOLUTE_PHYSICAL = "absolute_physical"
    SHIFTED_67_MV = "shifted_67_mV"
    RELATIVE_TO_SOMA_LEAK = "relative_to_soma_leak"


class SpikeEventRule(StrEnum):
    """Alternative readings of SMART Equation 8's preceding-voltage term."""

    LATCHED_PEAK_THEN_ZERO = "latched_peak_then_zero"
    HYSTERETIC_THRESHOLD_THEN_ZERO = "hysteretic_threshold_then_zero"
    LITERAL_PREVIOUS_SAMPLE = "literal_previous_sample"


@dataclass(slots=True)
class CompartmentalPopulation:
    group: Any
    cell_spec: CellSpec
    compiled: CompiledCellEquations

    @property
    def compartments(self) -> tuple[str, ...]:
        return self.compiled.compartments

    def trigger_ach(self, indices: Any = slice(None)) -> None:
        """Apply one normalized ACh event to layer-5 cells."""

        if not self.compiled.ahp_ach_enabled:
            raise ValueError("ACh modulation is not enabled for this population")
        self.group.ach_rise[indices] += 1
        self.group.ach_fall[indices] += 1

    def set_external_input(
        self,
        record_id: str,
        channel: str,
        value: float,
        indices: Any = slice(None),
    ) -> None:
        """Set one KInNeSS external input channel in its documented 0--255 range."""

        port = next(
            (port for port in self.compiled.external_input_ports if port.record_id == record_id),
            None,
        )
        if port is None:
            raise KeyError(f"no external input port for {record_id!r}")
        if channel not in {"red", "green", "blue", "alpha"}:
            raise ValueError("channel must be red, green, blue, or alpha")
        if not 0 <= value <= 255:
            raise ValueError("external input value must be between zero and 255")
        getattr(self.group, f"{port.name}_input_{channel}")[indices] = value
        getattr(self.group, f"{port.name}_input_source_count")[indices] = 1

    def set_convergent_external_input(
        self,
        record_id: str,
        channel: str,
        source_values: Any,
        indices: Any = slice(None),
    ) -> None:
        """Sum valid pixel values for a KInNeSS ``connectFromAll`` input gate."""

        values = np.asarray(source_values, dtype=float)
        if values.size == 0 or np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 255)):
            raise ValueError("each external input source must be between zero and 255")
        port = next(
            port for port in self.compiled.external_input_ports if port.record_id == record_id
        )
        for source_channel in ("red", "green", "blue", "alpha"):
            getattr(self.group, f"{port.name}_input_{source_channel}")[indices] = 0.0
        getattr(self.group, f"{port.name}_input_{channel}")[indices] = float(values.sum())
        getattr(self.group, f"{port.name}_input_source_count")[indices] = values.size

    def set_external_injection(
        self,
        record_id: str,
        channel: str,
        value: float,
        indices: Any = slice(None),
    ) -> None:
        """Set one KInNeSS Equation 6 current-source channel."""

        port = next(
            (port for port in self.compiled.injection_ports if port.record_id == record_id),
            None,
        )
        if port is None:
            raise KeyError(f"no external injection port for {record_id!r}")
        if channel not in {"red", "green", "blue", "alpha"}:
            raise ValueError("channel must be red, green, blue, or alpha")
        if not 0 <= value <= 255:
            raise ValueError("external injection value must be between zero and 255")
        getattr(self.group, f"{port.name}_input_{channel}")[indices] = value


def _set(group: Any, name: str, value: Any) -> None:
    setattr(group, name, value)


def create_compartmental_hh_population(
    *,
    name: str,
    size: int,
    params: dict[str, Any],
    brian=None,
) -> CompartmentalPopulation:
    """Create a Table 3 population with all fidelity conventions explicit."""

    if brian is None:
        import brian2 as brian

    cell = params.get("cell_spec")
    if cell is None:
        cell = get_cell_spec(params["cell_class"])
    elif not isinstance(cell, CellSpec):
        raise TypeError("cell_spec must be an explicit CellSpec instance")
    axial = AxialConvention(params["axial_convention"])
    leak = LeakConvention(params["leak_convention"])
    voltage = VoltageCoordinate(params["voltage_coordinate"])
    nak_rate = NaKRateConvention(params["nak_rate_convention"])
    calcium_gate = TTypeGateConvention(params["calcium_gate_convention"])
    gate_initialization = GateInitializationConvention(params["gate_initialization_convention"])
    membrane_initialization = MembraneInitializationConvention(
        params["membrane_initialization_convention"]
    )
    calcium_density = CalciumDensityConvention(params["calcium_density_convention"])
    calcium_voltage_coordinate = CalciumVoltageCoordinate(
        params["calcium_voltage_coordinate"]
    )
    spike_coordinate = SpikeEventCoordinate(params["spike_event_coordinate"])
    spike_event_voltage_offset_mV = params.get("spike_event_voltage_offset_mV")
    spike_event_proximal_blend_fraction = params.get(
        "spike_event_proximal_blend_fraction"
    )
    if spike_event_voltage_offset_mV is not None:
        spike_event_voltage_offset_mV = float(spike_event_voltage_offset_mV)
        if not np.isfinite(spike_event_voltage_offset_mV):
            raise ValueError("spike_event_voltage_offset_mV must be finite")
        if spike_coordinate is not SpikeEventCoordinate.ABSOLUTE_PHYSICAL:
            raise ValueError(
                "numeric spike-event offset cannot be combined with a named "
                "non-absolute coordinate"
            )
    if spike_event_proximal_blend_fraction is not None:
        spike_event_proximal_blend_fraction = float(
            spike_event_proximal_blend_fraction
        )
        if not np.isfinite(spike_event_proximal_blend_fraction) or not (
            0.0 <= spike_event_proximal_blend_fraction <= 1.0
        ):
            raise ValueError(
                "spike_event_proximal_blend_fraction must be finite and between zero and one"
            )
        if spike_event_voltage_offset_mV is not None:
            raise ValueError(
                "proximal spike-event blend cannot be combined with a numeric voltage offset"
            )
        if spike_coordinate is not SpikeEventCoordinate.ABSOLUTE_PHYSICAL:
            raise ValueError(
                "proximal spike-event blend cannot be combined with a named "
                "non-absolute coordinate"
            )
    spike_event_rule = SpikeEventRule(params["spike_event_rule"])
    spike_event_threshold_mV = float(params["spike_event_threshold_mV"])
    spike_event_release_mV = float(params.get("spike_event_release_mV", 0.0))
    if not np.isfinite(spike_event_threshold_mV):
        raise ValueError("spike_event_threshold_mV must be finite")
    if not np.isfinite(spike_event_release_mV):
        raise ValueError("spike_event_release_mV must be finite")
    ahp_convention = AHPConvention(params["ahp_convention"])
    synaptic_ports = params.get("synaptic_ports", ())
    if not isinstance(synaptic_ports, tuple) or not all(
        isinstance(port, SynapticPortSpec) for port in synaptic_ports
    ):
        raise TypeError("synaptic_ports must be a tuple of SynapticPortSpec")
    gap_junction_ports = params.get("gap_junction_ports", ())
    if not isinstance(gap_junction_ports, tuple) or not all(
        isinstance(port, GapJunctionPortSpec) for port in gap_junction_ports
    ):
        raise TypeError("gap_junction_ports must be a tuple of GapJunctionPortSpec")
    external_input_ports = params.get("external_input_ports", ())
    if not isinstance(external_input_ports, tuple) or not all(
        isinstance(port, ExternalInputPortSpec) for port in external_input_ports
    ):
        raise TypeError("external_input_ports must be a tuple of ExternalInputPortSpec")
    injection_ports = params.get("injection_ports", ())
    if not isinstance(injection_ports, tuple) or not all(
        isinstance(port, InjectionPortSpec) for port in injection_ports
    ):
        raise TypeError("injection_ports must be a tuple of InjectionPortSpec")
    depletion_epsilon = params.get("depletion_epsilon")
    depletion_recovery_ms = params.get("depletion_recovery_ms")
    voltage_clamps_mV = params.get("voltage_clamps_mV", {})
    if not isinstance(voltage_clamps_mV, dict):
        raise TypeError("voltage_clamps_mV must be a dict")
    compartment_names = {compartment.name for compartment in cell.compartments}
    if (
        spike_event_proximal_blend_fraction is not None
        and "proximal_dendrite" not in compartment_names
    ):
        raise ValueError(
            "proximal spike-event blend requires a proximal_dendrite compartment"
        )
    unknown_clamps = set(voltage_clamps_mV) - compartment_names
    if unknown_clamps:
        raise ValueError(f"unknown voltage-clamped compartments: {sorted(unknown_clamps)}")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in voltage_clamps_mV.values()
    ):
        raise ValueError("voltage clamp values must be finite numbers")
    enable_ahp_ach = params["enable_ahp_ach"]
    if not isinstance(enable_ahp_ach, bool):
        raise TypeError("enable_ahp_ach must be an explicit bool")
    specific_capacitance = float(params["specific_capacitance_uF_cm2"])
    if specific_capacitance <= 0:
        raise ValueError("specific_capacitance_uF_cm2 must be positive")
    if enable_ahp_ach and "ahp_max_conductance_nS" not in params:
        raise ValueError(
            "AHP/ACh-enabled cells require explicit ahp_max_conductance_nS; "
            "Grossberg & Versace (2008) do not report a unique value"
        )
    if enable_ahp_ach and "ahp_event_weight" not in params:
        raise ValueError("AHP/ACh-enabled cells require explicit ahp_event_weight")
    if enable_ahp_ach and "e_ahp_mV" not in params:
        raise ValueError("AHP/ACh-enabled cells require explicit e_ahp_mV")
    compiled = compile_cell_equations(
        cell,
        axial_convention=axial,
        leak_convention=leak,
        voltage_coordinate=voltage,
        nak_rate_convention=nak_rate,
        calcium_gate_convention=calcium_gate,
        calcium_voltage_coordinate=calcium_voltage_coordinate,
        calcium_density_convention=calcium_density,
        ahp_convention=ahp_convention,
        enable_ahp_ach=enable_ahp_ach,
        synaptic_ports=synaptic_ports,
        gap_junction_ports=gap_junction_ports,
        external_input_ports=external_input_ports,
        injection_ports=injection_ports,
        voltage_clamped_compartments=frozenset(voltage_clamps_mV),
        depletion_epsilon=depletion_epsilon,
        depletion_recovery_ms=depletion_recovery_ms,
    )
    # Protocols can request a one-event somatic current pulse. The flag
    # defaults to zero, preserving sustained-current behavior exactly.
    spike_reset = (
        "armed = 0; "
        "i_drive_soma *= (1-clear_drive_on_spike); "
        "i_drive_soma *= int(drive_spikes_until_clear != 1); "
        "drive_spikes_until_clear -= int(drive_spikes_until_clear > 0)"
    )
    if enable_ahp_ach:
        spike_reset += "; ahp_rise += 1; ahp_fall += 1"
    if compiled.depletion_enabled:
        spike_reset += f"; transmitter *= {1 - float(depletion_epsilon)}"
    if spike_event_proximal_blend_fraction is not None:
        spike_voltage = (
            "v_soma+"
            f"({spike_event_proximal_blend_fraction!r})*"
            "(v_proximal_dendrite-v_soma)"
        )
    elif spike_event_voltage_offset_mV is not None:
        spike_voltage = f"v_soma+({spike_event_voltage_offset_mV!r})*mV"
    elif spike_coordinate is SpikeEventCoordinate.ABSOLUTE_PHYSICAL:
        spike_voltage = "v_soma"
    elif spike_coordinate is SpikeEventCoordinate.SHIFTED_67_MV:
        spike_voltage = "v_soma+67*mV"
    else:
        spike_voltage = "v_soma-e_l_soma"
    group_kwargs: dict[str, Any] = {
        "method": params.get("method", "exponential_euler"),
        "name": name,
    }
    # Expose the exact coordinate used by Equation 8 as a read-only state
    # subexpression. Thresholds still use the same algebra, while pathway
    # diagnostics can now distinguish membrane dynamics from detector state.
    equations = compiled.equations + (
        f"\nspike_detector_voltage = {spike_voltage} : volt"
        "\nclear_drive_on_spike : 1 (constant)"
        "\ndrive_spikes_until_clear : integer"
    )
    if spike_event_rule in {
        SpikeEventRule.LATCHED_PEAK_THEN_ZERO,
        SpikeEventRule.HYSTERETIC_THRESHOLD_THEN_ZERO,
    }:
        events = {
            "arm_spike": (
                "armed == 0 and spike_detector_voltage > "
                f"{spike_event_threshold_mV}*mV"
            )
        }
        if spike_event_rule is SpikeEventRule.HYSTERETIC_THRESHOLD_THEN_ZERO:
            spike_reset = spike_reset.replace("armed = 0", "armed = -1", 1)
            events["release_spike_detector"] = (
                "armed < -0.5 and spike_detector_voltage < "
                f"{spike_event_threshold_mV}*mV"
            )
        group_kwargs.update(
            threshold=(
                "armed > 0.5 and spike_detector_voltage < "
                f"{spike_event_release_mV}*mV"
            ),
            events=events,
        )
    else:
        equations += "\nprevious_spike_voltage : volt"
        group_kwargs["threshold"] = (
            f"spike_detector_voltage < {spike_event_release_mV}*mV and "
            f"previous_spike_voltage > {spike_event_threshold_mV}*mV"
        )
    group = brian.NeuronGroup(size, equations, reset=spike_reset, **group_kwargs)
    group.clear_drive_on_spike = 0
    group.drive_spikes_until_clear = 0
    if spike_event_rule in {
        SpikeEventRule.LATCHED_PEAK_THEN_ZERO,
        SpikeEventRule.HYSTERETIC_THRESHOLD_THEN_ZERO,
    }:
        group.run_on_event(
            "arm_spike",
            "armed = 1; last_spike_onset = t",
            when="after_thresholds",
            order=1,
        )
        if spike_event_rule is SpikeEventRule.HYSTERETIC_THRESHOLD_THEN_ZERO:
            group.run_on_event(
                "release_spike_detector", "armed = 0", when="after_thresholds", order=2
            )
    else:
        # Equation 8 literally writes V(t-dt), not a remembered action-potential
        # peak. Capture the current source-coordinate voltage after all events so
        # the next threshold pass can evaluate that printed expression exactly.
        group.run_regularly(
            "previous_spike_voltage = spike_detector_voltage", when="end", order=1
        )
    if compiled.depletion_enabled:
        group.transmitter = 1
    if enable_ahp_ach:
        group.g_ahp_max = float(params["ahp_max_conductance_nS"]) * brian.nsiemens
        group.ahp_event_weight = float(params["ahp_event_weight"])
        group.e_ahp = float(params["e_ahp_mV"]) * brian.mV
        group.ahp_rise = 0
        group.ahp_fall = 0
        group.ach_rise = 0
        group.ach_fall = 0
    # SMART Equation 8 emits on the falling phase: first remember a sample
    # above V_theta, then release one event when the soma returns below 0 mV.
    # The source value remains the default; an explicit alternative release
    # coordinate is available only for separately labeled detector calibration.
    group.armed = 0
    group.last_spike_onset = -1 * brian.second
    if spike_event_rule is SpikeEventRule.LITERAL_PREVIOUS_SAMPLE:
        default_soma_voltage = (
            0.0
            if membrane_initialization is MembraneInitializationConvention.KINNESS_INTERNAL_ZERO
            else cell.soma.e_leak_mV
        )
        initial_soma_voltage = params.get("v_init_mV", default_soma_voltage)
        if spike_event_proximal_blend_fraction is not None:
            default_proximal_voltage = (
                0.0
                if membrane_initialization
                is MembraneInitializationConvention.KINNESS_INTERNAL_ZERO
                else cell.compartment("proximal_dendrite").e_leak_mV
            )
            initial_proximal_voltage = params.get(
                "v_init_mV", default_proximal_voltage
            )
            initial_soma_voltage += spike_event_proximal_blend_fraction * (
                initial_proximal_voltage - initial_soma_voltage
            )
        elif spike_event_voltage_offset_mV is not None:
            initial_soma_voltage += spike_event_voltage_offset_mV
        elif spike_coordinate is SpikeEventCoordinate.SHIFTED_67_MV:
            initial_soma_voltage += 67.0
        elif spike_coordinate is SpikeEventCoordinate.RELATIVE_TO_SOMA_LEAK:
            initial_soma_voltage -= (
                0.0 if leak is LeakConvention.PRINTED_ZERO else cell.soma.e_leak_mV
            )
        group.previous_spike_voltage = initial_soma_voltage * brian.mV
    group.e_na = float(params.get("e_na_mV", E_NA_MV)) * brian.mV
    group.e_k = float(params.get("e_k_mV", E_K_MV)) * brian.mV
    group.e_ca = float(params.get("e_ca_mV", E_CA_MV)) * brian.mV
    compartments_by_name = {compartment.name: compartment for compartment in cell.compartments}
    for port in synaptic_ports:
        compartment = compartments_by_name[port.compartment]
        _set(
            group,
            f"g_{port.name}",
            port.conductance_density_mS_cm2 * compartment.lateral_area_cm2 * 1e6 * brian.nsiemens,
        )
        _set(group, f"e_{port.name}", port.reversal_mV * brian.mV)
        _set(group, f"{port.name}_gate", 0)
    for port in gap_junction_ports:
        _set(group, f"i_{port.name}", 0 * brian.pA)
    for port in external_input_ports:
        compartment = compartments_by_name[port.compartment]
        _set(
            group,
            f"g_{port.name}",
            port.conductance_density_mS_cm2 * compartment.lateral_area_cm2 * 1e6 * brian.nsiemens,
        )
        for channel in ("red", "green", "blue", "alpha"):
            _set(group, f"{port.name}_input_{channel}", 0)
        _set(group, f"{port.name}_input_source_count", 1)
    for port in injection_ports:
        for channel in ("red", "green", "blue", "alpha"):
            _set(group, f"{port.name}_input_{channel}", 0)

    for compartment in cell.compartments:
        compartment_name = compartment.name
        _set(
            group,
            f"C_{compartment_name}",
            compartment.capacitance_pF(specific_capacitance) * brian.pfarad,
        )
        _set(
            group,
            f"g_l_{compartment_name}",
            compartment.conductance_nS("leak") * brian.nsiemens,
        )
        leak_reversal = 0.0 if leak is LeakConvention.PRINTED_ZERO else compartment.e_leak_mV
        _set(group, f"e_l_{compartment_name}", leak_reversal * brian.mV)
        default_initial_voltage = (
            0.0
            if membrane_initialization is MembraneInitializationConvention.KINNESS_INTERNAL_ZERO
            else compartment.e_leak_mV
        )
        initial_voltage = params.get("v_init_mV", default_initial_voltage)
        _set(group, f"v_{compartment_name}", initial_voltage * brian.mV)
        _set(group, f"i_syn_{compartment_name}", 0 * brian.pA)
        _set(group, f"i_drive_{compartment_name}", 0 * brian.pA)
        if voltage is VoltageCoordinate.ABSOLUTE:
            paper_voltage = initial_voltage
        elif voltage is VoltageCoordinate.SHIFTED_67_MV:
            paper_voltage = initial_voltage + 67.0
        else:
            paper_voltage = initial_voltage - compartment.e_leak_mV
        if compartment.g_na_mS_cm2 is not None:
            _set(
                group,
                f"g_na_{compartment_name}",
                compartment.conductance_nS("na") * brian.nsiemens,
            )
            _set(
                group,
                f"g_k_{compartment_name}",
                compartment.conductance_nS("k") * brian.nsiemens,
            )
            if gate_initialization is GateInitializationConvention.ZERO:
                _set(group, f"m_{compartment_name}", 0)
                _set(group, f"h_{compartment_name}", 0)
                _set(group, f"n_{compartment_name}", 0)
            else:
                rates = traub_miles_rates(paper_voltage, nak_rate)
                _set(
                    group,
                    f"m_{compartment_name}",
                    rates.alpha_m / (rates.alpha_m + rates.beta_m),
                )
                _set(
                    group,
                    f"h_{compartment_name}",
                    rates.alpha_h / (rates.alpha_h + rates.beta_h),
                )
                _set(
                    group,
                    f"n_{compartment_name}",
                    rates.alpha_n / (rates.alpha_n + rates.beta_n),
                )
        if compartment.g_ca_mS_cm2 is not None:
            density = (
                compartment.g_ca_mS_cm2
                if calcium_density is CalciumDensityConvention.TABLE3
                else G_CA_MSIEMENS_CM2
            )
            total_nS = density * compartment.lateral_area_cm2 * 1e6
            _set(group, f"g_ca_{compartment_name}", total_nS * brian.nsiemens)
            calcium_voltage = (
                (
                    initial_voltage + compartment.e_leak_mV
                    if calcium_voltage_coordinate
                    is CalciumVoltageCoordinate.INTERNAL_ZERO_PLUS_SERIALIZED_LEAK
                    else initial_voltage
                )
                if calcium_gate
                in {
                    TTypeGateConvention.MODELDB_112923,
                    TTypeGateConvention.MODELDB_RETICULAR_112923,
                }
                else paper_voltage
            )
            if gate_initialization is GateInitializationConvention.ZERO:
                _set(group, f"m_ca_{compartment_name}", 0)
                _set(group, f"h_ca_{compartment_name}", 0)
            else:
                _set(
                    group,
                    f"m_ca_{compartment_name}",
                    t_type_m_inf(calcium_voltage, calcium_gate),
                )
                _set(
                    group,
                    f"h_ca_{compartment_name}",
                    t_type_h_inf(calcium_voltage, calcium_gate),
                )

    axial_edges = build_axial_edges(cell, axial)
    axial_edge_scales = params.get(
        "axial_edge_conductance_scales", (1.0,) * len(axial_edges)
    )
    if (
        not isinstance(axial_edge_scales, tuple)
        or len(axial_edge_scales) != len(axial_edges)
        or any(
            isinstance(scale, bool)
            or not isinstance(scale, (int, float))
            or not math.isfinite(scale)
            or scale <= 0
            for scale in axial_edge_scales
        )
    ):
        raise ValueError(
            "axial_edge_conductance_scales must contain one finite positive "
            "number per adjacent edge"
        )
    for edge_index, (edge, scale) in enumerate(
        zip(axial_edges, axial_edge_scales, strict=True)
    ):
        _set(
            group,
            f"g_ax_{edge_index}_into_{edge.near.compartment_name}",
            edge.conductance_into_near_nS * scale * brian.nsiemens,
        )
        _set(
            group,
            f"g_ax_{edge_index}_into_{edge.far.compartment_name}",
            edge.conductance_into_far_nS * scale * brian.nsiemens,
        )
    for compartment_name, clamp_mV in voltage_clamps_mV.items():
        _set(group, f"v_{compartment_name}", float(clamp_mV) * brian.mV)
    return CompartmentalPopulation(group=group, cell_spec=cell, compiled=compiled)
