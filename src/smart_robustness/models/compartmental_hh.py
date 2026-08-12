"""Executable vectorized multicompartment SMART populations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

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


class SpikeEventCoordinate(StrEnum):
    """Voltage coordinate used by SMART's two-stage spike event rule."""

    ABSOLUTE_PHYSICAL = "absolute_physical"
    RELATIVE_TO_SOMA_LEAK = "relative_to_soma_leak"


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
    calcium_density = CalciumDensityConvention(params["calcium_density_convention"])
    spike_coordinate = SpikeEventCoordinate(params["spike_event_coordinate"])
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
    spike_reset = "armed = 0"
    if enable_ahp_ach:
        spike_reset += "; ahp_rise += 1; ahp_fall += 1"
    if compiled.depletion_enabled:
        spike_reset += f"; transmitter *= {1 - float(depletion_epsilon)}"
    spike_voltage = (
        "v_soma"
        if spike_coordinate is SpikeEventCoordinate.ABSOLUTE_PHYSICAL
        else "v_soma-e_l_soma"
    )
    group = brian.NeuronGroup(
        size,
        compiled.equations,
        threshold=f"armed > 0.5 and {spike_voltage} < 0*mV",
        reset=spike_reset,
        events={"arm_spike": f"armed < 0.5 and {spike_voltage} > 30*mV"},
        method=params.get("method", "exponential_euler"),
        name=name,
    )
    group.run_on_event("arm_spike", "armed = 1", when="after_thresholds", order=1)
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
    group.armed = 0
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
        _set(group, f"{port.name}_rise", 0)
        _set(group, f"{port.name}_fall", 0)
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
        initial_voltage = params.get("v_init_mV", compartment.e_leak_mV)
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
                initial_voltage
                if calcium_gate is TTypeGateConvention.MODELDB_112923
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

    for edge_index, edge in enumerate(build_axial_edges(cell, axial)):
        _set(
            group,
            f"g_ax_{edge_index}_into_{edge.near.compartment_name}",
            edge.conductance_into_near_nS * brian.nsiemens,
        )
        _set(
            group,
            f"g_ax_{edge_index}_into_{edge.far.compartment_name}",
            edge.conductance_into_far_nS * brian.nsiemens,
        )
    for compartment_name, clamp_mV in voltage_clamps_mV.items():
        _set(group, f"v_{compartment_name}", float(clamp_mV) * brian.mV)
    return CompartmentalPopulation(group=group, cell_spec=cell, compiled=compiled)
