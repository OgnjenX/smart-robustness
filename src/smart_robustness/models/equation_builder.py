"""Compile Table 3 cells into explicit vectorized Brian2 equations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .axial import AxialConvention, build_axial_edges
from .currents import (
    ACH_FALL_MS,
    ACH_NORMALIZATION,
    ACH_RISE_MS,
    AHP_FALL_MS,
    AHP_MODELDB_FALL_MS,
    AHP_MODELDB_NORMALIZATION,
    AHP_NETWORK_FALL_MS,
    AHP_NETWORK_NORMALIZATION,
    AHP_NETWORK_RISE_MS,
    AHP_NORMALIZATION,
    AHP_RISE_MS,
    AHPConvention,
    NaKRateConvention,
    TTypeGateConvention,
)
from .ports import ExternalInputPortSpec, GapJunctionPortSpec, InjectionPortSpec, SynapticPortSpec
from .table3 import CellSpec


class LeakConvention(StrEnum):
    PRINTED_ZERO = "printed_zero"
    TABLE3_REVERSAL = "table3_reversal"


class VoltageCoordinate(StrEnum):
    ABSOLUTE = "absolute"
    RELATIVE_TO_TABLE3_LEAK = "relative_to_table3_leak"
    SHIFTED_67_MV = "shifted_67_mV"


class CalciumDensityConvention(StrEnum):
    TABLE3 = "table3"
    METHODS_GLOBAL_250 = "methods_global_250"


class CalciumVoltageCoordinate(StrEnum):
    """Voltage supplied to serialized absolute-voltage T-current kinetics."""

    INTEGRATED_VOLTAGE = "integrated_voltage"
    INTERNAL_ZERO_PLUS_SERIALIZED_LEAK = "internal_zero_plus_serialized_leak"


@dataclass(frozen=True, slots=True)
class CompiledCellEquations:
    cell_name: str
    equations: str
    axial_convention: AxialConvention
    leak_convention: LeakConvention
    voltage_coordinate: VoltageCoordinate
    nak_rate_convention: NaKRateConvention
    calcium_gate_convention: TTypeGateConvention
    calcium_voltage_coordinate: CalciumVoltageCoordinate
    calcium_density_convention: CalciumDensityConvention
    ahp_convention: AHPConvention
    ahp_ach_enabled: bool
    compartments: tuple[str, ...]
    axial_parameter_names: tuple[str, ...]
    synaptic_ports: tuple[SynapticPortSpec, ...]
    gap_junction_ports: tuple[GapJunctionPortSpec, ...]
    external_input_ports: tuple[ExternalInputPortSpec, ...]
    injection_ports: tuple[InjectionPortSpec, ...]
    voltage_clamped_compartments: frozenset[str]
    depletion_enabled: bool


def _enum(value, enum_type, label):
    if not isinstance(value, enum_type):
        raise TypeError(f"{label} must be an explicit {enum_type.__name__} member")
    return value


def _paper_voltage(name: str, coordinate: VoltageCoordinate) -> str:
    if coordinate is VoltageCoordinate.ABSOLUTE:
        return f"v_{name}"
    if coordinate is VoltageCoordinate.SHIFTED_67_MV:
        return f"v_{name}+67*mV"
    return f"v_{name}-e_l_{name}"


def _nak_lines(
    name: str, coordinate: VoltageCoordinate, convention: NaKRateConvention
) -> list[str]:
    v = f"v_{name}"
    vp = _paper_voltage(name, coordinate)
    alpha_m_scale = 1.28 if convention is NaKRateConvention.STANDARD_TRAUB_MILES else 0.128
    alpha_h_offset = 17 if convention is NaKRateConvention.STANDARD_TRAUB_MILES else 27
    return [
        f"dm_{name}/dt = alpha_m_{name}*(1-m_{name})-beta_m_{name}*m_{name} : 1",
        f"dh_{name}/dt = alpha_h_{name}*(1-h_{name})-beta_h_{name}*h_{name} : 1",
        f"dn_{name}/dt = alpha_n_{name}*(1-n_{name})-beta_n_{name}*n_{name} : 1",
        f"alpha_m_{name}={alpha_m_scale}/exprel((13*mV-({vp}))/(4*mV))/ms : Hz",
        f"beta_m_{name}=1.4/exprel((({vp})-40*mV)/(5*mV))/ms : Hz",
        f"alpha_h_{name}=0.128*exp(({alpha_h_offset}*mV-({vp}))/(18*mV))/ms : Hz",
        f"beta_h_{name}=4/(exp((40*mV-({vp}))/(5*mV))+1)/ms : Hz",
        f"alpha_n_{name}=0.16/exprel((15*mV-({vp}))/(5*mV))/ms : Hz",
        f"beta_n_{name}=0.5*exp((10*mV-({vp}))/(40*mV))/ms : Hz",
        f"i_na_{name}=g_na_{name}*m_{name}**3*h_{name}*(e_na-{v}) : amp",
        f"i_k_{name}=g_k_{name}*n_{name}**4*(e_k-{v}) : amp",
        f"g_na_{name} : siemens (constant)",
        f"g_k_{name} : siemens (constant)",
    ]


def _calcium_lines(
    name: str,
    coordinate: VoltageCoordinate,
    gate: TTypeGateConvention,
    calcium_voltage_coordinate: CalciumVoltageCoordinate,
    serialized_leak_mV: float,
) -> list[str]:
    membrane_voltage = f"v_{name}"
    modeldb_voltage = (
        f"(v_{name}+({serialized_leak_mV})*mV)"
        if calcium_voltage_coordinate
        is CalciumVoltageCoordinate.INTERNAL_ZERO_PLUS_SERIALIZED_LEAK
        else f"v_{name}"
    )
    if gate is TTypeGateConvention.MODELDB_RETICULAR_112923:
        v = modeldb_voltage
        return [
            f"dm_ca_{name}/dt=(m_ca_inf_{name}-m_ca_{name})/tau_m_ca_{name} : 1",
            f"dh_ca_{name}/dt=(h_ca_inf_{name}-h_ca_{name})/tau_h_ca_{name} : 1",
            f"m_ca_inf_{name}=1/(exp((-52*mV-{v})/(7.4*mV))+1) : 1",
            (
                f"tau_m_ca_{name}=(1+0.33/(exp(({v}+27*mV)/(10*mV))"
                f"+exp(({v}+102*mV)/(-15*mV))))*ms : second"
            ),
            f"h_ca_inf_{name}=1/(exp((-80*mV-{v})/(-5*mV))+1) : 1",
            (
                f"tau_h_ca_{name}=(28.3+0.33/(exp(({v}+48*mV)/(4*mV))"
                f"+exp(({v}+407*mV)/(-50*mV))))*ms : second"
            ),
            f"i_ca_{name}=g_ca_{name}*m_ca_{name}**2*h_ca_{name}*(e_ca-{membrane_voltage}) : amp",
            f"g_ca_{name} : siemens (constant)",
        ]
    if gate is TTypeGateConvention.MODELDB_112923:
        v = modeldb_voltage
        return [
            f"dm_ca_{name}/dt=(m_ca_inf_{name}-m_ca_{name})/tau_m_ca_{name} : 1",
            f"dh_ca_{name}/dt=(h_ca_inf_{name}-h_ca_{name})/tau_h_ca_{name} : 1",
            f"m_ca_inf_{name}=1/(exp((-63*mV-{v})/(7.8*mV))+1) : 1",
            f"tau_m_ca_{name}=(2.44+2.506e-2*exp(-9.84e-2*{v}/mV))*ms : second",
            f"h_ca_inf_{name}=1/(exp(({v}+83.5*mV)/(6.3*mV))+1) : 1",
            f"tau_h_ca_{name}=(19.15+7.171e-2*exp(-10.54e-2*{v}/mV))*ms : second",
            f"i_ca_{name}=g_ca_{name}*m_ca_{name}**3*h_ca_{name}*(e_ca-{membrane_voltage}) : amp",
            f"g_ca_{name} : siemens (constant)",
        ]
    vp = _paper_voltage(name, coordinate)
    m_literal = f"2.44+2.506e-2*exp(-9.84e-2*({vp})/mV)"
    h_literal = f"19.5+7.171e-2*exp(-10.54e-2*({vp})/mV)"
    if gate is TTypeGateConvention.RECIPROCAL:
        m_inf, h_inf = f"1/({m_literal})", f"1/({h_literal})"
    else:
        m_inf, h_inf = m_literal, h_literal
    return [
        f"dm_ca_{name}/dt=(m_ca_inf_{name}-m_ca_{name})/tau_m_ca_{name} : 1",
        f"dh_ca_{name}/dt=(h_ca_inf_{name}-h_ca_{name})/tau_h_ca_{name} : 1",
        f"tau_m_ca_{name}=1/(exp((-63*mV-({vp}))/(7.8*mV))+1)*ms : second",
        f"tau_h_ca_{name}=1/(exp((-83*mV-({vp}))/(6.3*mV))+1)*ms : second",
        f"m_ca_inf_{name}={m_inf} : 1",
        f"h_ca_inf_{name}={h_inf} : 1",
        f"i_ca_{name}=g_ca_{name}*m_ca_{name}**3*h_ca_{name}*(e_ca-v_{name}) : amp",
        f"g_ca_{name} : siemens (constant)",
    ]


def _layer5_ahp_lines(convention: AHPConvention) -> list[str]:
    """Published AHP/ACh waveform with an explicitly calibrated conductance."""

    if convention is AHPConvention.MODELDB_112923:
        rise_ms = AHP_RISE_MS
        fall_ms = AHP_MODELDB_FALL_MS
        normalization = AHP_MODELDB_NORMALIZATION
    elif convention is AHPConvention.SMART_NETWORK_112923:
        rise_ms = AHP_NETWORK_RISE_MS
        fall_ms = AHP_NETWORK_FALL_MS
        normalization = AHP_NETWORK_NORMALIZATION
    else:
        rise_ms = AHP_RISE_MS
        fall_ms = AHP_FALL_MS
        normalization = AHP_NORMALIZATION

    return [
        f"dahp_rise/dt=-ahp_rise/({rise_ms}*ms) : 1",
        f"dahp_fall/dt=-ahp_fall/({fall_ms}*ms) : 1",
        f"ahp_gate={normalization}*(ahp_fall-ahp_rise) : 1",
        f"dach_rise/dt=-ach_rise/({ACH_RISE_MS}*ms) : 1",
        f"dach_fall/dt=-ach_fall/({ACH_FALL_MS}*ms) : 1",
        f"ach_gate=clip({ACH_NORMALIZATION}*(ach_fall-ach_rise), 0, 1) : 1",
        "i_ahp=g_ahp_max*ahp_event_weight*ahp_gate*(1-ach_gate)*(e_ahp-v_soma) : amp",
        "g_ahp_max : siemens (constant)",
        "ahp_event_weight : 1 (constant)",
        "e_ahp : volt (constant)",
    ]


def compile_cell_equations(
    cell: CellSpec,
    *,
    axial_convention: AxialConvention,
    leak_convention: LeakConvention,
    voltage_coordinate: VoltageCoordinate,
    nak_rate_convention: NaKRateConvention,
    calcium_gate_convention: TTypeGateConvention,
    calcium_voltage_coordinate: CalciumVoltageCoordinate,
    calcium_density_convention: CalciumDensityConvention,
    ahp_convention: AHPConvention,
    enable_ahp_ach: bool,
    synaptic_ports: tuple[SynapticPortSpec, ...] = (),
    gap_junction_ports: tuple[GapJunctionPortSpec, ...] = (),
    external_input_ports: tuple[ExternalInputPortSpec, ...] = (),
    injection_ports: tuple[InjectionPortSpec, ...] = (),
    voltage_clamped_compartments: frozenset[str] = frozenset(),
    depletion_epsilon: float | None = None,
    depletion_recovery_ms: float | None = None,
) -> CompiledCellEquations:
    """Compile one source-specified cell; every ambiguous convention is required."""

    axial = _enum(axial_convention, AxialConvention, "axial_convention")
    leak = _enum(leak_convention, LeakConvention, "leak_convention")
    voltage = _enum(voltage_coordinate, VoltageCoordinate, "voltage_coordinate")
    nak_rate = _enum(nak_rate_convention, NaKRateConvention, "nak_rate_convention")
    calcium_gate = _enum(calcium_gate_convention, TTypeGateConvention, "calcium_gate_convention")
    calcium_voltage = _enum(
        calcium_voltage_coordinate,
        CalciumVoltageCoordinate,
        "calcium_voltage_coordinate",
    )
    calcium_density = _enum(
        calcium_density_convention,
        CalciumDensityConvention,
        "calcium_density_convention",
    )
    ahp = _enum(ahp_convention, AHPConvention, "ahp_convention")
    if not isinstance(enable_ahp_ach, bool):
        raise TypeError("enable_ahp_ach must be an explicit bool")
    if not isinstance(synaptic_ports, tuple) or not all(
        isinstance(port, SynapticPortSpec) for port in synaptic_ports
    ):
        raise TypeError("synaptic_ports must be an explicit tuple of SynapticPortSpec")
    compartment_names = {compartment.name for compartment in cell.compartments}
    for port in synaptic_ports:
        if port.compartment not in compartment_names:
            raise ValueError(f"{port.record_id}: unknown target compartment {port.compartment}")
    if not isinstance(gap_junction_ports, tuple) or not all(
        isinstance(port, GapJunctionPortSpec) for port in gap_junction_ports
    ):
        raise TypeError("gap_junction_ports must be a tuple of GapJunctionPortSpec")
    for port in gap_junction_ports:
        if port.compartment not in compartment_names:
            raise ValueError(f"{port.record_id}: unknown target compartment {port.compartment}")
    if not isinstance(external_input_ports, tuple) or not all(
        isinstance(port, ExternalInputPortSpec) for port in external_input_ports
    ):
        raise TypeError("external_input_ports must be a tuple of ExternalInputPortSpec")
    for port in external_input_ports:
        if port.compartment not in compartment_names:
            raise ValueError(f"{port.record_id}: unknown target compartment {port.compartment}")
    if not isinstance(injection_ports, tuple) or not all(
        isinstance(port, InjectionPortSpec) for port in injection_ports
    ):
        raise TypeError("injection_ports must be a tuple of InjectionPortSpec")
    for port in injection_ports:
        if port.compartment not in compartment_names:
            raise ValueError(f"{port.record_id}: unknown target compartment {port.compartment}")
    if not isinstance(voltage_clamped_compartments, frozenset):
        raise TypeError("voltage_clamped_compartments must be a frozenset")
    unknown_clamps = voltage_clamped_compartments - compartment_names
    if unknown_clamps:
        raise ValueError(f"unknown voltage-clamped compartments: {sorted(unknown_clamps)}")
    depletion_enabled = depletion_epsilon is not None or depletion_recovery_ms is not None
    if depletion_enabled:
        if depletion_epsilon is None or depletion_recovery_ms is None:
            raise ValueError("depletion epsilon and recovery must be supplied together")
        if not 0 <= depletion_epsilon <= 1 or depletion_recovery_ms <= 0:
            raise ValueError("invalid source-backed transmitter depletion parameters")
    edges = build_axial_edges(cell, axial)
    axial_terms: dict[str, list[str]] = {c.name: [] for c in cell.compartments}
    axial_parameters: list[str] = []
    for index, edge in enumerate(edges):
        near, far = edge.near.compartment_name, edge.far.compartment_name
        near_parameter = f"g_ax_{index}_into_{near}"
        far_parameter = f"g_ax_{index}_into_{far}"
        axial_terms[near].append(f"{near_parameter}*(v_{far}-v_{near})")
        axial_terms[far].append(f"{far_parameter}*(v_{near}-v_{far})")
        axial_parameters.extend((near_parameter, far_parameter))

    lines = [
        "e_na : volt (constant)",
        "e_k : volt (constant)",
        "e_ca : volt (constant)",
        "armed : 1",
    ]
    for parameter in axial_parameters:
        lines.append(f"{parameter} : siemens (constant)")
    if enable_ahp_ach:
        lines.extend(_layer5_ahp_lines(ahp))
    if depletion_enabled:
        lines.append(f"dtransmitter/dt=(1-transmitter)/({depletion_recovery_ms}*ms) : 1")
    for port in synaptic_ports:
        block = f"1/(1+0.33*exp(-v_{port.compartment}/(16.7*mV)))" if port.voltage_block else "1"
        lines.extend(
            (
                f"{port.name}_gate : 1",
                f"{port.name}_block={block} : 1",
                f"i_{port.name}=g_{port.name}*{port.name}_gate*{port.name}_block*(e_{port.name}-v_{port.compartment}) : amp",
                f"g_{port.name} : siemens (constant)",
                f"e_{port.name} : volt (constant)",
            )
        )
    for port in gap_junction_ports:
        lines.append(f"i_{port.name} : amp")
    for port in external_input_ports:
        effective_reversal = "+".join(
            (
                f"e_l_{port.compartment}",
                f"{port.reversal_mV}*mV",
                *(
                    f"{sensitivity}*mV*{port.name}_input_{channel}"
                    for sensitivity, channel in zip(
                        port.sensitivities_mV,
                        ("red", "green", "blue", "alpha"),
                        strict=True,
                    )
                ),
            )
        )
        lines.extend(
            (
                f"e_{port.name}_effective={effective_reversal} : volt",
                f"i_{port.name}=g_{port.name}*(e_{port.name}_effective-v_{port.compartment}) : amp",
                f"g_{port.name} : siemens (constant)",
                f"{port.name}_input_red : 1",
                f"{port.name}_input_green : 1",
                f"{port.name}_input_blue : 1",
                f"{port.name}_input_alpha : 1",
            )
        )
    for port in injection_ports:
        area_cm2 = cell.compartment(port.compartment).lateral_area_cm2
        current = "+".join(
            f"{sensitivity * area_cm2}*pA*{port.name}_input_{channel}"
            for sensitivity, channel in zip(
                port.sensitivities_pA_cm2,
                ("red", "green", "blue", "alpha"),
                strict=True,
            )
        )
        lines.extend(
            (
                f"i_{port.name}={current} : amp",
                f"{port.name}_input_red : 1",
                f"{port.name}_input_green : 1",
                f"{port.name}_input_blue : 1",
                f"{port.name}_input_alpha : 1",
            )
        )
    for compartment in cell.compartments:
        name = compartment.name
        membrane_current_terms = [f"g_l_{name}*(e_l_{name}-v_{name})"]
        if compartment.g_na_mS_cm2 is not None:
            lines.extend(_nak_lines(name, voltage, nak_rate))
            membrane_current_terms.extend((f"i_na_{name}", f"i_k_{name}"))
        if compartment.g_ca_mS_cm2 is not None:
            lines.extend(
                _calcium_lines(
                    name,
                    voltage,
                    calcium_gate,
                    calcium_voltage,
                    compartment.e_leak_mV,
                )
            )
            membrane_current_terms.append(f"i_ca_{name}")
        if enable_ahp_ach and name == "soma":
            membrane_current_terms.append("i_ahp")
        membrane_current_terms.extend(
            f"i_{port.name}" for port in synaptic_ports if port.compartment == name
        )
        membrane_current_terms.extend(
            f"i_{port.name}" for port in gap_junction_ports if port.compartment == name
        )
        membrane_current_terms.extend(
            f"i_{port.name}" for port in external_input_ports if port.compartment == name
        )
        membrane_current_terms.extend(
            f"i_{port.name}" for port in injection_ports if port.compartment == name
        )
        membrane_current_terms.extend((f"i_syn_{name}", f"i_drive_{name}"))
        axial_expression = " + ".join(axial_terms[name]) or "0*amp"
        membrane_expression = " + ".join(membrane_current_terms)
        current_terms = membrane_current_terms + axial_terms[name]
        voltage_equation = (
            f"dv_{name}/dt=0*volt/second : volt"
            if name in voltage_clamped_compartments
            else f"dv_{name}/dt=({' + '.join(current_terms)})/C_{name} : volt"
        )
        lines.extend(
            (
                voltage_equation,
                f"i_membrane_inward_{name}={membrane_expression} : amp",
                f"i_axial_inward_{name}={axial_expression} : amp",
                f"i_transmembrane_paper_{name}=i_axial_inward_{name} : amp",
                f"i_transmembrane_outward_{name}=-i_axial_inward_{name} : amp",
                f"C_{name} : farad (constant)",
                f"g_l_{name} : siemens (constant)",
                f"e_l_{name} : volt (constant)",
                f"i_syn_{name} : amp",
                f"i_drive_{name} : amp",
            )
        )
    return CompiledCellEquations(
        cell_name=cell.name,
        equations="\n".join(lines),
        axial_convention=axial,
        leak_convention=leak,
        voltage_coordinate=voltage,
        nak_rate_convention=nak_rate,
        calcium_gate_convention=calcium_gate,
        calcium_voltage_coordinate=calcium_voltage,
        calcium_density_convention=calcium_density,
        ahp_convention=ahp,
        ahp_ach_enabled=enable_ahp_ach,
        compartments=tuple(c.name for c in cell.compartments),
        axial_parameter_names=tuple(axial_parameters),
        synaptic_ports=synaptic_ports,
        gap_junction_ports=gap_junction_ports,
        external_input_ports=external_input_ports,
        injection_ports=injection_ports,
        voltage_clamped_compartments=voltage_clamped_compartments,
        depletion_enabled=depletion_enabled,
    )
