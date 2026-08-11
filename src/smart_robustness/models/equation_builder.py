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
    AHP_NORMALIZATION,
    AHP_RISE_MS,
    TTypeGateConvention,
)
from .table3 import CellSpec


class LeakConvention(StrEnum):
    PRINTED_ZERO = "printed_zero"
    TABLE3_REVERSAL = "table3_reversal"


class VoltageCoordinate(StrEnum):
    ABSOLUTE = "absolute"
    RELATIVE_TO_TABLE3_LEAK = "relative_to_table3_leak"


class CalciumDensityConvention(StrEnum):
    TABLE3 = "table3"
    METHODS_GLOBAL_250 = "methods_global_250"


@dataclass(frozen=True, slots=True)
class CompiledCellEquations:
    cell_name: str
    equations: str
    axial_convention: AxialConvention
    leak_convention: LeakConvention
    voltage_coordinate: VoltageCoordinate
    calcium_gate_convention: TTypeGateConvention
    calcium_density_convention: CalciumDensityConvention
    compartments: tuple[str, ...]
    axial_parameter_names: tuple[str, ...]


def _enum(value, enum_type, label):
    if not isinstance(value, enum_type):
        raise TypeError(f"{label} must be an explicit {enum_type.__name__} member")
    return value


def _paper_voltage(name: str, coordinate: VoltageCoordinate) -> str:
    if coordinate is VoltageCoordinate.ABSOLUTE:
        return f"v_{name}"
    return f"v_{name}-e_l_{name}"


def _nak_lines(name: str, coordinate: VoltageCoordinate) -> list[str]:
    v = f"v_{name}"
    vp = _paper_voltage(name, coordinate)
    return [
        f"dm_{name}/dt = alpha_m_{name}*(1-m_{name})-beta_m_{name}*m_{name} : 1",
        f"dh_{name}/dt = alpha_h_{name}*(1-h_{name})-beta_h_{name}*h_{name} : 1",
        f"dn_{name}/dt = alpha_n_{name}*(1-n_{name})-beta_n_{name}*n_{name} : 1",
        f"alpha_m_{name}=0.128/exprel((13*mV-({vp}))/(4*mV))/ms : Hz",
        f"beta_m_{name}=1.4/exprel((({vp})-40*mV)/(5*mV))/ms : Hz",
        f"alpha_h_{name}=0.128*exp((27*mV-({vp}))/(18*mV))/ms : Hz",
        f"beta_h_{name}=4/(exp((40*mV-({vp}))/(5*mV))+1)/ms : Hz",
        f"alpha_n_{name}=0.16/exprel((15*mV-({vp}))/(5*mV))/ms : Hz",
        f"beta_n_{name}=0.5*exp((10*mV-({vp}))/(40*mV))/ms : Hz",
        f"i_na_{name}=g_na_{name}*m_{name}**3*h_{name}*(e_na-{v}) : amp",
        f"i_k_{name}=g_k_{name}*n_{name}**4*(e_k-{v}) : amp",
        f"g_na_{name} : siemens (constant)",
        f"g_k_{name} : siemens (constant)",
    ]


def _calcium_lines(
    name: str, coordinate: VoltageCoordinate, gate: TTypeGateConvention
) -> list[str]:
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


def _layer5_ahp_lines() -> list[str]:
    """Published AHP/ACh waveform with an explicitly calibrated conductance."""

    return [
        f"dahp_rise/dt=-ahp_rise/({AHP_RISE_MS}*ms) : 1",
        f"dahp_fall/dt=-ahp_fall/({AHP_FALL_MS}*ms) : 1",
        f"ahp_gate={AHP_NORMALIZATION}*(ahp_fall-ahp_rise) : 1",
        f"dach_rise/dt=-ach_rise/({ACH_RISE_MS}*ms) : 1",
        f"dach_fall/dt=-ach_fall/({ACH_FALL_MS}*ms) : 1",
        f"ach_gate=clip({ACH_NORMALIZATION}*(ach_fall-ach_rise), 0, 1) : 1",
        "i_ahp=g_ahp_max*ahp_gate*(1-ach_gate)*(e_k-v_soma) : amp",
        "g_ahp_max : siemens (constant)",
    ]


def compile_cell_equations(
    cell: CellSpec,
    *,
    axial_convention: AxialConvention,
    leak_convention: LeakConvention,
    voltage_coordinate: VoltageCoordinate,
    calcium_gate_convention: TTypeGateConvention,
    calcium_density_convention: CalciumDensityConvention,
) -> CompiledCellEquations:
    """Compile one source-specified cell; every ambiguous convention is required."""

    axial = _enum(axial_convention, AxialConvention, "axial_convention")
    leak = _enum(leak_convention, LeakConvention, "leak_convention")
    voltage = _enum(voltage_coordinate, VoltageCoordinate, "voltage_coordinate")
    calcium_gate = _enum(calcium_gate_convention, TTypeGateConvention, "calcium_gate_convention")
    calcium_density = _enum(
        calcium_density_convention,
        CalciumDensityConvention,
        "calcium_density_convention",
    )
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
    if cell.name == "layer5_excitatory":
        lines.extend(_layer5_ahp_lines())
    for compartment in cell.compartments:
        name = compartment.name
        current_terms = [f"g_l_{name}*(e_l_{name}-v_{name})"]
        if compartment.g_na_mS_cm2 is not None:
            lines.extend(_nak_lines(name, voltage))
            current_terms.extend((f"i_na_{name}", f"i_k_{name}"))
        if compartment.g_ca_mS_cm2 is not None:
            lines.extend(_calcium_lines(name, voltage, calcium_gate))
            current_terms.append(f"i_ca_{name}")
        if cell.name == "layer5_excitatory" and name == "soma":
            current_terms.append("i_ahp")
        current_terms.extend(axial_terms[name])
        current_terms.extend((f"i_syn_{name}", f"i_drive_{name}"))
        lines.extend(
            (
                f"dv_{name}/dt=({' + '.join(current_terms)})/C_{name} : volt",
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
        calcium_gate_convention=calcium_gate,
        calcium_density_convention=calcium_density,
        compartments=tuple(c.name for c in cell.compartments),
        axial_parameter_names=tuple(axial_parameters),
    )
