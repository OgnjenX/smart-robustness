"""Executable vectorized multicompartment SMART populations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .axial import AxialConvention, build_axial_edges
from .currents import (
    E_CA_MV,
    E_K_MV,
    E_NA_MV,
    G_CA_MSIEMENS_CM2,
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
from .table3 import CellSpec, get_cell_spec


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

        if self.cell_spec.name != "layer5_excitatory":
            raise ValueError("ACh modulation is specified only for layer5_excitatory cells")
        self.group.ach_rise[indices] += 1
        self.group.ach_fall[indices] += 1


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

    cell = get_cell_spec(params["cell_class"])
    axial = AxialConvention(params["axial_convention"])
    leak = LeakConvention(params["leak_convention"])
    voltage = VoltageCoordinate(params["voltage_coordinate"])
    calcium_gate = TTypeGateConvention(params["calcium_gate_convention"])
    calcium_density = CalciumDensityConvention(params["calcium_density_convention"])
    if cell.name == "layer5_excitatory" and "ahp_max_conductance_nS" not in params:
        raise ValueError(
            "layer5_excitatory requires explicit ahp_max_conductance_nS; "
            "Grossberg & Versace (2008) do not report a unique value"
        )
    compiled = compile_cell_equations(
        cell,
        axial_convention=axial,
        leak_convention=leak,
        voltage_coordinate=voltage,
        calcium_gate_convention=calcium_gate,
        calcium_density_convention=calcium_density,
    )
    spike_reset = "armed = 0"
    if cell.name == "layer5_excitatory":
        spike_reset += "; ahp_rise += 1; ahp_fall += 1"
    group = brian.NeuronGroup(
        size,
        compiled.equations,
        threshold="armed > 0.5 and v_soma < 0*mV",
        reset=spike_reset,
        events={"arm_spike": "armed < 0.5 and v_soma > 30*mV"},
        method=params.get("method", "exponential_euler"),
        name=name,
    )
    group.run_on_event("arm_spike", "armed = 1", when="after_thresholds", order=1)
    if cell.name == "layer5_excitatory":
        group.g_ahp_max = float(params["ahp_max_conductance_nS"]) * brian.nsiemens
        group.ahp_rise = 0
        group.ahp_fall = 0
        group.ach_rise = 0
        group.ach_fall = 0
    group.armed = 0
    group.e_na = E_NA_MV * brian.mV
    group.e_k = E_K_MV * brian.mV
    group.e_ca = E_CA_MV * brian.mV

    for compartment in cell.compartments:
        compartment_name = compartment.name
        _set(group, f"C_{compartment_name}", compartment.capacitance_pF() * brian.pfarad)
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
        paper_voltage = (
            initial_voltage
            if voltage is VoltageCoordinate.ABSOLUTE
            else initial_voltage - compartment.e_leak_mV
        )
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
            rates = traub_miles_rates(paper_voltage)
            _set(group, f"m_{compartment_name}", rates.alpha_m / (rates.alpha_m + rates.beta_m))
            _set(group, f"h_{compartment_name}", rates.alpha_h / (rates.alpha_h + rates.beta_h))
            _set(group, f"n_{compartment_name}", rates.alpha_n / (rates.alpha_n + rates.beta_n))
        if compartment.g_ca_mS_cm2 is not None:
            density = (
                compartment.g_ca_mS_cm2
                if calcium_density is CalciumDensityConvention.TABLE3
                else G_CA_MSIEMENS_CM2
            )
            total_nS = density * compartment.lateral_area_cm2 * 1e6
            _set(group, f"g_ca_{compartment_name}", total_nS * brian.nsiemens)
            _set(group, f"m_ca_{compartment_name}", t_type_m_inf(paper_voltage, calcium_gate))
            _set(group, f"h_ca_{compartment_name}", t_type_h_inf(paper_voltage, calcium_gate))

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
    return CompartmentalPopulation(group=group, cell_spec=cell, compiled=compiled)
