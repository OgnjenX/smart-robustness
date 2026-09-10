"""Model-neutral population contract for complete SMART networks."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .equation_builder import CompiledCellEquations
from .table3 import CellSpec


@runtime_checkable
class SmartPopulationAdapter(Protocol):
    """State surface required by SMART circuits, protocols, and analyses."""

    group: Any
    cell_spec: CellSpec
    compiled: CompiledCellEquations

    @property
    def compartments(self) -> tuple[str, ...]: ...

    def trigger_ach(self, indices: Any = slice(None)) -> None: ...

    def set_external_input(
        self,
        record_id: str,
        channel: str,
        value: float,
        indices: Any = slice(None),
    ) -> None: ...

    def set_convergent_external_input(
        self,
        record_id: str,
        channel: str,
        source_values: Any,
        indices: Any = slice(None),
    ) -> None: ...

    def set_external_injection(
        self,
        record_id: str,
        channel: str,
        value: float,
        indices: Any = slice(None),
    ) -> None: ...


def validate_smart_population_adapter(population: SmartPopulationAdapter) -> None:
    """Fail before simulation when an alternative backend drops SMART state."""

    if not isinstance(population, SmartPopulationAdapter):
        raise TypeError("population does not implement the SMART adapter methods")
    if population.compartments != population.compiled.compartments:
        raise ValueError("population and compiled compartment declarations differ")

    variables = population.group.variables
    required = {
        "last_spike_onset",
        "i_drive_soma",
        "clear_drive_on_spike",
        "drive_spikes_until_clear",
    }
    for compartment in population.compartments:
        required.update(
            {
                f"v_{compartment}",
                f"C_{compartment}",
                f"g_l_{compartment}",
                f"e_l_{compartment}",
                f"i_syn_{compartment}",
                f"i_drive_{compartment}",
                f"i_membrane_inward_{compartment}",
                f"i_axial_inward_{compartment}",
                f"i_transmembrane_paper_{compartment}",
                f"i_transmembrane_outward_{compartment}",
            }
        )
    for port in population.compiled.synaptic_ports:
        required.update(
            {
                f"{port.name}_gate",
                f"{port.name}_block",
                f"i_{port.name}",
                f"g_{port.name}",
                f"e_{port.name}",
            }
        )
    for port in population.compiled.gap_junction_ports:
        required.add(f"i_{port.name}")
    missing = sorted(required - set(variables))
    if missing:
        raise ValueError(f"population is missing SMART adapter variables: {missing}")
