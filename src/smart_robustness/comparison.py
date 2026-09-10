"""Controlled neuron-model comparison contracts for complete SMART sectors."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


def _array_digest(values: Any) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    payload = str(array.dtype).encode() + repr(array.shape).encode() + array.tobytes()
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class PopulationContract:
    name: str
    size: int
    compartments: tuple[str, ...]
    synaptic_records: tuple[str, ...]
    gap_records: tuple[str, ...]
    external_records: tuple[str, ...]
    injection_records: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ProjectionContract:
    record_id: str
    edge_count: int
    source_indices_sha256: str
    target_indices_sha256: str
    delay_sha256: str | None
    initial_strength_sha256: str | None


@dataclass(frozen=True, slots=True)
class SectorContract:
    populations: tuple[PopulationContract, ...]
    projections: tuple[ProjectionContract, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def snapshot_sector_contract(sector: Any) -> SectorContract:
    """Capture every model-invariant structural feature of a built sector."""

    populations = tuple(
        PopulationContract(
            name=name,
            size=int(population.group.N),
            compartments=population.compartments,
            synaptic_records=tuple(
                port.record_id for port in population.compiled.synaptic_ports
            ),
            gap_records=tuple(
                port.record_id for port in population.compiled.gap_junction_ports
            ),
            external_records=tuple(
                port.record_id for port in population.compiled.external_input_ports
            ),
            injection_records=tuple(
                port.record_id for port in population.compiled.injection_ports
            ),
        )
        for name, population in sorted(sector.populations.items())
    )
    projections = []
    for record_id, projection in sorted(sector.projections.items()):
        variables = projection.variables
        delay = None
        if "axonal_delay" in variables:
            delay = _array_digest(projection.axonal_delay[:])
        else:
            try:
                delay = _array_digest(projection.delay[:])
            except (AttributeError, TypeError):
                pass
        strength = None
        for candidate in ("w", "w_baseline", "g"):
            if candidate in variables:
                strength = _array_digest(getattr(projection, candidate)[:])
                break
        projections.append(
            ProjectionContract(
                record_id=record_id,
                edge_count=len(projection.i[:]),
                source_indices_sha256=_array_digest(projection.i[:]),
                target_indices_sha256=_array_digest(projection.j[:]),
                delay_sha256=delay,
                initial_strength_sha256=strength,
            )
        )
    return SectorContract(populations=populations, projections=tuple(projections))


def assert_controlled_substitution(
    classic: SectorContract,
    alternative: SectorContract,
) -> None:
    """Reject a comparison arm that changes anything structural."""

    if classic != alternative:
        raise ValueError(
            "alternative neuron model changed the SMART sector contract; "
            "topology, ports, delays, and initial strengths must remain fixed"
        )
