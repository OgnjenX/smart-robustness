"""Projection-specific postsynaptic ports for classic SMART cells."""

from __future__ import annotations

from dataclasses import dataclass

from ..projections import ConnectionKind, ProjectionRecord, Receptor
from .currents import biexponential_normalization


@dataclass(frozen=True, slots=True)
class SynapticPortSpec:
    name: str
    record_id: str
    compartment: str
    receptor: Receptor
    reversal_mV: float
    conductance_density_mS_cm2: float
    rise_ms: float
    fall_ms: float
    normalization: float
    voltage_block: bool


@dataclass(frozen=True, slots=True)
class GapJunctionPortSpec:
    name: str
    record_id: str
    compartment: str
    conductance_density_mS_cm2: float


@dataclass(frozen=True, slots=True)
class ExternalInputPortSpec:
    name: str
    record_id: str
    compartment: str
    conductance_density_mS_cm2: float
    reversal_mV: float


def chemical_port(record: ProjectionRecord, index: int) -> SynapticPortSpec:
    """Convert one audited chemical record into an executable receptor port."""

    if record.kind is not ConnectionKind.CHEMICAL:
        raise ValueError(f"{record.id}: only chemical records define kinetic ports")
    parsed = record.parsed
    required = {
        "target compartment": record.target.compartment,
        "reversal": parsed.reversal_mV,
        "conductance": parsed.conductance_pS,
        "rise": parsed.rise_ms,
        "fall": parsed.fall_ms,
    }
    missing = tuple(label for label, value in required.items() if value is None)
    if missing:
        raise ValueError(f"{record.id}: missing executable values: {', '.join(missing)}")
    return SynapticPortSpec(
        name=f"port_{index:03d}",
        record_id=record.id,
        compartment=str(record.target.compartment),
        receptor=parsed.receptor,
        reversal_mV=float(parsed.reversal_mV),
        # The recovered supplement calls this conductance, while SMART.nml
        # serializes the same numeric value as channel g_bar in mS/cm^2.
        conductance_density_mS_cm2=float(parsed.conductance_pS),
        rise_ms=float(parsed.rise_ms),
        fall_ms=float(parsed.fall_ms),
        normalization=biexponential_normalization(float(parsed.rise_ms), float(parsed.fall_ms))
        if parsed.rise_ms != parsed.fall_ms
        else 1.0,
        voltage_block=parsed.receptor is Receptor.NMDA,
    )


def ports_for_target(
    records: tuple[ProjectionRecord, ...], target_population: str
) -> tuple[SynapticPortSpec, ...]:
    chemical = tuple(
        record
        for record in records
        if record.kind is ConnectionKind.CHEMICAL and record.target.population == target_population
    )
    return tuple(chemical_port(record, index) for index, record in enumerate(chemical))


def gap_ports_for_target(
    records: tuple[ProjectionRecord, ...], target_population: str
) -> tuple[GapJunctionPortSpec, ...]:
    gap_records = tuple(
        record
        for record in records
        if record.kind is ConnectionKind.GAP_JUNCTION
        and record.target.population == target_population
    )
    ports: list[GapJunctionPortSpec] = []
    for index, record in enumerate(gap_records):
        if record.target.compartment is None or record.parsed.conductance_pS is None:
            raise ValueError(f"{record.id}: incomplete gap-junction record")
        ports.append(
            GapJunctionPortSpec(
                name=f"gap_{index:03d}",
                record_id=record.id,
                compartment=record.target.compartment,
                conductance_density_mS_cm2=record.parsed.conductance_pS,
            )
        )
    return tuple(ports)


def external_ports_for_target(
    records: tuple[ProjectionRecord, ...], target_population: str
) -> tuple[ExternalInputPortSpec, ...]:
    external = tuple(
        record
        for record in records
        if record.kind is ConnectionKind.EXTERNAL_INPUT
        and record.target.population == target_population
    )
    ports: list[ExternalInputPortSpec] = []
    for index, record in enumerate(external):
        if (
            record.target.compartment is None
            or record.parsed.conductance_pS is None
            or record.parsed.reversal_mV is None
        ):
            raise ValueError(f"{record.id}: incomplete external-input record")
        ports.append(
            ExternalInputPortSpec(
                name=f"external_{index:03d}",
                record_id=record.id,
                compartment=record.target.compartment,
                conductance_density_mS_cm2=record.parsed.conductance_pS,
                reversal_mV=record.parsed.reversal_mV,
            )
        )
    return tuple(ports)
