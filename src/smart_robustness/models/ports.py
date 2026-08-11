"""Projection-specific postsynaptic ports for classic SMART cells."""

from __future__ import annotations

from dataclasses import dataclass

from ..modeldb_projections import ModelDBExternalChannel, ModelDBProjection
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
    sensitivities_mV: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True, slots=True)
class InjectionPortSpec:
    name: str
    record_id: str
    compartment: str
    sensitivities_pA_cm2: tuple[float, float, float, float]


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


def _modeldb_receptor(record: ModelDBProjection) -> Receptor:
    channel = record.channel_name.upper()
    if "GABA" in channel:
        return Receptor.GABA
    if "NMDA" in channel:
        return Receptor.NMDA
    return Receptor.AMPA


def modeldb_chemical_port(record: ModelDBProjection, index: int) -> SynapticPortSpec:
    if record.kind != "chemical":
        raise ValueError(f"{record.id}: only chemical records define kinetic ports")
    required = (
        record.channel_conductance_mS_cm2,
        record.reversal_mV,
        record.rise_ms,
        record.fall_ms,
    )
    if any(value is None for value in required):
        raise ValueError(f"{record.id}: incomplete executable chemical projection")
    rise_ms = float(record.rise_ms)
    fall_ms = float(record.fall_ms)
    receptor = _modeldb_receptor(record)
    return SynapticPortSpec(
        name=f"port_{index:03d}",
        record_id=record.id,
        compartment=record.target_compartment,
        receptor=receptor,
        reversal_mV=float(record.reversal_mV),
        conductance_density_mS_cm2=float(record.channel_conductance_mS_cm2),
        rise_ms=rise_ms,
        fall_ms=fall_ms,
        normalization=biexponential_normalization(rise_ms, fall_ms) if rise_ms != fall_ms else 1.0,
        voltage_block=receptor is Receptor.NMDA,
    )


def modeldb_ports_for_target(
    records: tuple[ModelDBProjection, ...], target_population: str
) -> tuple[SynapticPortSpec, ...]:
    chemical = tuple(
        record
        for record in records
        if record.kind == "chemical" and record.target_population == target_population
    )
    return tuple(modeldb_chemical_port(record, index) for index, record in enumerate(chemical))


def modeldb_gap_ports_for_target(
    records: tuple[ModelDBProjection, ...], target_population: str
) -> tuple[GapJunctionPortSpec, ...]:
    gaps = tuple(
        record
        for record in records
        if record.kind == "gap_junction" and record.target_population == target_population
    )
    return tuple(
        GapJunctionPortSpec(
            name=f"gap_{index:03d}",
            record_id=record.id,
            compartment=record.target_compartment,
            conductance_density_mS_cm2=float(record.channel_conductance_mS_cm2),
        )
        for index, record in enumerate(gaps)
        if record.channel_conductance_mS_cm2 is not None
    )


def modeldb_external_ports_for_target(
    records: tuple[ModelDBExternalChannel, ...], target_population: str
) -> tuple[ExternalInputPortSpec, ...]:
    channels = tuple(
        record
        for record in records
        if record.dependency == "input" and record.target_population == target_population
    )
    ports: list[ExternalInputPortSpec] = []
    for index, record in enumerate(channels):
        conductance = record.channel.get("g_bar")
        reversal = record.channel.get("equilibriumPotential")
        if conductance is None or reversal is None:
            raise ValueError(f"{record.id}: incomplete ModelDB input channel")
        ports.append(
            ExternalInputPortSpec(
                name=f"external_{index:03d}",
                record_id=record.id,
                compartment=record.target_compartment,
                conductance_density_mS_cm2=float(conductance),
                reversal_mV=float(reversal),
                sensitivities_mV=tuple(
                    float(record.gate.get(f"input{index}", 0.0)) for index in range(1, 5)
                ),
            )
        )
    return tuple(ports)


def modeldb_injection_ports_for_target(
    records: tuple[ModelDBExternalChannel, ...], target_population: str
) -> tuple[InjectionPortSpec, ...]:
    channels = tuple(
        record
        for record in records
        if record.dependency == "injection" and record.target_population == target_population
    )
    return tuple(
        InjectionPortSpec(
            name=f"injection_{index:03d}",
            record_id=record.id,
            compartment=record.target_compartment,
            sensitivities_pA_cm2=tuple(
                float(record.gate.get(f"input{channel}", 0.0)) for channel in range(1, 5)
            ),
        )
        for index, record in enumerate(channels)
    )
