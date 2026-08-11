"""Typed catalog of SMART Supplementary Table 3 projections.

The main article calls this source ``Supplementary Table 4``.  The recovered
supplement labels it ``Supplementary Table 3``; that label is used here to
avoid silently rewriting the primary source.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml


class CatalogValidationError(ValueError):
    """Raised when the machine-readable projection catalog is inconsistent."""


class ConnectionKind(StrEnum):
    CHEMICAL = "chemical"
    EXTERNAL_INPUT = "external_input"
    GAP_JUNCTION = "gap_junction"


class Receptor(StrEnum):
    AMPA = "ampa"
    NMDA = "nmda"
    GABA = "gaba"
    INPUT = "input"
    GAP_JUNCTION = "gap_junction"
    UNKNOWN = "unknown"


class TopologyKind(StrEnum):
    GAUSSIAN = "gaussian"
    ONE_TO_ONE = "one_to_one"
    ALL_TO_ONE = "all_to_one"


class VerificationStatus(StrEnum):
    CROSS_CHECKED = "cross_checked"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class Endpoint:
    population: str
    compartment: str | None


@dataclass(frozen=True)
class RawProjectionValues:
    receptor: str
    reversal_mV: str | None
    conductance_pS: str | None
    weight_density_1e6_cm2: str | None
    topology: str | None
    delay_ms: str | None
    rise_fall_ms: str | None


@dataclass(frozen=True)
class Topology:
    kind: TopologyKind
    sigma: float | None


@dataclass(frozen=True)
class Plasticity:
    max_weight_density_1e6_cm2: float
    baseline_weight_density_1e6_cm2: float
    learning_rate: float


@dataclass(frozen=True)
class Depletion:
    epsilon: float
    tau_ms: float


@dataclass(frozen=True)
class ParsedProjectionValues:
    receptor: Receptor
    reversal_mV: float | None
    conductance_pS: float | None
    weight_density_1e6_cm2: float | None
    topology: Topology
    delay_ms: float | None
    rise_ms: float | None
    fall_ms: float | None
    plasticity: Plasticity | None
    depletion: Depletion | None


@dataclass(frozen=True)
class Verification:
    status: VerificationStatus
    source_forms: tuple[str, ...]
    notes: tuple[str, ...]


@dataclass(frozen=True)
class ProjectionRecord:
    id: str
    kind: ConnectionKind
    source: Endpoint
    target: Endpoint
    raw: RawProjectionValues
    parsed: ParsedProjectionValues
    verification: Verification


@dataclass(frozen=True)
class ProjectionCatalog:
    schema_version: int
    source_title: str
    source_alias: str
    expected_record_count: int
    records: tuple[ProjectionRecord, ...]

    def by_id(self, record_id: str) -> ProjectionRecord:
        for record in self.records:
            if record.id == record_id:
                return record
        raise KeyError(record_id)


def _required(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise CatalogValidationError(f"{context}: missing required field {key!r}")
    return mapping[key]


def _endpoint(data: Mapping[str, Any], context: str) -> Endpoint:
    return Endpoint(
        population=str(_required(data, "population", context)),
        compartment=data.get("compartment"),
    )


def _optional_float(value: Any, context: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CatalogValidationError(f"{context}: expected a number or null, got {value!r}")
    return float(value)


def _parse_record(data: Mapping[str, Any]) -> ProjectionRecord:
    record_id = str(_required(data, "id", "record"))
    context = f"record {record_id!r}"
    raw_data = _required(data, "raw", context)
    parsed_data = _required(data, "parsed", context)
    topology_data = _required(parsed_data, "topology", context)
    plasticity_data = parsed_data.get("plasticity")
    depletion_data = parsed_data.get("depletion")
    verification_data = _required(data, "verification", context)

    raw = RawProjectionValues(
        receptor=str(_required(raw_data, "receptor", context)),
        reversal_mV=raw_data.get("reversal_mV"),
        conductance_pS=raw_data.get("conductance_pS"),
        weight_density_1e6_cm2=raw_data.get("weight_density_1e6_cm2"),
        topology=raw_data.get("topology"),
        delay_ms=raw_data.get("delay_ms"),
        rise_fall_ms=raw_data.get("rise_fall_ms"),
    )
    parsed = ParsedProjectionValues(
        receptor=Receptor(_required(parsed_data, "receptor", context)),
        reversal_mV=_optional_float(parsed_data.get("reversal_mV"), context),
        conductance_pS=_optional_float(parsed_data.get("conductance_pS"), context),
        weight_density_1e6_cm2=_optional_float(
            parsed_data.get("weight_density_1e6_cm2"), context
        ),
        topology=Topology(
            kind=TopologyKind(_required(topology_data, "kind", context)),
            sigma=_optional_float(topology_data.get("sigma"), context),
        ),
        delay_ms=_optional_float(parsed_data.get("delay_ms"), context),
        rise_ms=_optional_float(parsed_data.get("rise_ms"), context),
        fall_ms=_optional_float(parsed_data.get("fall_ms"), context),
        plasticity=None
        if plasticity_data is None
        else Plasticity(
            max_weight_density_1e6_cm2=float(
                _required(plasticity_data, "max_weight_density_1e6_cm2", context)
            ),
            baseline_weight_density_1e6_cm2=float(
                _required(plasticity_data, "baseline_weight_density_1e6_cm2", context)
            ),
            learning_rate=float(_required(plasticity_data, "learning_rate", context)),
        ),
        depletion=None
        if depletion_data is None
        else Depletion(
            epsilon=float(_required(depletion_data, "epsilon", context)),
            tau_ms=float(_required(depletion_data, "tau_ms", context)),
        ),
    )
    verification = Verification(
        status=VerificationStatus(_required(verification_data, "status", context)),
        source_forms=tuple(_required(verification_data, "source_forms", context)),
        notes=tuple(verification_data.get("notes", ())),
    )
    record = ProjectionRecord(
        id=record_id,
        kind=ConnectionKind(_required(data, "kind", context)),
        source=_endpoint(_required(data, "source", context), context),
        target=_endpoint(_required(data, "target", context), context),
        raw=raw,
        parsed=parsed,
        verification=verification,
    )
    _validate_record(record)
    return record


def _validate_record(record: ProjectionRecord) -> None:
    context = f"record {record.id!r}"
    if not record.id or record.id.lower() != record.id or " " in record.id:
        raise CatalogValidationError(f"{context}: ID must be nonempty lowercase without spaces")
    if not record.source.population or not record.target.population:
        raise CatalogValidationError(f"{context}: source and target populations are required")
    if record.target.compartment not in {"soma", "proximal_dendrite", "distal_dendrite"}:
        raise CatalogValidationError(f"{context}: invalid target compartment")

    p = record.parsed
    for label, value in (
        ("conductance", p.conductance_pS),
        ("weight density", p.weight_density_1e6_cm2),
        ("delay", p.delay_ms),
        ("rise", p.rise_ms),
        ("fall", p.fall_ms),
    ):
        if value is not None and value < 0:
            raise CatalogValidationError(f"{context}: {label} cannot be negative")
    if p.topology.kind is TopologyKind.GAUSSIAN:
        if p.topology.sigma is None or p.topology.sigma <= 0:
            raise CatalogValidationError(f"{context}: Gaussian topology needs positive sigma")
    elif p.topology.sigma is not None:
        raise CatalogValidationError(f"{context}: non-Gaussian topology cannot have sigma")

    expected_receptor = {
        ConnectionKind.EXTERNAL_INPUT: Receptor.INPUT,
        ConnectionKind.GAP_JUNCTION: Receptor.GAP_JUNCTION,
    }.get(record.kind)
    if expected_receptor is not None and p.receptor is not expected_receptor:
        raise CatalogValidationError(f"{context}: kind and receptor disagree")
    if record.kind is ConnectionKind.CHEMICAL and p.receptor in {
        Receptor.INPUT,
        Receptor.GAP_JUNCTION,
    }:
        raise CatalogValidationError(f"{context}: chemical record has nonchemical receptor")
    if p.receptor is Receptor.UNKNOWN and record.verification.status is not VerificationStatus.AMBIGUOUS:
        raise CatalogValidationError(f"{context}: unknown receptor must be marked ambiguous")

    if p.plasticity is not None:
        if min(
            p.plasticity.max_weight_density_1e6_cm2,
            p.plasticity.baseline_weight_density_1e6_cm2,
            p.plasticity.learning_rate,
        ) < 0:
            raise CatalogValidationError(f"{context}: plasticity values cannot be negative")
        if (
            p.plasticity.baseline_weight_density_1e6_cm2
            > p.plasticity.max_weight_density_1e6_cm2
        ):
            raise CatalogValidationError(f"{context}: baseline exceeds plastic maximum")
    if p.depletion is not None and (p.depletion.epsilon < 0 or p.depletion.tau_ms <= 0):
        raise CatalogValidationError(f"{context}: invalid depletion parameters")
    if not record.verification.source_forms:
        raise CatalogValidationError(f"{context}: verification must identify source forms")
    if record.verification.status is VerificationStatus.AMBIGUOUS and not record.verification.notes:
        raise CatalogValidationError(f"{context}: ambiguous record needs a note")


def catalog_path() -> Path:
    """Return the installed catalog path."""
    return Path(files("smart_robustness").joinpath("data/supplementary_table3.yaml"))


def load_projection_catalog(path: str | Path | None = None) -> ProjectionCatalog:
    """Load and validate the source-backed Supplementary Table 3 catalog."""
    source_path = Path(path) if path is not None else catalog_path()
    data = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise CatalogValidationError("catalog root must be a mapping")
    records_data = _required(data, "records", "catalog")
    if not isinstance(records_data, list):
        raise CatalogValidationError("catalog records must be a list")
    records = tuple(_parse_record(item) for item in records_data)
    expected = int(_required(data, "expected_record_count", "catalog"))
    if len(records) != expected:
        raise CatalogValidationError(f"catalog expected {expected} records, found {len(records)}")
    ids = [record.id for record in records]
    if len(ids) != len(set(ids)):
        raise CatalogValidationError("catalog record IDs must be unique")
    if ids != sorted(ids):
        raise CatalogValidationError("catalog records must be sorted by stable ID")
    return ProjectionCatalog(
        schema_version=int(_required(data, "schema_version", "catalog")),
        source_title=str(_required(data, "source_title", "catalog")),
        source_alias=str(_required(data, "source_alias", "catalog")),
        expected_record_count=expected,
        records=records,
    )


def serialize_projection_catalog(catalog: ProjectionCatalog) -> str:
    """Serialize a catalog deterministically for audit snapshots and hashing."""
    def primitive(value: Any) -> Any:
        if isinstance(value, StrEnum):
            return value.value
        if isinstance(value, dict):
            return {key: primitive(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [primitive(item) for item in value]
        return value

    return yaml.safe_dump(primitive(asdict(catalog)), sort_keys=True, allow_unicode=True)


SUPPLEMENTARY_TABLE3 = load_projection_catalog()
