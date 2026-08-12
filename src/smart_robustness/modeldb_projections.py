"""Typed derived projection facts from the official ModelDB 112923 SMART XML."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from .models.modeldb112923 import SMART_NML_SHA256


@dataclass(frozen=True, slots=True)
class ModelDBKernel:
    sigma_x: float | None
    sigma_y: float | None
    width: float | None
    height: float | None
    ring: bool | None
    wrap: bool | None
    border_effect: str | None


@dataclass(frozen=True, slots=True)
class ModelDBProjection:
    id: str
    kind: str
    source_population: str
    source_compartment: str | None
    target_population: str
    target_compartment: str
    channel_name: str
    dependency: str
    channel_conductance_mS_cm2: float | None
    reversal_mV: float | None
    rise_ms: float | None
    fall_ms: float | None
    delay_ms: float | None
    method: str | None
    weight: float | None
    asymptotic_weight: float | None
    kernel: ModelDBKernel | None
    projection_attributes: Mapping[str, str]
    method_attributes: Mapping[str, str]
    gate_attributes: Mapping[str, str]

    @property
    def modifiable(self) -> bool:
        return self.projection_attributes.get("modifiable") == "true"

    @property
    def learning_rule(self) -> str | None:
        return self.projection_attributes.get("learningRule")

    @property
    def learning_rate(self) -> float | None:
        value = self.projection_attributes.get("learningRate")
        return None if value is None else float(value)

    @property
    def depotentiation_ms(self) -> float | None:
        value = self.projection_attributes.get("depotentiationLength")
        return None if value is None else float(value)


@dataclass(frozen=True, slots=True)
class ModelDBExternalChannel:
    id: str
    target_population: str
    target_compartment: str
    channel_name: str
    dependency: str
    channel: Mapping[str, str]
    gate: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class ModelDBFirstOrderCatalog:
    source_sha256: str
    projections: tuple[ModelDBProjection, ...]
    external_channels: tuple[ModelDBExternalChannel, ...]
    intrinsic_populations: tuple[Mapping[str, Any], ...]

    def by_id(self, record_id: str) -> ModelDBProjection:
        return next(record for record in self.projections if record.id == record_id)


def catalog_path() -> Path:
    return Path(files("smart_robustness").joinpath("data/modeldb112923_first_order.yaml"))


def full_catalog_path() -> Path:
    return Path(files("smart_robustness").joinpath("data/modeldb112923_full.yaml"))


def _mapping(value: Mapping[str, Any] | None) -> Mapping[str, str]:
    return MappingProxyType({} if value is None else {str(k): str(v) for k, v in value.items()})


def load_modeldb_first_order_catalog(
    path: str | Path | None = None,
) -> ModelDBFirstOrderCatalog:
    source_path = Path(path) if path is not None else catalog_path()
    data = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    if data["source_sha256"] != SMART_NML_SHA256:
        raise ValueError("derived ModelDB catalog does not match the pinned SMART.nml")
    projections: list[ModelDBProjection] = []
    for item in data["projections"]:
        kernel_data = item["kernel"]
        kernel = None if kernel_data is None else ModelDBKernel(**kernel_data)
        projections.append(
            ModelDBProjection(
                **{
                    key: item[key]
                    for key in (
                        "id",
                        "kind",
                        "source_population",
                        "source_compartment",
                        "target_population",
                        "target_compartment",
                        "channel_name",
                        "dependency",
                        "channel_conductance_mS_cm2",
                        "reversal_mV",
                        "rise_ms",
                        "fall_ms",
                        "delay_ms",
                        "method",
                        "weight",
                        "asymptotic_weight",
                    )
                },
                kernel=kernel,
                projection_attributes=_mapping(item["projection_attributes"]),
                method_attributes=_mapping(item["method_attributes"]),
                gate_attributes=_mapping(item["gate_attributes"]),
            )
        )
    external = tuple(
        ModelDBExternalChannel(
            id=item["id"],
            target_population=item["target_population"],
            target_compartment=item["target_compartment"],
            channel_name=item["channel_name"],
            dependency=item["dependency"],
            channel=_mapping(item["channel"]),
            gate=_mapping(item["gate"]),
        )
        for item in data["external_channels"]
    )
    if (
        len(projections) != data["projection_count"]
        or len(external) != data["external_channel_count"]
    ):
        raise ValueError("derived ModelDB catalog count mismatch")
    ids = tuple(record.id for record in projections)
    if len(ids) != len(set(ids)):
        raise ValueError("derived ModelDB projection IDs are not unique")
    return ModelDBFirstOrderCatalog(
        source_sha256=data["source_sha256"],
        projections=tuple(projections),
        external_channels=external,
        intrinsic_populations=tuple(
            MappingProxyType(population) for population in data["intrinsic_populations"]
        ),
    )


def load_modeldb_full_catalog(path: str | Path | None = None) -> ModelDBFirstOrderCatalog:
    """Load the integrity-pinned complete V1-pulvinar-V2 SMART catalog."""

    return load_modeldb_first_order_catalog(path or full_catalog_path())


MODELDB_FIRST_ORDER = load_modeldb_first_order_catalog()
MODELDB_FULL = load_modeldb_full_catalog()
