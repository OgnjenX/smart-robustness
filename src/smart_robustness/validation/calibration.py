"""Validated contracts for constrained classic-SMART calibration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ALLOWED_STATUSES = {
    "conflicting_official_source",
    "derived_source_ambiguity",
    "source_ambiguity",
    "conflicting_protocol_source",
    "not_identifiable",
}


@dataclass(frozen=True, slots=True)
class CalibrationDimension:
    name: str
    kind: str
    status: str
    source: str
    values: tuple[str, ...] = ()
    bounds: tuple[float, float] | None = None
    grid: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class CalibrationContract:
    name: str
    base_tag: str
    training_targets: tuple[str, ...]
    holdout_targets: tuple[str, ...]
    dimensions: tuple[CalibrationDimension, ...]
    forbidden_free_parameters: tuple[str, ...]

    @property
    def fingerprint(self) -> str:
        payload = {
            "name": self.name,
            "base_tag": self.base_tag,
            "training_targets": self.training_targets,
            "holdout_targets": self.holdout_targets,
            "dimensions": [
                {
                    "name": item.name,
                    "kind": item.kind,
                    "status": item.status,
                    "source": item.source,
                    "values": item.values,
                    "bounds": item.bounds,
                    "grid": item.grid,
                }
                for item in self.dimensions
            ],
            "forbidden_free_parameters": self.forbidden_free_parameters,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    def candidate_fingerprint(self, values: dict[str, Any]) -> str:
        expected = {dimension.name for dimension in self.dimensions}
        if set(values) != expected:
            missing = sorted(expected - set(values))
            extra = sorted(set(values) - expected)
            raise ValueError(f"candidate dimensions differ: missing={missing}, extra={extra}")
        for dimension in self.dimensions:
            value = values[dimension.name]
            if dimension.kind == "categorical" and value not in dimension.values:
                raise ValueError(f"{dimension.name}: value {value!r} is not admissible")
            if dimension.kind == "numeric":
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise ValueError(f"{dimension.name}: numeric value required")
                assert dimension.bounds is not None
                if not dimension.bounds[0] <= float(value) <= dimension.bounds[1]:
                    raise ValueError(f"{dimension.name}: value outside declared bounds")
        encoded = json.dumps(values, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(f"{self.fingerprint}:{encoded}".encode()).hexdigest()


def load_calibration_contract(path: str | Path) -> CalibrationContract:
    raw = yaml.safe_load(Path(path).read_text())
    if raw.get("schema_version") != 1:
        raise ValueError("calibration contract requires schema_version 1")
    training = tuple(raw["training_targets"])
    holdout = tuple(raw["holdout_targets"])
    overlap = sorted(set(training) & set(holdout))
    if overlap:
        raise ValueError(f"training and holdout targets overlap: {overlap}")
    dimensions: list[CalibrationDimension] = []
    for name, spec in raw["dimensions"].items():
        kind = spec["kind"]
        status = spec["status"]
        if status not in ALLOWED_STATUSES:
            raise ValueError(f"{name}: status {status!r} is not calibration-admissible")
        if kind == "categorical":
            values = tuple(spec.get("values", ()))
            if len(values) < 2:
                raise ValueError(f"{name}: categorical dimension needs at least two values")
            bounds = None
            grid = ()
        elif kind == "numeric":
            values = ()
            bounds_raw = tuple(float(value) for value in spec.get("bounds", ()))
            if len(bounds_raw) != 2 or bounds_raw[0] >= bounds_raw[1]:
                raise ValueError(f"{name}: numeric bounds must be increasing")
            bounds = (bounds_raw[0], bounds_raw[1])
            grid = tuple(float(value) for value in spec.get("grid", ()))
            if not grid or any(not bounds[0] <= value <= bounds[1] for value in grid):
                raise ValueError(f"{name}: grid must be nonempty and within bounds")
        else:
            raise ValueError(f"{name}: unknown dimension kind {kind!r}")
        dimensions.append(
            CalibrationDimension(
                name=name,
                kind=kind,
                status=status,
                source=str(spec["source"]),
                values=values,
                bounds=bounds,
                grid=grid,
            )
        )
    forbidden = tuple(raw.get("forbidden_free_parameters", ()))
    if not forbidden:
        raise ValueError("calibration contract must declare forbidden free parameters")
    return CalibrationContract(
        name=str(raw["name"]),
        base_tag=str(raw["base_tag"]),
        training_targets=training,
        holdout_targets=holdout,
        dimensions=tuple(dimensions),
        forbidden_free_parameters=forbidden,
    )
