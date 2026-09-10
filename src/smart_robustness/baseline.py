"""Immutable manifest loader for the calibrated classic-SMART baseline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from smart_robustness.validation.calibration import runtime_conventions_for_candidate


@dataclass(frozen=True, slots=True)
class FrozenClassicBaseline:
    """One integrity-checked classic-SMART control endpoint."""

    path: Path
    repository_root: Path
    name: str
    scientific_label: str
    implementation_commit: str
    runtime_fingerprint: str
    projection_weight_scales: tuple[tuple[str, float], ...]
    exact_numerical_reproduction_claimed: bool
    _raw: dict[str, Any]

    @property
    def manifest_fingerprint(self) -> str:
        """Return a stable fingerprint of the complete frozen manifest."""

        payload = json.dumps(self._raw, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    @property
    def projection036_variance_topology(self) -> bool:
        """Whether the frozen control uses the calibrated variance reading."""

        return self._raw["topology"]["projection036_spread_convention"] == "variance"

    def runtime_conventions(self):
        """Construct and verify the exact runtime convention tuple."""

        profile_path = self.repository_root / self._raw["runtime"]["base_candidate_profile"]
        profile = yaml.safe_load(profile_path.read_text())
        conventions = replace(
            runtime_conventions_for_candidate(profile["candidate"]),
            **self._raw["runtime"]["overrides"],
        )
        if conventions.fingerprint != self.runtime_fingerprint:
            raise ValueError("frozen baseline runtime fingerprint does not reconstruct")
        return conventions


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen_classic_baseline(
    path: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> FrozenClassicBaseline:
    """Load a frozen baseline and verify every pinned local evidence file."""

    manifest_path = Path(path).resolve()
    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else manifest_path.parents[2]
    )
    raw = yaml.safe_load(manifest_path.read_text())
    if raw.get("schema_version") != 1:
        raise ValueError("frozen baseline manifest requires schema_version 1")
    if raw.get("status") != "frozen-calibrated-behavioral-baseline":
        raise ValueError("manifest is not an approved calibrated behavioral freeze")
    if raw.get("classification") != "calibrated-classic-smart-reconstruction":
        raise ValueError("manifest classification is not calibrated classic SMART")
    if raw.get("baseline_frozen") is not True:
        raise ValueError("manifest does not declare a frozen baseline")
    if raw.get("exact_numerical_reproduction_claimed") is not False:
        raise ValueError("calibrated freeze must not claim exact numerical recovery")
    prohibited = set(raw.get("prohibited_labels", ()))
    if "exact Grossberg-Versace-2008 numerical reproduction" not in prohibited:
        raise ValueError("manifest must preserve the exact-reproduction boundary")

    pinned = [raw["implementation"]["profile"], *raw["evidence"]]
    for item in pinned:
        candidate = root / item["path"]
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        if _sha256(candidate) != item["sha256"]:
            raise ValueError(f"frozen evidence differs: {item['path']}")

    scales = tuple(
        sorted(
            (str(projection_id), float(scale))
            for projection_id, scale in raw["projection_weight_scales"].items()
        )
    )
    if not scales or any(scale <= 0 for _, scale in scales):
        raise ValueError("projection scales must be positive")

    baseline = FrozenClassicBaseline(
        path=manifest_path,
        repository_root=root,
        name=str(raw["name"]),
        scientific_label=str(raw["scientific_label"]),
        implementation_commit=str(raw["implementation"]["frozen_code_commit"]),
        runtime_fingerprint=str(raw["runtime"]["fingerprint"]),
        projection_weight_scales=scales,
        exact_numerical_reproduction_claimed=False,
        _raw=raw,
    )
    baseline.runtime_conventions()
    return baseline

