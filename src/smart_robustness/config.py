from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    raw: dict[str, Any]

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    @property
    def duration_ms(self) -> float:
        return float(self.raw["duration_ms"])

    @property
    def dt_ms(self) -> float:
        return float(self.raw["dt_ms"])

    @property
    def condition(self) -> str:
        return str(self.raw["condition"])

    @property
    def fingerprint(self) -> str:
        canonical = json.dumps(self.raw, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def load_config(path: str | Path) -> ExperimentConfig:
    data = yaml.safe_load(Path(path).read_text())
    required = {"seed", "duration_ms", "dt_ms", "condition", "model", "network", "analysis"}
    missing = required - data.keys()
    if missing:
        raise ValueError(f"Missing configuration keys: {sorted(missing)}")
    if data["condition"] not in {"match", "mismatch"}:
        raise ValueError("condition must be 'match' or 'mismatch'")
    if float(data["duration_ms"]) <= 0 or float(data["dt_ms"]) <= 0:
        raise ValueError("duration_ms and dt_ms must be positive")
    return ExperimentConfig(data)

