"""Screen a preregistered effective T-conductance grid on one fixed match trace."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from smart_robustness.validation.calibration import runtime_conventions_for_candidate
from smart_robustness.validation.nonspecific_replay import run_nonspecific_replay


def _plain(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return _plain(asdict(value))
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Enum):
        return value.value
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True)
    args = parser.parse_args()
    registration = yaml.safe_load(Path(args.registration).read_text())
    trace_path = Path(registration["source_trace"]["path"])
    trace_sha = sha256(trace_path.read_bytes()).hexdigest()
    if trace_sha != registration["source_trace"]["sha256"]:
        raise ValueError("source trace differs from the preregistered hash")

    profile = yaml.safe_load(
        Path(
            "configs/calibration/"
            "figure7_legacy_detector_paper_nonspecific_paper_kinetics_match_v1.yaml"
        ).read_text()
    )
    training_profile = yaml.safe_load(Path(profile["training_profile"]).read_text())
    base_profile = yaml.safe_load(Path(training_profile["base_profile"]).read_text())
    conventions = replace(
        runtime_conventions_for_candidate(base_profile["candidate"]),
        **training_profile["runtime_overrides"],
    )
    if conventions.fingerprint != registration["source_trace"]["runtime_fingerprint"]:
        raise ValueError("runtime conventions differ from the trace registration")

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    outcomes = []
    target_count = int(registration["target"]["nonspecific_event_count"])
    for scale in registration["common_proximal_distal_scale_grid"]:
        replay = run_nonspecific_replay(
            trace_path,
            conventions=conventions,
            calcium_conductance_scale=float(scale),
            brian=brian,
        )
        outcomes.append(
            {
                "scale": float(scale),
                "effective_density_mS_cm2": (
                    float(scale) * float(registration["paper_source_density_mS_cm2"])
                ),
                "event_count": len(replay.replay_spike_times_ms),
                "event_times_ms": replay.replay_spike_times_ms,
                "finite": replay.finite,
                "survives": (
                    replay.finite
                    and len(replay.replay_spike_times_ms) == target_count
                ),
            }
        )
    survivors = [item for item in outcomes if item["survives"]]
    selected = max(survivors, key=lambda item: item["scale"]) if survivors else None
    artifact = {
        "schema_version": 1,
        "id": registration["result_id"],
        "date": registration["date"],
        "status": "match-only-survivor-selected" if selected else "no-survivor",
        "classification": registration["classification"],
        "registration": args.registration,
        "source_trace_sha256": trace_sha,
        "runtime_fingerprint": conventions.fingerprint,
        "outcomes": outcomes,
        "survivor_scales": [item["scale"] for item in survivors],
        "selected": selected,
        "connected_verification_required": selected is not None,
        "promotable": False,
        "original_smart_reproduced": False,
        "baseline_promoted": False,
        "locked_holdouts": registration["locked_holdouts"],
        "classification_boundary": registration["classification_boundary"],
    }
    print(yaml.safe_dump(_plain(artifact), sort_keys=False), end="")


if __name__ == "__main__":
    main()
