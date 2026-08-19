"""Measure Equation 25/28 terms in the leading Figure 6 source profile."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import yaml

from smart_robustness.validation.calibration import runtime_conventions_for_candidate
from smart_robustness.validation.figure6 import run_figure6_top_down_learning_phase


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        default="configs/calibration/figure6_relay_axial_source_hybrid_v1.yaml",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    profile = yaml.safe_load(Path(args.profile).read_text())
    conventions = runtime_conventions_for_candidate(profile["candidate"])
    result = run_figure6_top_down_learning_phase(conventions=conventions, brian=brian)
    maximum_error = max(
        abs(connection.measured_delta - connection.reconstructed_delta)
        for connection in result.connections
    )
    artifact = {
        "schema_version": 1,
        "id": Path(args.output).stem,
        "date": datetime.now(tz=UTC).date().isoformat(),
        "profile": args.profile,
        "candidate_fingerprint": profile["candidate_fingerprint"],
        "runtime_fingerprint": conventions.fingerprint,
        "holdouts_consulted": False,
        "status": "top-down-learning-phase-measured",
        "result": asdict(result),
        "assessment": {
            "maximum_delta_reconstruction_error": maximum_error,
            "integration_consistent": maximum_error < 5e-4,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(artifact, sort_keys=False))


if __name__ == "__main__":
    main()
