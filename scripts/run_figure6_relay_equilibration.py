"""Test the registered TRN pre-drive as a Figure 6 equilibration control."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import yaml

from smart_robustness.validation.calibration import runtime_conventions_for_candidate
from smart_robustness.validation.figure6 import (
    Figure6LearningProtocol,
    run_figure6_relay_current_balance,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile", default="configs/calibration/trn_stage_a_survivor_v1.yaml"
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    profile = yaml.safe_load(Path(args.profile).read_text())
    protocol = Figure6LearningProtocol(warmup_ms=5.0, stimulus_ms=20.0)
    result = run_figure6_relay_current_balance(
        conventions=runtime_conventions_for_candidate(profile["candidate"]),
        protocol=protocol,
        brian=brian,
    )
    artifact = {
        "schema_version": 1,
        "id": Path(args.output).stem,
        "date": datetime.now(tz=UTC).date().isoformat(),
        "profile": args.profile,
        "candidate_fingerprint": profile["candidate_fingerprint"],
        "holdouts_consulted": False,
        "status": "equilibration-rejected-as-relay-repetition-rescue",
        "protocol": asdict(protocol),
        "result": asdict(result),
        "assessment": {
            "relay_repetition_rescued": result.relay_repeats_during_stimulus,
            "equilibration_rejected": not result.relay_repeats_during_stimulus,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(artifact, sort_keys=False))


if __name__ == "__main__":
    main()
