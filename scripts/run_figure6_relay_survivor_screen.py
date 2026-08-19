"""Screen all pre-registered TRN survivors at the earliest Figure 6 gate."""

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
        "--source",
        default="docs/validation-results/calibration-stage-a-trn-complete-drive-120.yaml",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--duration-ms", type=float, default=20.0)
    args = parser.parse_args()

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    source = yaml.safe_load(Path(args.source).read_text())
    promoted = [outcome for outcome in source["outcomes"] if outcome["promoted"]]
    protocol = Figure6LearningProtocol(stimulus_ms=args.duration_ms)
    outcomes = []
    for outcome in promoted:
        candidate = outcome["candidate"]
        result = run_figure6_relay_current_balance(
            conventions=runtime_conventions_for_candidate(candidate),
            protocol=protocol,
            brian=brian,
        )
        outcomes.append(
            {
                "ordinal": outcome["ordinal"],
                "candidate": candidate,
                "candidate_fingerprint": outcome["result"]["candidate_fingerprint"],
                "result": asdict(result),
                "relay_repetition_pass": result.relay_repeats_during_stimulus,
            }
        )
        print(
            f"ordinal={outcome['ordinal']} relay_events={len(result.relay_event_times_ms)}",
            flush=True,
        )
    survivors = [item for item in outcomes if item["relay_repetition_pass"]]
    artifact = {
        "schema_version": 1,
        "id": Path(args.output).stem,
        "date": datetime.now(tz=UTC).date().isoformat(),
        "source": args.source,
        "holdouts_consulted": False,
        "protocol": asdict(protocol),
        "candidate_count": len(outcomes),
        "relay_repetition_survivor_count": len(survivors),
        "status": "relay-repetition-screen-complete",
        "outcomes": outcomes,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(artifact, sort_keys=False))


if __name__ == "__main__":
    main()
