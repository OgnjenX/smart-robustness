"""Record the earliest relay-to-cortex bottleneck in the Figure 6 episode."""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import yaml

from smart_robustness.validation.calibration import runtime_conventions_for_candidate
from smart_robustness.validation.figure6 import run_figure6_relay_current_balance

TRN_TO_RELAY_PROJECTION_IDS = (
    "modeldb112923.projection.000",
    "modeldb112923.projection.001",
    "modeldb112923.projection.004",
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
    conventions = runtime_conventions_for_candidate(profile["candidate"])
    connected = run_figure6_relay_current_balance(conventions=conventions, brian=brian)
    intrinsic = run_figure6_relay_current_balance(
        conventions=conventions, connected=False, brian=brian
    )
    without_trn_to_relay = run_figure6_relay_current_balance(
        conventions=conventions,
        disabled_projection_ids=TRN_TO_RELAY_PROJECTION_IDS,
        brian=brian,
    )
    assessment = {
        "source_drive_resolved": math.isclose(
            connected.external_effective_reversal_mV, -12.0, abs_tol=1e-9
        ),
        "connected_relay_repeats": connected.relay_repeats_during_stimulus,
        "intrinsic_relay_repeats": intrinsic.relay_repeats_during_stimulus,
        "relay_recruits_target_layer4": connected.relay_recruits_target_layer4,
        "network_inhibition_explains_repetition_failure": (
            intrinsic.relay_repeats_during_stimulus
            and not connected.relay_repeats_during_stimulus
        ),
        "trn_inhibition_explains_repetition_failure": (
            without_trn_to_relay.relay_repeats_during_stimulus
            and not connected.relay_repeats_during_stimulus
        ),
        "earliest_failed_gate": (
            "relay_repetition"
            if connected.relay_recruits_target_layer4
            and not connected.relay_repeats_during_stimulus
            else None
        ),
    }
    artifact = {
        "schema_version": 1,
        "id": Path(args.output).stem,
        "date": datetime.now(tz=UTC).date().isoformat(),
        "profile": args.profile,
        "candidate_fingerprint": profile["candidate_fingerprint"],
        "holdouts_consulted": False,
        "status": "localized-relay-repetition-failure",
        "connected_result": asdict(connected),
        "intrinsic_only_result": asdict(intrinsic),
        "without_trn_to_relay_result": asdict(without_trn_to_relay),
        "assessment": assessment,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(artifact, sort_keys=False))
    print(yaml.safe_dump(artifact, sort_keys=False), end="")


if __name__ == "__main__":
    main()
