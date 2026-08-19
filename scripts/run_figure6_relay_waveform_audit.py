"""Measure Figure 6 relay action-potential widths across official Na/K families."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

import yaml

from smart_robustness.validation.calibration import runtime_conventions_for_candidate
from smart_robustness.validation.figure6 import (
    Figure6LearningProtocol,
    run_figure6_relay_current_balance,
)

NAK_FAMILIES = (
    "standard_traub_miles",
    "archived_activation_printed_inactivation",
    "printed_activation_archived_inactivation",
    "printed_smart",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        default="configs/calibration/figure6_population_resolved_axial_v1.yaml",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    profile = yaml.safe_load(Path(args.profile).read_text())
    base = runtime_conventions_for_candidate(profile["candidate"])
    protocol = Figure6LearningProtocol(stimulus_ms=20.0)
    results = []
    for family in NAK_FAMILIES:
        conventions = replace(base, nak_rate_convention=family)
        result = run_figure6_relay_current_balance(
            conventions=conventions,
            protocol=protocol,
            target_index=40,
            connected=True,
            brian=brian,
        )
        results.append(
            {
                "nak_rate_convention": family,
                "runtime_fingerprint": conventions.fingerprint,
                "result": asdict(result),
            }
        )

    artifact = {
        "schema_version": 1,
        "id": Path(args.output).stem,
        "date": datetime.now(tz=UTC).date().isoformat(),
        "profile": args.profile,
        "protocol": asdict(protocol),
        "results": results,
        "assessment": {
            "waveform_survivor": None,
            "nak_family_explanation_sufficient": False,
            "selection_rule": (
                "A survivor must emit the Figure 6 relay event and materially extend "
                "the positive Equation 6 phase without introducing autonomous or "
                "off-protocol activity."
            ),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(artifact, sort_keys=False))


if __name__ == "__main__":
    main()
