"""Reassess source-bounded Figure 6 learning-threshold candidates."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import yaml

from smart_robustness.validation.figure6 import (
    HORIZONTAL_INDICES,
    MINIMUM_TOP_DOWN_COMBINED_PEAK,
    MINIMUM_TOP_DOWN_HORIZONTAL_CONTRAST,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("artifacts", nargs="+")
    args = parser.parse_args()

    candidates = []
    for artifact_path in args.artifacts:
        artifact = yaml.safe_load(Path(artifact_path).read_text())
        horizontal_times = artifact["active_cell_times_ms"]["relay_horizontal"]
        relay_confined = bool(
            artifact["population_spikes"]["thalamic_relay"] == 20
            and all(len(horizontal_times[str(index)]) == 4 for index in HORIZONTAL_INDICES)
        )
        combined = artifact["maps"]["top_down_combined"]
        peak = float(combined["maximum_after"])
        contrast = float(combined["horizontal_orientation_contrast"])
        reproduced = bool(
            relay_confined
            and artifact["recruitment"]["feedforward_chain_complete"]
            and artifact["top_down_timing"]["causal_pair_in_learning_window"]
            and artifact["maps"]["bottom_up_oriented"]
            and contrast >= MINIMUM_TOP_DOWN_HORIZONTAL_CONTRAST
            and peak >= MINIMUM_TOP_DOWN_COMBINED_PEAK
        )
        candidates.append(
            {
                "artifact": artifact_path,
                "profile": artifact["profile"],
                "combined_peak": peak,
                "combined_horizontal_contrast": contrast,
                "relay_spikes": artifact["population_spikes"]["thalamic_relay"],
                "relay_recruitment_confined": relay_confined,
                "figure6_reproduced": reproduced,
            }
        )

    output = {
        "schema_version": 1,
        "id": Path(args.output).stem,
        "date": datetime.now(tz=UTC).date().isoformat(),
        "registered_gates": {
            "minimum_combined_peak": MINIMUM_TOP_DOWN_COMBINED_PEAK,
            "minimum_horizontal_contrast": MINIMUM_TOP_DOWN_HORIZONTAL_CONTRAST,
            "relay_horizontal_cells": list(HORIZONTAL_INDICES),
            "events_per_relay_in_100_ms": 4,
        },
        "candidates": candidates,
        "assessment": {
            "threshold_value_explanation_sufficient": False,
            "learning_coordinate_explanation_sufficient": False,
            "learning_rule_explanation_sufficient": False,
            "promoted_profile": None,
        },
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(output, sort_keys=False))


if __name__ == "__main__":
    main()
