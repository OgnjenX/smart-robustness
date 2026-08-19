"""Evaluate the archived-relay/Table-3-others source hybrid on Figure 6."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml

from smart_robustness.validation.calibration import runtime_conventions_for_candidate
from smart_robustness.validation.figure6 import (
    HORIZONTAL_ONLY_INDICES,
    VERTICAL_ONLY_INDICES,
    Figure6LearningProtocol,
    assess_figure6_cortical_recruitment,
    assess_figure6_top_down_timing,
    run_figure6_learning,
)

MONITORED = (
    "thalamic_relay",
    "layer4_excitatory_v1",
    "layer23_inhibitory_v1",
    "layer23_excitatory_v1",
    "layer5_excitatory_v1",
    "layer6i_excitatory_v1",
    "layer6ii_excitatory_v1",
    "trn",
    "thalamic_nonspecific",
)


def _map_details(summary) -> dict[str, object]:
    delta = np.asarray(summary.delta, dtype=float)
    after = np.asarray(summary.after, dtype=float)
    return {
        "minimum_delta": float(np.min(delta)),
        "maximum_delta": float(np.max(delta)),
        "maximum_after": float(np.max(after)),
        "horizontal_orientation_contrast": summary.horizontal_orientation_contrast,
        "horizontal_retention_advantage": summary.horizontal_retention_advantage,
        "horizontal_arm_after": [float(after[index]) for index in HORIZONTAL_ONLY_INDICES],
        "vertical_arm_after": [float(after[index]) for index in VERTICAL_ONLY_INDICES],
        "horizontal_arm_delta": [float(delta[index]) for index in HORIZONTAL_ONLY_INDICES],
        "vertical_arm_delta": [float(delta[index]) for index in VERTICAL_ONLY_INDICES],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile", default="configs/calibration/figure6_relay_source_hybrid_v1.yaml"
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    profile = yaml.safe_load(Path(args.profile).read_text())
    conventions = runtime_conventions_for_candidate(profile["candidate"])
    protocol = Figure6LearningProtocol(monitored_populations=MONITORED)
    run = run_figure6_learning(conventions=conventions, protocol=protocol, brian=brian)
    recruitment = assess_figure6_cortical_recruitment(run.result)
    timing = assess_figure6_top_down_timing(run.result)
    spike_indices = run.result.population_spike_indices or {}
    spike_times = run.result.population_spike_times_ms or {}

    def indexed_times(population: str, index: int) -> list[float]:
        return [
            time
            for source_index, time in zip(
                spike_indices.get(population, ()),
                spike_times.get(population, ()),
                strict=True,
            )
            if source_index == index
        ]
    relay_indices = tuple(spike_indices.get("thalamic_relay", ()))
    relay_recruitment_confined = bool(
        len(relay_indices) == 20
        and set(relay_indices) == {38, 39, 40, 41, 42}
        and all(relay_indices.count(index) == 4 for index in (38, 39, 40, 41, 42))
    )
    figure6_reproduced = bool(
        relay_recruitment_confined
        and recruitment.feedforward_chain_complete
        and timing.causal_pair_in_learning_window
        and run.result.bottom_up_oriented
        and run.result.top_down_oriented
    )
    artifact = {
        "schema_version": 1,
        "id": Path(args.output).stem,
        "date": datetime.now(tz=UTC).date().isoformat(),
        "profile": args.profile,
        "candidate_fingerprint": profile["candidate_fingerprint"],
        "runtime_fingerprint": conventions.fingerprint,
        "holdouts_consulted": False,
        "status": (
            "complete-figure6-pass"
            if figure6_reproduced
            else "partial-figure6b-pass-figure6c-fail"
        ),
        "protocol": asdict(protocol),
        "population_spikes": run.result.population_spikes,
        "relay_recruitment": {
            "active_indices": sorted(set(relay_indices)),
            "confined_to_horizontal_bar_at_40_hz": relay_recruitment_confined,
        },
        "active_cell_times_ms": {
            "relay_horizontal": {
                str(index): indexed_times("thalamic_relay", index)
                for index in (38, 39, 40, 41, 42)
            },
            "category_40": indexed_times("layer6ii_excitatory_v1", 40),
        },
        "recruitment": {
            **asdict(recruitment),
            "feedforward_chain_complete": recruitment.feedforward_chain_complete,
        },
        "top_down_timing": {
            **asdict(timing),
            "causal_pair_in_learning_window": timing.causal_pair_in_learning_window,
        },
        "maps": {
            "bottom_up_horizontal_orientation_contrast": (
                run.result.bottom_up.horizontal_orientation_contrast
            ),
            "top_down_horizontal_orientation_contrast": (
                run.result.top_down_combined.horizontal_orientation_contrast
            ),
            "bottom_up_oriented": run.result.bottom_up_oriented,
            "top_down_oriented": run.result.top_down_oriented,
            "bottom_up": _map_details(run.result.bottom_up),
            "top_down_wide": _map_details(run.result.top_down_wide),
            "top_down_narrow": _map_details(run.result.top_down_narrow),
            "top_down_combined": _map_details(run.result.top_down_combined),
        },
        "assessment": {
            "figure6_reproduced": figure6_reproduced,
            "promoted": figure6_reproduced,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(artifact, sort_keys=False))


if __name__ == "__main__":
    main()
