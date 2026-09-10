"""Decompose somatic and proximal recurrent TRN GABA in the fixed pair."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

from smart_robustness.protocols import MatchCondition
from smart_robustness.validation.calibration import runtime_conventions_for_candidate
from smart_robustness.validation.figure6 import Figure6LearningProtocol, run_figure6_learning
from smart_robustness.validation.figure7 import TopDownCurrentMode, run_figure7_condition


def _condition_summary(result):
    counts = Counter(int(index) for index in result.trn_spike_indices)
    bins = Counter(int(np.floor(time_ms)) for time_ms in result.trn_spike_times_ms)
    diagnostic = (22, 31, 38, 39, 40, 41, 42, 49, 58)
    diagnostic_total = sum(counts[index] for index in diagnostic)
    return {
        "relay_event_count": len(result.relay_spike_times_ms),
        "relay_active_indices": sorted(set(result.relay_spike_indices)),
        "relay_spike_times_ms": list(result.relay_spike_times_ms),
        "category_event_count": len(result.category_spike_times_ms),
        "trn_event_count": len(result.trn_spike_times_ms),
        "trn_event_count_by_index": [[index, counts.get(index, 0)] for index in range(81)],
        "trn_diagnostic_index_event_count": diagnostic_total,
        "trn_remaining_index_event_count": len(result.trn_spike_times_ms) - diagnostic_total,
        "trn_nonzero_one_ms_bins": [
            [bin_index, bins[bin_index]] for bin_index in sorted(bins)
        ],
        "nonspecific_event_count": len(result.nonspecific_spike_times_ms),
        "nonspecific_spike_times_ms": list(result.nonspecific_spike_times_ms),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True)
    args = parser.parse_args()
    registration = yaml.safe_load(Path(args.registration).read_text())
    profile = yaml.safe_load(Path(registration["profile"]).read_text())
    training_profile = yaml.safe_load(Path(profile["training_profile"]).read_text())
    base_profile = yaml.safe_load(Path(training_profile["base_profile"]).read_text())
    conventions = replace(
        runtime_conventions_for_candidate(base_profile["candidate"]),
        **training_profile["runtime_overrides"],
        **profile["runtime_overrides"],
    )
    if conventions.fingerprint != registration["runtime_fingerprint"]:
        raise ValueError("registration runtime differs from executable profile")

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    scales = {
        str(key): float(value)
        for key, value in training_profile["projection_weight_scales"].items()
    }
    training = run_figure6_learning(
        conventions=conventions,
        protocol=Figure6LearningProtocol(
            monitored_populations=tuple(training_profile["monitored_populations"])
        ),
        projection_weight_scales=scales,
        brian=brian,
    )
    expected = set(profile["figure6_gates"]["relay_active_indices"])
    relay_indices = training.result.population_spike_indices["thalamic_relay"]
    relay_counts = {index: relay_indices.count(index) for index in expected}
    figure6_gates = {
        "relay_active_indices": set(relay_indices) == expected,
        "relay_events_per_active_index": set(relay_counts.values()) == {4},
        "relay_events": len(relay_indices) == 20,
        "top_down_horizontal_contrast": (
            training.result.top_down_combined.horizontal_orientation_contrast > 0
        ),
    }
    if not all(figure6_gates.values()):
        raise RuntimeError(f"fresh Figure 6 prerequisite failed: {figure6_gates}")

    protocol = profile["protocol"]
    comparator = profile["comparator"]
    common = {
        "learned_weights": training.learned_weights,
        "conventions": conventions,
        "persistent_projection_weight_scales": scales,
        "top_down_current_pA": float(protocol["top_down_current_pA"]),
        "top_down_current_mode": TopDownCurrentMode(protocol["top_down_current_mode"]),
        "top_down_cue_lead_ms": float(protocol["top_down_cue_lead_ms"]),
        "duration_ms": float(protocol["duration_ms"]),
        "dt_ms": float(protocol["dt_ms"]),
        "equilibration_ms": float(protocol["equilibration_ms"]),
        "comparator_top_k_targets": int(comparator["target_count"]),
        "comparator_source_index": int(comparator["source_index"]),
        "brian": brian,
    }
    outcomes = []
    for arm in registration["recognition_only_ablation_arms"]:
        disabled = tuple(arm["projection_ids"])
        match = run_figure7_condition(
            condition=MatchCondition.MATCH,
            disabled_projection_ids=disabled,
            **common,
        )
        mismatch = run_figure7_condition(
            condition=MatchCondition.MISMATCH,
            disabled_projection_ids=disabled,
            **common,
        )
        outcomes.append(
            {
                "arm": arm["name"],
                "disabled_projection_ids": list(disabled),
                "match": _condition_summary(match),
                "mismatch": _condition_summary(mismatch),
            }
        )

    print(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "id": registration["result_id"],
                "date": registration["date"],
                "status": "completed-recognition-only-path-decomposition",
                "classification": registration["classification"],
                "registration": args.registration,
                "runtime_fingerprint": conventions.fingerprint,
                "figure6_gates": figure6_gates,
                "trn_gap_junction_retained": True,
                "outcomes": outcomes,
                "model_candidate": False,
                "original_smart_reproduced": False,
                "baseline_promoted": False,
            },
            sort_keys=False,
        ),
        end="",
    )


if __name__ == "__main__":
    main()
