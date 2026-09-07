"""Run mismatch for the two preregistered projection-008 match survivors."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from pathlib import Path

import yaml

from smart_robustness.protocols import MatchCondition
from smart_robustness.validation.calibration import runtime_conventions_for_candidate
from smart_robustness.validation.figure6 import Figure6LearningProtocol, run_figure6_learning
from smart_robustness.validation.figure7 import TopDownCurrentMode, run_figure7_condition


def _condition_summary(result):
    counts = Counter(int(index) for index in result.trn_spike_indices)
    diagnostic = (22, 31, 38, 39, 40, 41, 42, 49, 58)
    diagnostic_total = sum(counts[index] for index in diagnostic)
    return {
        "relay_event_count": len(result.relay_spike_times_ms),
        "relay_active_indices": sorted(set(result.relay_spike_indices)),
        "relay_spike_times_ms": list(result.relay_spike_times_ms),
        "category_event_count": len(result.category_spike_times_ms),
        "trn_event_count": len(result.trn_spike_times_ms),
        "trn_diagnostic_index_event_count": diagnostic_total,
        "trn_remaining_index_event_count": len(result.trn_spike_times_ms) - diagnostic_total,
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
    persistent_scales = {
        str(key): float(value)
        for key, value in training_profile["projection_weight_scales"].items()
    }
    projection_id = registration["calibrated_projection_id"]
    if projection_id in persistent_scales:
        raise ValueError("screened projection overlaps a persistent training scale")
    training = run_figure6_learning(
        conventions=conventions,
        protocol=Figure6LearningProtocol(
            monitored_populations=tuple(training_profile["monitored_populations"])
        ),
        projection_weight_scales=persistent_scales,
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
    common = dict(
        condition=MatchCondition.MISMATCH,
        learned_weights=training.learned_weights,
        conventions=conventions,
        persistent_projection_weight_scales=persistent_scales,
        top_down_current_pA=float(protocol["top_down_current_pA"]),
        top_down_current_mode=TopDownCurrentMode(protocol["top_down_current_mode"]),
        top_down_cue_lead_ms=float(protocol["top_down_cue_lead_ms"]),
        duration_ms=float(protocol["duration_ms"]),
        dt_ms=float(protocol["dt_ms"]),
        equilibration_ms=float(protocol["equilibration_ms"]),
        comparator_top_k_targets=int(comparator["target_count"]),
        comparator_source_index=int(comparator["source_index"]),
        brian=brian,
    )
    outcomes = []
    fixed_match_trn = {
        float(key): int(value)
        for key, value in registration["fixed_match_trn_events_by_scale"].items()
    }
    for scale in registration["survivor_scales"]:
        result = run_figure7_condition(
            projection_weight_scales={projection_id: float(scale)},
            **common,
        )
        summary = _condition_summary(result)
        summary["mismatch_gates"] = {
            "relay_active_indices": summary["relay_active_indices"]
            == registration["fixed_mismatch_gates"]["relay_active_indices"],
            "match_more_active_relay_cells": 5 > len(summary["relay_active_indices"]),
            "match_more_trn_events": fixed_match_trn[float(scale)]
            > summary["trn_event_count"],
            "nonspecific_events": summary["nonspecific_event_count"]
            == registration["fixed_mismatch_gates"]["nonspecific_events"],
        }
        outcomes.append({"effective_scale": float(scale), "mismatch": summary})

    print(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "id": registration["result_id"],
                "date": registration["date"],
                "status": "completed-calibrated-mismatch-survivor-test",
                "classification": registration["classification"],
                "registration": args.registration,
                "runtime_fingerprint": conventions.fingerprint,
                "figure6_gates": figure6_gates,
                "calibrated_projection_id": projection_id,
                "fixed_match_trn_events_by_scale": fixed_match_trn,
                "outcomes": outcomes,
                "original_smart_reproduced": False,
                "baseline_promoted": False,
            },
            sort_keys=False,
        ),
        end="",
    )


if __name__ == "__main__":
    main()
