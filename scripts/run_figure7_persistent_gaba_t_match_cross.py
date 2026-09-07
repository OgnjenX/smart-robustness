"""Run the preregistered persistent projection-008 by T-current match cross."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from pathlib import Path

import yaml

from smart_robustness.protocols import MatchCondition
from smart_robustness.validation.calibration import runtime_conventions_for_candidate
from smart_robustness.validation.figure6 import (
    Figure6LearningProtocol,
    assess_figure6_cortical_recruitment,
    assess_figure6_top_down_timing,
    run_figure6_learning,
)
from smart_robustness.validation.figure7 import TopDownCurrentMode, run_figure7_condition


def _figure6_gates(training, profile):
    expected = set(profile["figure6_gates"]["relay_active_indices"])
    relay_indices = training.result.population_spike_indices["thalamic_relay"]
    relay_counts = {index: relay_indices.count(index) for index in expected}
    result = training.result
    recruitment = assess_figure6_cortical_recruitment(result)
    timing = assess_figure6_top_down_timing(result)
    return {
        "relay_active_indices": set(relay_indices) == expected,
        "relay_events_per_active_index": set(relay_counts.values()) == {4},
        "relay_events": len(relay_indices) == 20,
        "cortical_chain_complete": recruitment.feedforward_chain_complete,
        "causal_learning_pair": timing.causal_pair_in_learning_window,
        "top_down_horizontal_contrast": (
            result.top_down_combined.horizontal_orientation_contrast > 0
        ),
    }


def _match_summary(result, profile):
    counts = Counter(int(index) for index in result.trn_spike_indices)
    diagnostic = (22, 31, 38, 39, 40, 41, 42, 49, 58)
    diagnostic_total = sum(counts[index] for index in diagnostic)
    summary = {
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
    summary["match_gates"] = {
        "relay_active_indices": summary["relay_active_indices"]
        == profile["figure7_gates"]["match_relay_active_indices"],
        "relay_events": summary["relay_event_count"]
        == profile["figure7_gates"]["match_relay_events"],
        "nonspecific_events": summary["nonspecific_event_count"]
        == profile["figure7_gates"]["match_nonspecific_events"],
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True)
    args = parser.parse_args()
    registration = yaml.safe_load(Path(args.registration).read_text())
    profile = yaml.safe_load(Path(registration["profile"]).read_text())
    training_profile = yaml.safe_load(Path(profile["training_profile"]).read_text())
    base_profile = yaml.safe_load(Path(training_profile["base_profile"]).read_text())
    base_conventions = replace(
        runtime_conventions_for_candidate(base_profile["candidate"]),
        **training_profile["runtime_overrides"],
        **profile["runtime_overrides"],
    )
    if base_conventions.fingerprint != registration["base_runtime_fingerprint"]:
        raise ValueError("registration runtime differs from executable profile")

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    base_scales = {
        str(key): float(value)
        for key, value in training_profile["projection_weight_scales"].items()
    }
    projection_id = registration["persistent_projection_scale"]["projection_id"]
    if projection_id in base_scales:
        raise ValueError("calibrated projection overlaps an existing scale")
    persistent_scales = {
        **base_scales,
        projection_id: float(registration["persistent_projection_scale"]["scale"]),
    }
    protocol = profile["protocol"]
    comparator = profile["comparator"]
    outcomes = []
    for t_scale in registration["nonspecific_t_scale_grid"]:
        conventions = replace(
            base_conventions,
            nonspecific_dendritic_calcium_density_scale=float(t_scale),
        )
        training = run_figure6_learning(
            conventions=conventions,
            protocol=Figure6LearningProtocol(
                monitored_populations=tuple(training_profile["monitored_populations"])
            ),
            projection_weight_scales=persistent_scales,
            brian=brian,
        )
        figure6_gates = _figure6_gates(training, profile)
        outcome = {
            "nonspecific_t_scale": float(t_scale),
            "effective_density_mS_cm2": 250.0 * float(t_scale),
            "runtime_fingerprint": conventions.fingerprint,
            "figure6_gates": figure6_gates,
            "match": None,
        }
        if all(figure6_gates.values()):
            match = run_figure7_condition(
                condition=MatchCondition.MATCH,
                learned_weights=training.learned_weights,
                conventions=conventions,
                persistent_projection_weight_scales=persistent_scales,
                top_down_current_pA=float(protocol["top_down_current_pA"]),
                top_down_current_mode=TopDownCurrentMode(
                    protocol["top_down_current_mode"]
                ),
                top_down_cue_lead_ms=float(protocol["top_down_cue_lead_ms"]),
                duration_ms=float(protocol["duration_ms"]),
                dt_ms=float(protocol["dt_ms"]),
                equilibration_ms=float(protocol["equilibration_ms"]),
                comparator_top_k_targets=int(comparator["target_count"]),
                comparator_source_index=int(comparator["source_index"]),
                brian=brian,
            )
            outcome["match"] = _match_summary(match, profile)
        outcomes.append(outcome)

    print(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "id": registration["result_id"],
                "date": registration["date"],
                "status": "completed-persistent-gaba-t-match-cross",
                "classification": registration["classification"],
                "registration": args.registration,
                "base_runtime_fingerprint": base_conventions.fingerprint,
                "persistent_projection_weight_scales": persistent_scales,
                "outcomes": outcomes,
                "mismatch_consulted": False,
                "original_smart_reproduced": False,
                "baseline_promoted": False,
            },
            sort_keys=False,
        ),
        end="",
    )


if __name__ == "__main__":
    main()
