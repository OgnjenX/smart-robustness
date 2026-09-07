"""Run the one authorized connected match/mismatch pair at effective T scale."""

from __future__ import annotations

import argparse
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
from smart_robustness.validation.figure7 import (
    TopDownCurrentMode,
    assess_figure7_reproduction,
    run_figure7_condition,
)


def _condition_summary(result):
    return {
        "relay_event_count": len(result.relay_spike_times_ms),
        "relay_active_indices": sorted(set(result.relay_spike_indices)),
        "relay_spike_indices": list(result.relay_spike_indices),
        "relay_spike_times_ms": list(result.relay_spike_times_ms),
        "trn_event_count": len(result.trn_spike_times_ms),
        "nonspecific_event_count": len(result.nonspecific_spike_times_ms),
        "nonspecific_rate_hz": result.nonspecific_rate_hz,
        "nonspecific_spike_times_ms": list(result.nonspecific_spike_times_ms),
        "category_event_count": len(result.category_spike_times_ms),
        "top_down_current_termination_time_ms": (
            result.top_down_current_termination_time_ms
        ),
        "comparator_transform": result.comparator_transform,
        "relay_calcium_ablated": result.relay_calcium_ablated_at_stimulus,
        "nonspecific_calcium_ablated": (
            result.nonspecific_calcium_ablated_at_stimulus
        ),
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
    recruitment = assess_figure6_cortical_recruitment(training.result)
    timing = assess_figure6_top_down_timing(training.result)
    figure6_gates = {
        "relay_active_indices": set(relay_indices) == expected,
        "relay_events_per_active_index": set(relay_counts.values()) == {4},
        "relay_events": len(relay_indices) == 20,
        "cortical_chain_complete": recruitment.feedforward_chain_complete,
        "causal_learning_pair": timing.causal_pair_in_learning_window,
        "top_down_horizontal_contrast": (
            training.result.top_down_combined.horizontal_orientation_contrast > 0
        ),
    }
    if not all(figure6_gates.values()):
        artifact = {
            "schema_version": 1,
            "id": registration["result_id"],
            "date": registration["date"],
            "status": "figure6-failed",
            "registration": args.registration,
            "runtime_fingerprint": conventions.fingerprint,
            "figure6_gates": figure6_gates,
            "match_constructed": False,
            "mismatch_constructed": False,
            "reproduced": False,
        }
        print(yaml.safe_dump(artifact, sort_keys=False), end="")
        return

    protocol = profile["protocol"]
    common = dict(
        learned_weights=training.learned_weights,
        conventions=conventions,
        persistent_projection_weight_scales=scales,
        top_down_current_pA=float(protocol["top_down_current_pA"]),
        top_down_current_mode=TopDownCurrentMode(protocol["top_down_current_mode"]),
        top_down_cue_lead_ms=float(protocol["top_down_cue_lead_ms"]),
        duration_ms=float(protocol["duration_ms"]),
        dt_ms=float(protocol["dt_ms"]),
        equilibration_ms=float(protocol["equilibration_ms"]),
        brian=brian,
    )
    match = run_figure7_condition(condition=MatchCondition.MATCH, **common)
    mismatch = run_figure7_condition(condition=MatchCondition.MISMATCH, **common)
    assessment = assess_figure7_reproduction(match, mismatch)
    gates = {
        "figure6_all_existing_gates": all(figure6_gates.values()),
        "match_relay_spatial_set": assessment.pathway.relay_spatial_match_pass,
        "mismatch_relay_overlap_only": assessment.pathway.relay_mismatch_overlap_pass,
        "match_more_active_relay_cells": assessment.pathway.relay_subset_pass,
        "match_more_trn_events": assessment.pathway.trn_order_pass,
        "match_nonspecific_40_hz": assessment.arousal.match_numeric_target_pass,
        "mismatch_more_nonspecific_events": (
            assessment.arousal.mismatch_disinhibition_pass
        ),
        "mismatch_nonspecific_70_hz": (
            assessment.arousal.mismatch_numeric_target_pass
        ),
        "figure7_target_duration": assessment.arousal.target_duration_pass,
        "no_reconstructed_comparator": (
            match.comparator_transform is None
            and mismatch.comparator_transform is None
        ),
        "no_calcium_ablation": not any(
            (
                match.relay_calcium_ablated_at_stimulus,
                match.nonspecific_calcium_ablated_at_stimulus,
                mismatch.relay_calcium_ablated_at_stimulus,
                mismatch.nonspecific_calcium_ablated_at_stimulus,
            )
        ),
    }
    reproduced = all(gates.values()) and assessment.reproduced
    artifact = {
        "schema_version": 1,
        "id": registration["result_id"],
        "date": registration["date"],
        "status": "figure7-calibrated-reproduced" if reproduced else "figure7-failed",
        "classification": registration["classification"],
        "registration": args.registration,
        "runtime_fingerprint": conventions.fingerprint,
        "nonspecific_dendritic_calcium_density_scale": (
            conventions.nonspecific_dendritic_calcium_density_scale
        ),
        "figure6_population_spikes": training.result.population_spikes,
        "figure6_gates": figure6_gates,
        "match": _condition_summary(match),
        "mismatch": _condition_summary(mismatch),
        "gates": gates,
        "reproduced": reproduced,
        "original_smart_reproduced": False,
        "baseline_promoted": False,
        "classification_boundary": registration["classification_boundary"],
    }
    print(yaml.safe_dump(artifact, sort_keys=False), end="")


if __name__ == "__main__":
    main()
