"""Test every Stage-1 T-current survivor on fresh match and mismatch runs."""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import replace
from pathlib import Path

import yaml

from smart_robustness import classic_sector
from smart_robustness.protocols import MatchCondition
from smart_robustness.validation import figure7 as figure7_module
from smart_robustness.validation.calibration import runtime_conventions_for_candidate
from smart_robustness.validation.figure6 import (
    Figure6LearningProtocol,
    assess_figure6_cortical_recruitment,
    assess_figure6_top_down_timing,
    run_figure6_learning,
)
from smart_robustness.validation.figure7 import TopDownCurrentMode
from smart_robustness.validation.figure10_search_cycle_spread import (
    build_projection036_variance_sector,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _figure6_gates(training, profile) -> dict[str, bool]:
    expected = set(profile["figure6_gates"]["relay_active_indices"])
    relay_indices = training.result.population_spike_indices["thalamic_relay"]
    relay_counts = {index: relay_indices.count(index) for index in expected}
    recruitment = assess_figure6_cortical_recruitment(training.result)
    timing = assess_figure6_top_down_timing(training.result)
    return {
        "relay_active_indices": set(relay_indices) == expected,
        "relay_events_per_active_index": set(relay_counts.values()) == {4},
        "relay_events": len(relay_indices) == 20,
        "cortical_chain_complete": recruitment.feedforward_chain_complete,
        "causal_learning_pair": timing.causal_pair_in_learning_window,
        "top_down_horizontal_contrast": (
            training.result.top_down_combined.horizontal_orientation_contrast > 0
        ),
    }


def _condition_summary(result) -> dict[str, object]:
    return {
        "relay_events": len(result.relay_spike_times_ms),
        "relay_active_indices": sorted(set(result.relay_spike_indices)),
        "trn_events": len(result.trn_spike_times_ms),
        "nonspecific_events": len(result.nonspecific_spike_times_ms),
        "nonspecific_event_times_ms": list(result.nonspecific_spike_times_ms),
        "trn_to_nonspecific_gaba_integral_ms": (result.nonspecific_trn_gaba_integral_ms),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True)
    args = parser.parse_args()
    registration_path = Path(args.registration)
    registration = yaml.safe_load(registration_path.read_text())
    stage1_path = Path(registration["stage1_result"])
    if _sha256(stage1_path) != registration["stage1_result_sha256"]:
        raise ValueError("Stage-1 result differs from the registered artifact")
    stage1 = yaml.safe_load(stage1_path.read_text())
    observed_survivors = [
        item["nonspecific_t_scale"]
        for item in stage1["outcomes"]
        if item["match"]["exact_match_gate"]
    ]
    if observed_survivors != registration["survivors"]:
        raise ValueError("registered survivors differ from the Stage-1 result")

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
    scales = {
        str(key): float(value)
        for key, value in training_profile["projection_weight_scales"].items()
    }
    for item in profile["persistent_projection_scales"]:
        projection_id = str(item["projection_id"])
        if projection_id in scales:
            raise ValueError(f"{projection_id} overlaps an existing scale")
        scales[projection_id] = float(item["scale"])
    if "modeldb112923.projection.025" in scales:
        raise ValueError("Stage 2 requires released projection-025 scale 1")

    protocol = profile["figure7_protocol"]
    comparator = profile["comparator"]
    expected_match_indices = profile["figure7_gates"]["match_relay_active_indices"]
    expected_mismatch_indices = profile["figure7_gates"]["mismatch_relay_allowed_indices"]
    outcomes = []
    original_builder = classic_sector.build_first_order_connected_sector
    try:
        classic_sector.build_first_order_connected_sector = build_projection036_variance_sector
        for t_scale in registration["survivors"]:
            conventions = replace(
                base_conventions,
                nonspecific_dendritic_calcium_density_scale=float(t_scale),
            )
            training = run_figure6_learning(
                conventions=conventions,
                protocol=Figure6LearningProtocol(
                    monitored_populations=tuple(training_profile["monitored_populations"])
                ),
                projection_weight_scales=scales,
                brian=brian,
            )
            figure6_gates = _figure6_gates(training, profile)
            outcome = {
                "nonspecific_t_scale": float(t_scale),
                "effective_density_mS_cm2": 250.0 * float(t_scale),
                "runtime_fingerprint": conventions.fingerprint,
                "figure6_gates": figure6_gates,
                "match": None,
                "mismatch": None,
                "figure7_gates": None,
            }
            if all(figure6_gates.values()):
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
                    "record_relay_diagnostics": True,
                    "brian": brian,
                }
                match = figure7_module.run_figure7_condition(
                    condition=MatchCondition.MATCH,
                    **common,
                )
                mismatch = figure7_module.run_figure7_condition(
                    condition=MatchCondition.MISMATCH,
                    **common,
                )
                match_summary = _condition_summary(match)
                mismatch_summary = _condition_summary(mismatch)
                gates = {
                    "match_relay_active_indices": (
                        match_summary["relay_active_indices"] == expected_match_indices
                    ),
                    "match_relay_events": (
                        match_summary["relay_events"]
                        == profile["figure7_gates"]["match_relay_events"]
                    ),
                    "mismatch_relay_allowed_indices": (
                        mismatch_summary["relay_active_indices"] == expected_mismatch_indices
                    ),
                    "match_more_active_relay_cells": (
                        len(match_summary["relay_active_indices"])
                        > len(mismatch_summary["relay_active_indices"])
                    ),
                    "match_more_trn_events": (
                        match_summary["trn_events"] > mismatch_summary["trn_events"]
                    ),
                    "match_more_trn_to_nonspecific_gaba": (
                        match_summary["trn_to_nonspecific_gaba_integral_ms"]
                        > mismatch_summary["trn_to_nonspecific_gaba_integral_ms"]
                    ),
                    "match_nonspecific_events": (
                        match_summary["nonspecific_events"]
                        == profile["figure7_gates"]["match_nonspecific_events"]
                    ),
                    "mismatch_nonspecific_events": (
                        mismatch_summary["nonspecific_events"]
                        == profile["figure7_gates"]["mismatch_nonspecific_events"]
                    ),
                }
                outcome["match"] = match_summary
                outcome["mismatch"] = mismatch_summary
                outcome["figure7_gates"] = gates
                outcome["joint_figure6_7_pass"] = all(gates.values())
            else:
                outcome["joint_figure6_7_pass"] = False
            outcomes.append(outcome)
    finally:
        classic_sector.build_first_order_connected_sector = original_builder

    print(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "id": registration["result_id"],
                "date": registration["date"],
                "status": "completed-joint-calibration-t-mismatch-screen",
                "classification": registration["classification"],
                "registration": args.registration,
                "stage1_result": str(stage1_path),
                "base_runtime_fingerprint": base_conventions.fingerprint,
                "persistent_projection_weight_scales": scales,
                "projection025_runtime_scale": 1.0,
                "projection036_spread_convention": "variance",
                "outcomes": outcomes,
                "figure10_consulted": False,
                "holdouts_consulted": False,
                "original_smart_reproduced": False,
                "baseline_promoted": False,
                "baseline_frozen": False,
            },
            sort_keys=False,
        ),
        end="",
    )


if __name__ == "__main__":
    main()
