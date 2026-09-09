"""Screen remaining projection-026 endpoints with fresh global retraining."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import yaml
from run_joint_calibration_first_order_confirmation import _condition_summary
from run_joint_calibration_projection038_figure10_screen import (
    _figure6_gates,
    _sha256,
    _target_summary,
)

from smart_robustness import classic_sector
from smart_robustness.protocols import MatchCondition
from smart_robustness.validation import figure7 as figure7_module
from smart_robustness.validation.calibration import runtime_conventions_for_candidate
from smart_robustness.validation.figure6 import Figure6LearningProtocol, run_figure6_learning
from smart_robustness.validation.figure7 import TopDownCurrentMode
from smart_robustness.validation.figure10_search_cycle_spread import (
    build_projection036_variance_sector,
    run_figure10_search_cycle_spread_condition,
)

PROJECTION025_ID = "modeldb112923.projection.025"
PROJECTION026_ID = "modeldb112923.projection.026"


def _figure7_gates(match, mismatch, profile) -> dict[str, bool]:
    expected_match = profile["figure7_gates"]["match_relay_active_indices"]
    expected_mismatch = profile["figure7_gates"]["mismatch_relay_allowed_indices"]
    return {
        "match_relay_active_indices": match["relay_active_indices"] == expected_match,
        "match_relay_events": (
            match["relay_events"] == profile["figure7_gates"]["match_relay_events"]
        ),
        "mismatch_relay_allowed_indices": (
            mismatch["relay_active_indices"] == expected_mismatch
        ),
        "match_more_active_relay_cells": (
            len(match["relay_active_indices"]) > len(mismatch["relay_active_indices"])
        ),
        "match_more_trn_events": match["trn_events"] > mismatch["trn_events"],
        "match_more_trn_to_nonspecific_gaba": (
            match["trn_to_nonspecific_gaba_integral_ms"]
            > mismatch["trn_to_nonspecific_gaba_integral_ms"]
        ),
        "match_nonspecific_events": (
            match["nonspecific_events"]
            == profile["figure7_gates"]["match_nonspecific_events"]
        ),
        "mismatch_nonspecific_events": (
            mismatch["nonspecific_events"]
            == profile["figure7_gates"]["mismatch_nonspecific_events"]
        ),
    }


def _figure10_gates(intact, control, intact_summary, control_summary) -> dict[str, bool]:
    return {
        "identical_nonempty_prestate": (
            intact.pre_layer4_events > 0
            and intact.pre_layer4_events == control.pre_layer4_events
            and intact.pre_layer4_active_indices == control.pre_layer4_active_indices
        ),
        "alternatives_quiet_before_release": (
            intact_summary["pre_release_alternative_events"] == 0
            and control_summary["pre_release_alternative_events"] == 0
        ),
        "full_input_delivery": all(
            value > 0
            for result in (intact, control)
            for _, value in result.projection035_post_release_integral_pA_ms
        ),
        "intact_reset_chain": (
            intact.nonspecific_post_events > 0
            and intact.layer5_post_events > 0
            and intact.layer6i_post_events > 0
        ),
        "aggregate_winner_suppression": intact.winner_post_events < control.winner_post_events,
        "intact_replacement_winner": bool(intact_summary["replacement_winner_candidates"]),
        "reset_favors_replacement_margin": (
            intact_summary["late_alternative_margin"]
            > control_summary["late_alternative_margin"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True)
    args = parser.parse_args()
    registration = yaml.safe_load(Path(args.registration).read_text())
    authorization = Path(registration["authorization"])
    if _sha256(authorization) != registration["authorization_sha256"]:
        raise ValueError("global projection-026 authorization differs from registration")

    profile = yaml.safe_load(Path(registration["profile"]).read_text())
    training_profile = yaml.safe_load(Path(profile["training_profile"]).read_text())
    base_profile = yaml.safe_load(Path(training_profile["base_profile"]).read_text())
    runtime_overrides = {
        **training_profile["runtime_overrides"],
        **profile["runtime_overrides"],
        "nonspecific_dendritic_calcium_density_scale": float(
            registration["nonspecific_t_scale"]
        ),
    }
    conventions = replace(
        runtime_conventions_for_candidate(base_profile["candidate"]),
        **runtime_overrides,
    )
    if conventions.fingerprint != registration["runtime_fingerprint"]:
        raise ValueError("runtime differs from registered global screen")

    base_scales = {
        str(key): float(value)
        for key, value in training_profile["projection_weight_scales"].items()
    }
    for item in profile["persistent_projection_scales"]:
        projection_id = str(item["projection_id"])
        if projection_id in base_scales:
            raise ValueError(f"{projection_id} overlaps an existing scale")
        base_scales[projection_id] = float(item["scale"])
    if PROJECTION025_ID in base_scales or PROJECTION026_ID in base_scales:
        raise ValueError("screened projections unexpectedly present in base scales")

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    figure7_protocol = profile["figure7_protocol"]
    figure10_protocol = profile["figure10_protocol"]
    comparator = profile["comparator"]
    release_ms = float(figure10_protocol["release_after_mismatch_ms"])
    late_start_ms = float(registration["late_reset_start_ms"])
    outcomes = []
    original_builder = classic_sector.build_first_order_connected_sector
    try:
        classic_sector.build_first_order_connected_sector = build_projection036_variance_sector
        for projection026_scale in registration["projection026_scale_grid"]:
            scales = dict(base_scales)
            scales[PROJECTION025_ID] = float(registration["projection025_scale"])
            if float(projection026_scale) != 1.0:
                scales[PROJECTION026_ID] = float(projection026_scale)
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
                "projection026_scale": float(projection026_scale),
                "global_projection_weight_scales": scales,
                "figure6_gates": figure6_gates,
                "figure7": None,
                "figure10": None,
                "joint_first_order_pass": False,
            }
            if not all(figure6_gates.values()):
                outcomes.append(outcome)
                continue

            figure7_common = {
                "learned_weights": training.learned_weights,
                "conventions": conventions,
                "persistent_projection_weight_scales": scales,
                "top_down_current_pA": float(figure7_protocol["top_down_current_pA"]),
                "top_down_current_mode": TopDownCurrentMode(
                    figure7_protocol["top_down_current_mode"]
                ),
                "top_down_cue_lead_ms": float(figure7_protocol["top_down_cue_lead_ms"]),
                "duration_ms": float(figure7_protocol["duration_ms"]),
                "dt_ms": float(figure7_protocol["dt_ms"]),
                "equilibration_ms": float(figure7_protocol["equilibration_ms"]),
                "comparator_top_k_targets": int(comparator["target_count"]),
                "comparator_source_index": int(comparator["source_index"]),
                "record_relay_diagnostics": True,
                "brian": brian,
            }
            match_result = figure7_module.run_figure7_condition(
                condition=MatchCondition.MATCH,
                **figure7_common,
            )
            mismatch_result = figure7_module.run_figure7_condition(
                condition=MatchCondition.MISMATCH,
                **figure7_common,
            )
            match = _condition_summary(match_result)
            mismatch = _condition_summary(mismatch_result)
            figure7_gates = _figure7_gates(match, mismatch, profile)
            outcome["figure7"] = {
                "match": match,
                "mismatch": mismatch,
                "gates": figure7_gates,
                "pass": all(figure7_gates.values()),
            }
            if not all(figure7_gates.values()):
                outcomes.append(outcome)
                continue

            figure10_common = {
                "top_down_current_pA": float(figure10_protocol["top_down_current_pA"]),
                "pre_match_duration_ms": float(figure10_protocol["pre_match_duration_ms"]),
                "mismatch_duration_ms": float(figure10_protocol["mismatch_duration_ms"]),
                "release_after_mismatch_ms": release_ms,
                "learned_weights": training.learned_weights,
                "persistent_projection_weight_scales": scales,
                "persistent_projection_delays_ms": {},
                "comparator_top_k_targets": int(comparator["target_count"]),
                "comparator_source_index": int(comparator["source_index"]),
                "top_down_current_mode": figure10_protocol["top_down_current_mode"],
                "conventions": conventions,
                "dt_ms": float(figure10_protocol["dt_ms"]),
                "brian": brian,
            }
            intact = run_figure10_search_cycle_spread_condition(
                reset_pathway_enabled=True,
                **figure10_common,
            )
            control = run_figure10_search_cycle_spread_condition(
                reset_pathway_enabled=False,
                **figure10_common,
            )
            intact_summary = _target_summary(
                intact,
                release_ms=release_ms,
                late_start_ms=late_start_ms,
            )
            control_summary = _target_summary(
                control,
                release_ms=release_ms,
                late_start_ms=late_start_ms,
            )
            figure10_gates = _figure10_gates(
                intact,
                control,
                intact_summary,
                control_summary,
            )
            outcome["figure10"] = {
                "intact": intact_summary,
                "disconnected_control": control_summary,
                "gates": figure10_gates,
                "pass": all(figure10_gates.values()),
            }
            outcome["joint_first_order_pass"] = all(figure10_gates.values())
            outcomes.append(outcome)
    finally:
        classic_sector.build_first_order_connected_sector = original_builder

    passing_scales = [
        item["projection026_scale"] for item in outcomes if item["joint_first_order_pass"]
    ]
    print(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "id": registration["result_id"],
                "date": registration["date"],
                "status": "completed-global-projection026-screen",
                "classification": registration["classification"],
                "registration": args.registration,
                "runtime_fingerprint": conventions.fingerprint,
                "nonspecific_t_scale": registration["nonspecific_t_scale"],
                "projection025_scale": registration["projection025_scale"],
                "projection036_spread_convention": "variance",
                "outcomes": outcomes,
                "passing_scales": passing_scales,
                "selected_largest_passing_scale": (
                    max(passing_scales) if passing_scales else None
                ),
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
