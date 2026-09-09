"""Confirm one fixed calibrated endpoint across SMART Figures 6, 7, and 10."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import yaml
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


def _condition_summary(result) -> dict[str, object]:
    return {
        "relay_events": len(result.relay_spike_times_ms),
        "relay_active_indices": sorted(set(result.relay_spike_indices)),
        "trn_events": len(result.trn_spike_times_ms),
        "nonspecific_events": len(result.nonspecific_spike_times_ms),
        "nonspecific_event_times_ms": list(result.nonspecific_spike_times_ms),
        "trn_to_nonspecific_gaba_integral_ms": result.nonspecific_trn_gaba_integral_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True)
    args = parser.parse_args()
    registration = yaml.safe_load(Path(args.registration).read_text())
    authorization = Path(registration["authorization"])
    if _sha256(authorization) != registration["authorization_sha256"]:
        raise ValueError("joint-confirmation authorization differs from registration")

    profile = yaml.safe_load(Path(registration["profile"]).read_text())
    training_profile = yaml.safe_load(Path(profile["training_profile"]).read_text())
    base_profile = yaml.safe_load(Path(training_profile["base_profile"]).read_text())
    fixed_endpoint = registration["fixed_endpoint"]
    runtime_overrides = {
        **training_profile["runtime_overrides"],
        **profile["runtime_overrides"],
        "nonspecific_dendritic_calcium_density_scale": float(
            fixed_endpoint["nonspecific_t_scale"]
        ),
    }
    conventions = replace(
        runtime_conventions_for_candidate(base_profile["candidate"]),
        **runtime_overrides,
    )
    if conventions.fingerprint != registration["runtime_fingerprint"]:
        raise ValueError("runtime differs from registered joint endpoint")

    scales = {
        str(key): float(value)
        for key, value in training_profile["projection_weight_scales"].items()
    }
    for item in profile["persistent_projection_scales"]:
        projection_id = str(item["projection_id"])
        if projection_id in scales:
            raise ValueError(f"{projection_id} overlaps an existing scale")
        scales[projection_id] = float(item["scale"])
    for projection_id, value in (
        (PROJECTION025_ID, fixed_endpoint["projection025_scale"]),
        (PROJECTION026_ID, fixed_endpoint["projection026_scale"]),
    ):
        if projection_id in scales:
            raise ValueError(f"{projection_id} unexpectedly present before confirmation")
        scales[projection_id] = float(value)

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    original_builder = classic_sector.build_first_order_connected_sector
    try:
        classic_sector.build_first_order_connected_sector = build_projection036_variance_sector
        training = run_figure6_learning(
            conventions=conventions,
            protocol=Figure6LearningProtocol(
                monitored_populations=tuple(training_profile["monitored_populations"])
            ),
            projection_weight_scales=scales,
            brian=brian,
        )
        figure6_gates = _figure6_gates(training, profile)
        if not all(figure6_gates.values()):
            raise RuntimeError(f"fresh Figure 6 prerequisite failed: {figure6_gates}")

        figure7_protocol = profile["figure7_protocol"]
        comparator = profile["comparator"]
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
        match = figure7_module.run_figure7_condition(
            condition=MatchCondition.MATCH,
            **figure7_common,
        )
        mismatch = figure7_module.run_figure7_condition(
            condition=MatchCondition.MISMATCH,
            **figure7_common,
        )
    finally:
        classic_sector.build_first_order_connected_sector = original_builder

    match_summary = _condition_summary(match)
    mismatch_summary = _condition_summary(mismatch)
    expected_match_indices = profile["figure7_gates"]["match_relay_active_indices"]
    expected_mismatch_indices = profile["figure7_gates"]["mismatch_relay_allowed_indices"]
    figure7_gates = {
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
        "match_more_trn_events": match_summary["trn_events"] > mismatch_summary["trn_events"],
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

    figure10_protocol = profile["figure10_protocol"]
    release_ms = float(figure10_protocol["release_after_mismatch_ms"])
    late_start_ms = float(registration["late_reset_start_ms"])
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
    figure10_gates = {
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
    joint_pass = (
        all(figure6_gates.values())
        and all(figure7_gates.values())
        and all(figure10_gates.values())
    )
    print(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "id": registration["result_id"],
                "date": registration["date"],
                "status": "completed-first-order-joint-confirmation",
                "classification": registration["classification"],
                "registration": args.registration,
                "runtime_fingerprint": conventions.fingerprint,
                "nonspecific_t_scale": fixed_endpoint["nonspecific_t_scale"],
                "projection_weight_scales": scales,
                "projection036_spread_convention": "variance",
                "figure6_gates": figure6_gates,
                "figure7": {
                    "match": match_summary,
                    "mismatch": mismatch_summary,
                    "gates": figure7_gates,
                    "pass": all(figure7_gates.values()),
                },
                "figure10": {
                    "intact": intact_summary,
                    "disconnected_control": control_summary,
                    "gates": figure10_gates,
                    "pass": all(figure10_gates.values()),
                },
                "joint_first_order_pass": joint_pass,
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
