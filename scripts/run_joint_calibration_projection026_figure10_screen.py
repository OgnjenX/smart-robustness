"""Screen a causal projection-026 reduction grid on target-resolved Figure 10."""

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
from smart_robustness.validation.calibration import runtime_conventions_for_candidate
from smart_robustness.validation.figure6 import Figure6LearningProtocol, run_figure6_learning
from smart_robustness.validation.figure10_search_cycle_spread import (
    build_projection036_variance_sector,
    run_figure10_search_cycle_spread_condition,
)

PROJECTION025_ID = "modeldb112923.projection.025"
PROJECTION026_ID = "modeldb112923.projection.026"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True)
    args = parser.parse_args()
    registration = yaml.safe_load(Path(args.registration).read_text())
    authorization = Path(registration["authorization"])
    if _sha256(authorization) != registration["authorization_sha256"]:
        raise ValueError("projection-026 authorization differs from registration")

    profile = yaml.safe_load(Path(registration["profile"]).read_text())
    training_profile = yaml.safe_load(Path(profile["training_profile"]).read_text())
    base_profile = yaml.safe_load(Path(training_profile["base_profile"]).read_text())
    runtime_overrides = {
        **training_profile["runtime_overrides"],
        **profile["runtime_overrides"],
        "nonspecific_dendritic_calcium_density_scale": float(registration["nonspecific_t_scale"]),
    }
    conventions = replace(
        runtime_conventions_for_candidate(base_profile["candidate"]),
        **runtime_overrides,
    )
    if conventions.fingerprint != registration["runtime_fingerprint"]:
        raise ValueError("runtime differs from registered calibration endpoint")

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
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

    original_builder = classic_sector.build_first_order_connected_sector
    classic_sector.build_first_order_connected_sector = build_projection036_variance_sector
    try:
        training = run_figure6_learning(
            conventions=conventions,
            protocol=Figure6LearningProtocol(
                monitored_populations=tuple(training_profile["monitored_populations"])
            ),
            projection_weight_scales=base_scales,
            brian=brian,
        )
    finally:
        classic_sector.build_first_order_connected_sector = original_builder
    figure6_gates = _figure6_gates(training, profile)
    if not all(figure6_gates.values()):
        raise RuntimeError(f"fresh Figure 6 prerequisite failed: {figure6_gates}")

    protocol = profile["figure10_protocol"]
    comparator = profile["comparator"]
    release_ms = float(protocol["release_after_mismatch_ms"])
    late_start_ms = float(registration["late_reset_start_ms"])
    outcomes = []
    for projection026_scale in registration["projection026_scale_grid"]:
        scales = dict(base_scales)
        scales[PROJECTION025_ID] = float(registration["projection025_scale"])
        if float(projection026_scale) != 1.0:
            scales[PROJECTION026_ID] = float(projection026_scale)
        common = {
            "top_down_current_pA": float(protocol["top_down_current_pA"]),
            "pre_match_duration_ms": float(protocol["pre_match_duration_ms"]),
            "mismatch_duration_ms": float(protocol["mismatch_duration_ms"]),
            "release_after_mismatch_ms": release_ms,
            "learned_weights": training.learned_weights,
            "persistent_projection_weight_scales": scales,
            "persistent_projection_delays_ms": {},
            "comparator_top_k_targets": int(comparator["target_count"]),
            "comparator_source_index": int(comparator["source_index"]),
            "top_down_current_mode": protocol["top_down_current_mode"],
            "conventions": conventions,
            "dt_ms": float(protocol["dt_ms"]),
            "brian": brian,
        }
        intact = run_figure10_search_cycle_spread_condition(
            reset_pathway_enabled=True,
            **common,
        )
        control = run_figure10_search_cycle_spread_condition(
            reset_pathway_enabled=False,
            **common,
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
        gates = {
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
            "aggregate_winner_suppression": (
                intact.winner_post_events < control.winner_post_events
            ),
            "intact_replacement_winner": bool(intact_summary["replacement_winner_candidates"]),
            "reset_favors_replacement_margin": (
                intact_summary["late_alternative_margin"]
                > control_summary["late_alternative_margin"]
            ),
        }
        outcomes.append(
            {
                "projection026_scale": float(projection026_scale),
                "intact": intact_summary,
                "disconnected_control": control_summary,
                "figure10_gates": gates,
                "figure10_pass": all(gates.values()),
            }
        )

    passing_scales = [item["projection026_scale"] for item in outcomes if item["figure10_pass"]]
    selected_scale = max(passing_scales) if passing_scales else None
    print(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "id": registration["result_id"],
                "date": registration["date"],
                "status": "completed-projection026-figure10-screen",
                "classification": registration["classification"],
                "registration": args.registration,
                "runtime_fingerprint": conventions.fingerprint,
                "nonspecific_t_scale": registration["nonspecific_t_scale"],
                "projection025_scale": registration["projection025_scale"],
                "figure6_gates": figure6_gates,
                "outcomes": outcomes,
                "passing_scales": passing_scales,
                "selected_largest_passing_scale": selected_scale,
                "figure7_rerun_in_this_stage": False,
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
