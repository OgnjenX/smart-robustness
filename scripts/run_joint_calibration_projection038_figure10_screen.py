"""Screen a causal projection-038 balance grid on target-resolved Figure 10."""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

from smart_robustness import classic_sector
from smart_robustness.validation.calibration import runtime_conventions_for_candidate
from smart_robustness.validation.figure6 import (
    Figure6LearningProtocol,
    assess_figure6_cortical_recruitment,
    assess_figure6_top_down_timing,
    run_figure6_learning,
)
from smart_robustness.validation.figure10_search_cycle_spread import (
    VERTICAL_ALTERNATIVES,
    VERTICAL_TARGETS,
    build_projection036_variance_sector,
    run_figure10_search_cycle_spread_condition,
)

PROJECTION025_ID = "modeldb112923.projection.025"
PROJECTION038_ID = "modeldb112923.projection.038"


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


def _target_summary(result, *, release_ms: float, late_start_ms: float) -> dict:
    events = result.mismatch_layer4_events
    indices = np.asarray([item[0] for item in events], dtype=int)
    times = np.asarray([item[1] for item in events], dtype=float)
    pre_release_alternative_events = int(
        np.count_nonzero((times < release_ms) & np.isin(indices, VERTICAL_ALTERNATIVES))
    )
    rows = []
    for target in VERTICAL_TARGETS:
        target_times = times[(times >= late_start_ms) & (indices == target)]
        rows.append(
            {
                "target": target,
                "events": int(target_times.size),
                "first_event_ms": (None if target_times.size == 0 else float(target_times[0])),
                "last_event_ms": (None if target_times.size == 0 else float(target_times[-1])),
            }
        )
    counts = {row["target"]: row["events"] for row in rows}
    max_alternative = max(counts[target] for target in VERTICAL_ALTERNATIVES)
    replacement_winners = sorted(
        target
        for target in VERTICAL_ALTERNATIVES
        if counts[target] == max_alternative and counts[target] > counts[40]
    )
    return {
        "pre_layer4_events": result.pre_layer4_events,
        "pre_layer4_active_indices": list(result.pre_layer4_active_indices),
        "aggregate_winner_events": result.winner_post_events,
        "aggregate_alternative_events": result.alternative_events,
        "pre_release_alternative_events": pre_release_alternative_events,
        "late_reset_targets": rows,
        "late_old_center_events": counts[40],
        "late_max_alternative_events": max_alternative,
        "late_alternative_margin": max_alternative - counts[40],
        "replacement_winner_candidates": replacement_winners,
        "nonspecific_post_events": result.nonspecific_post_events,
        "layer5_post_events": result.layer5_post_events,
        "layer6i_post_events": result.layer6i_post_events,
        "projection035_post_release_integral_pA_ms": [
            list(item) for item in result.projection035_post_release_integral_pA_ms
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True)
    args = parser.parse_args()
    registration = yaml.safe_load(Path(args.registration).read_text())
    authorization = Path(registration["authorization"])
    if _sha256(authorization) != registration["authorization_sha256"]:
        raise ValueError("target-resolved authorization differs from registration")

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
    if PROJECTION025_ID in base_scales or PROJECTION038_ID in base_scales:
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
    for projection038_scale in registration["projection038_scale_grid"]:
        scales = dict(base_scales)
        scales[PROJECTION025_ID] = float(registration["projection025_scale"])
        if float(projection038_scale) != 1.0:
            scales[PROJECTION038_ID] = float(projection038_scale)
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
                "projection038_scale": float(projection038_scale),
                "intact": intact_summary,
                "disconnected_control": control_summary,
                "figure10_gates": gates,
                "figure10_pass": all(gates.values()),
            }
        )

    passing_scales = [item["projection038_scale"] for item in outcomes if item["figure10_pass"]]
    selected_scale = min(passing_scales) if passing_scales else None
    print(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "id": registration["result_id"],
                "date": registration["date"],
                "status": "completed-projection038-figure10-screen",
                "classification": registration["classification"],
                "registration": args.registration,
                "runtime_fingerprint": conventions.fingerprint,
                "nonspecific_t_scale": registration["nonspecific_t_scale"],
                "projection025_scale": registration["projection025_scale"],
                "figure6_gates": figure6_gates,
                "outcomes": outcomes,
                "passing_scales": passing_scales,
                "selected_smallest_passing_scale": selected_scale,
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
