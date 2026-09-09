"""Screen a fixed projection-025 grid on Figure 10 after a Figure-6/7 pass."""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import replace
from pathlib import Path

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
    build_projection036_variance_sector,
    run_figure10_search_cycle_spread_condition,
)

PROJECTION025_ID = "modeldb112923.projection.025"


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
        "pre_layer4_events": result.pre_layer4_events,
        "pre_layer4_active_indices": list(result.pre_layer4_active_indices),
        "winner_post_events": result.winner_post_events,
        "alternative_events": result.alternative_events,
        "alternative_active_indices": list(result.alternative_active_indices),
        "first_alternative_event_ms": result.first_alternative_event_ms,
        "alternative_events_before_release": (result.alternative_events_before_release),
        "nonspecific_post_events": result.nonspecific_post_events,
        "layer5_post_events": result.layer5_post_events,
        "layer6i_post_events": result.layer6i_post_events,
        "layer4_inhibitory_post_events": result.layer4_inhibitory_post_events,
        "projection035_post_release_integral_pA_ms": [
            list(item) for item in result.projection035_post_release_integral_pA_ms
        ],
        "projection036_nonzero_edges": result.projection036_nonzero_edges,
        "projection036_incoming_weight_sum": (result.projection036_incoming_weight_sum),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True)
    args = parser.parse_args()
    registration_path = Path(args.registration)
    registration = yaml.safe_load(registration_path.read_text())
    stage2_path = Path(registration["stage2_result"])
    if _sha256(stage2_path) != registration["stage2_result_sha256"]:
        raise ValueError("Stage-2 result differs from the registered artifact")
    stage2 = yaml.safe_load(stage2_path.read_text())
    survivors = [item for item in stage2["outcomes"] if item["joint_figure6_7_pass"]]
    if len(survivors) != 1:
        raise ValueError("Figure-10 screen requires exactly one Figure-6/7 survivor")
    survivor = survivors[0]
    if survivor["nonspecific_t_scale"] != registration["nonspecific_t_scale"]:
        raise ValueError("registered T-current scale differs from Stage-2 survivor")

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
        raise ValueError("registration runtime differs from executable profile")

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
    if PROJECTION025_ID in base_scales:
        raise ValueError("training prerequisite requires released projection 025")

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
    outcomes = []
    for scale in registration["projection025_scale_grid"]:
        trial_scales = dict(base_scales)
        if float(scale) != 1.0:
            trial_scales[PROJECTION025_ID] = float(scale)
        common = {
            "top_down_current_pA": float(protocol["top_down_current_pA"]),
            "pre_match_duration_ms": float(protocol["pre_match_duration_ms"]),
            "mismatch_duration_ms": float(protocol["mismatch_duration_ms"]),
            "release_after_mismatch_ms": float(protocol["release_after_mismatch_ms"]),
            "learned_weights": training.learned_weights,
            "persistent_projection_weight_scales": trial_scales,
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
        gates = {
            "identical_nonempty_prestate": (
                intact.pre_layer4_events > 0
                and intact.pre_layer4_events == control.pre_layer4_events
                and intact.pre_layer4_active_indices == control.pre_layer4_active_indices
            ),
            "alternatives_quiet_before_release": (
                intact.alternative_events_before_release == 0
                and control.alternative_events_before_release == 0
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
            "winner_suppression": (intact.winner_post_events < control.winner_post_events),
            "earlier_intact_alternative": (
                intact.first_alternative_event_ms is not None
                and (
                    control.first_alternative_event_ms is None
                    or intact.first_alternative_event_ms < control.first_alternative_event_ms
                )
            ),
        }
        outcomes.append(
            {
                "projection025_scale": float(scale),
                "intact": _condition_summary(intact),
                "disconnected_control": _condition_summary(control),
                "figure10_gates": gates,
                "figure10_pass": all(gates.values()),
            }
        )

    passing_scales = [item["projection025_scale"] for item in outcomes if item["figure10_pass"]]
    selected_scale = min(passing_scales) if passing_scales else None
    print(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "id": registration["result_id"],
                "date": registration["date"],
                "status": "completed-projection025-figure10-screen",
                "classification": registration["classification"],
                "registration": args.registration,
                "stage2_result": str(stage2_path),
                "runtime_fingerprint": conventions.fingerprint,
                "nonspecific_t_scale": registration["nonspecific_t_scale"],
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
