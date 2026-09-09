"""Run the preregistered projection-036 spread search-cycle pair."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
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
        **registration.get("runtime_overrides", {}),
    )
    if conventions.fingerprint != registration["runtime_fingerprint"]:
        raise ValueError("registration runtime differs from executable profile")

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    scales = {
        str(key): float(value)
        for key, value in training_profile["projection_weight_scales"].items()
    }
    for item in registration["persistent_projection_scales"]:
        projection_id = str(item["projection_id"])
        if projection_id in scales:
            raise ValueError(f"{projection_id} overlaps an existing scale")
        scales[projection_id] = float(item["scale"])
    delays = {
        str(item["projection_id"]): float(item["delay_ms"])
        for item in registration.get("persistent_projection_delays", ())
    }

    original_builder = classic_sector.build_first_order_connected_sector
    classic_sector.build_first_order_connected_sector = (
        build_projection036_variance_sector
    )
    try:
        training = run_figure6_learning(
            conventions=conventions,
            protocol=Figure6LearningProtocol(
                monitored_populations=tuple(training_profile["monitored_populations"])
            ),
            projection_weight_scales=scales,
            brian=brian,
        )
    finally:
        classic_sector.build_first_order_connected_sector = original_builder
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
        raise RuntimeError(f"fresh Figure 6 prerequisite failed: {figure6_gates}")

    protocol = registration["protocol"]
    common = {
        "top_down_current_pA": float(profile["protocol"]["top_down_current_pA"]),
        "pre_match_duration_ms": float(protocol["pre_match_duration_ms"]),
        "mismatch_duration_ms": float(protocol["mismatch_duration_ms"]),
        "release_after_mismatch_ms": float(
            protocol["release_after_mismatch_ms"]
        ),
        "learned_weights": training.learned_weights,
        "persistent_projection_weight_scales": scales,
        "persistent_projection_delays_ms": delays,
        "comparator_top_k_targets": int(
            profile["pre_release_comparator"]["target_count"]
        ),
        "comparator_source_index": int(
            profile["pre_release_comparator"]["source_index"]
        ),
        "top_down_current_mode": profile["protocol"]["top_down_current_mode"],
        "conventions": conventions,
        "dt_ms": float(protocol["dt_ms"]),
        "brian": brian,
    }
    intact = run_figure10_search_cycle_spread_condition(
        reset_pathway_enabled=True, **common
    )
    control = run_figure10_search_cycle_spread_condition(
        reset_pathway_enabled=False, **common
    )

    pre_reset_winner = (
        intact.pre_layer4_events > 0
        and intact.pre_layer4_events == control.pre_layer4_events
        and intact.pre_layer4_active_indices == control.pre_layer4_active_indices
    )
    reset_chain = (
        intact.nonspecific_post_events > 0
        and intact.layer5_post_events > 0
        and intact.layer6i_post_events > 0
    )
    alternatives_quiet_before_release = (
        intact.alternative_events_before_release == 0
        and control.alternative_events_before_release == 0
    )
    input_delivery = all(
        value > 0
        for result in (intact, control)
        for _, value in result.projection035_post_release_integral_pA_ms
    )
    winner_suppression = intact.winner_post_events < control.winner_post_events
    alternative_latency = (
        intact.first_alternative_event_ms is not None
        and (
            control.first_alternative_event_ms is None
            or intact.first_alternative_event_ms
            < control.first_alternative_event_ms
        )
    )
    reset_gates = {
        "pre_reset_winner": pre_reset_winner,
        "reset_chain": reset_chain,
        "alternatives_quiet_before_release": alternatives_quiet_before_release,
        "full_input_delivery": input_delivery,
        "winner_suppression": winner_suppression,
        "alternative_latency": alternative_latency,
    }

    print(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "id": registration["result_id"],
                "date": registration["date"],
                "status": "completed-figure10-projection036-spread-pair",
                "classification": registration["classification"],
                "registration": args.registration,
                "runtime_fingerprint": conventions.fingerprint,
                "persistent_projection_weight_scales": scales,
                "persistent_projection_delays_ms": delays,
                "figure6_gates": figure6_gates,
                "intact": asdict(intact),
                "disconnected_control": asdict(control),
                "reset_gates": reset_gates,
                "reproduced_search_cycle": all(reset_gates.values()),
                "figure7_known_discrepancy_locked": True,
                "original_smart_reproduced": False,
                "baseline_promoted": False,
            },
            sort_keys=False,
        ),
        end="",
    )


if __name__ == "__main__":
    main()
