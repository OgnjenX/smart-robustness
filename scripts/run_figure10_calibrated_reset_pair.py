"""Run calibrated SMART Figure 10 intact and reset-pathway control episodes."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path

import yaml

from smart_robustness.validation.calibration import runtime_conventions_for_candidate
from smart_robustness.validation.figure6 import (
    Figure6LearningProtocol,
    assess_figure6_cortical_recruitment,
    assess_figure6_top_down_timing,
    run_figure6_learning,
)
from smart_robustness.validation.figure10 import (
    assess_figure10_reset,
    run_figure10_condition,
)


def _condition_summary(result):
    split = result.pre_match_duration_ms

    def phase_count(values, *, post):
        return sum((value >= split) is post for value in values)

    pre_layer4 = result.layer4_counts(after_mismatch=False)
    post_layer4 = result.layer4_counts(after_mismatch=True)
    layer4i_post_events = [
        [int(index), float(time_ms)]
        for index, time_ms in zip(
            result.layer4_inhibitory_spike_indices,
            result.layer4_inhibitory_spike_times_ms,
            strict=True,
        )
        if time_ms >= split
    ]
    return {
        "reset_pathway_enabled": result.reset_pathway_enabled,
        "layer4_pre_events": int(pre_layer4.sum()),
        "layer4_pre_active_indices": [int(index) for index in pre_layer4.nonzero()[0]],
        "layer4_post_events": int(post_layer4.sum()),
        "layer4_post_active_indices": [int(index) for index in post_layer4.nonzero()[0]],
        "nonspecific_pre_events": phase_count(result.nonspecific_spike_times_ms, post=False),
        "nonspecific_post_events": phase_count(result.nonspecific_spike_times_ms, post=True),
        "layer5_pre_events": phase_count(result.layer5_spike_times_ms, post=False),
        "layer5_post_events": phase_count(result.layer5_spike_times_ms, post=True),
        "layer6i_pre_events": phase_count(result.layer6i_spike_times_ms, post=False),
        "layer6i_post_events": phase_count(result.layer6i_spike_times_ms, post=True),
        "layer6i_mismatch_events": [
            [int(index), float(time_ms)]
            for index, time_ms in zip(
                result.layer6i_spike_indices,
                result.layer6i_spike_times_ms,
                strict=True,
            )
            if time_ms >= split
        ],
        "layer6i_mismatch_event_transmitter_samples": [
            [int(index), float(time_ms), float(transmitter)]
            for index, time_ms, transmitter in (result.layer6i_mismatch_event_transmitter_samples)
        ],
        "layer4_inhibitory_post_events": len(layer4i_post_events),
        "layer4_inhibitory_post_active_indices": sorted(
            {index for index, _ in layer4i_post_events}
        ),
        "layer6i_mismatch_gate_integral_ms_by_projection": dict(
            result.layer6i_mismatch_gate_integral_ms_by_projection
        ),
        "layer6i_mismatch_current_integral_pA_ms_by_projection": dict(
            result.layer6i_mismatch_current_integral_pA_ms_by_projection
        ),
        "layer6i_mismatch_current_peak_pA_by_projection": dict(
            result.layer6i_mismatch_current_peak_pA_by_projection
        ),
        "layer4i_mismatch_projection026_gate_integral_ms": (
            result.layer4i_mismatch_projection026_gate_integral_ms
        ),
        "layer4i_mismatch_projection026_current_integral_pA_ms": (
            result.layer4i_mismatch_projection026_current_integral_pA_ms
        ),
        "layer4i_mismatch_projection026_current_peak_pA": (
            result.layer4i_mismatch_projection026_current_peak_pA
        ),
        "layer4e_mismatch_projection036_gate_integral_ms": (
            result.layer4e_mismatch_projection036_gate_integral_ms
        ),
        "layer4e_mismatch_projection036_current_integral_pA_ms": (
            result.layer4e_mismatch_projection036_current_integral_pA_ms
        ),
        "layer4e_mismatch_projection036_current_trough_pA": (
            result.layer4e_mismatch_projection036_current_trough_pA
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
    for item in registration["persistent_projection_scales"]:
        projection_id = str(item["projection_id"])
        if projection_id in scales:
            raise ValueError(f"{projection_id} overlaps an existing scale")
        scales[projection_id] = float(item["scale"])
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
        raise RuntimeError(f"fresh Figure 6 prerequisite failed: {figure6_gates}")

    protocol = registration["protocol"]
    common = {
        "top_down_current_pA": float(profile["protocol"]["top_down_current_pA"]),
        "pre_match_duration_ms": float(protocol["pre_match_duration_ms"]),
        "mismatch_duration_ms": float(protocol["mismatch_duration_ms"]),
        "learned_weights": training.learned_weights,
        "persistent_projection_weight_scales": scales,
        "comparator_top_k_targets": int(profile["comparator"]["target_count"]),
        "comparator_source_index": int(profile["comparator"]["source_index"]),
        "top_down_current_mode": profile["protocol"]["top_down_current_mode"],
        "conventions": conventions,
        "dt_ms": float(protocol["dt_ms"]),
        "record_layer6i_diagnostics": bool(registration.get("record_layer6i_diagnostics", False)),
        "record_reset_chain_diagnostics": bool(
            registration.get("record_reset_chain_diagnostics", False)
        ),
        "brian": brian,
    }
    intact = run_figure10_condition(reset_pathway_enabled=True, **common)
    control = run_figure10_condition(reset_pathway_enabled=False, **common)
    assessment = assess_figure10_reset(intact, control)

    print(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "id": registration["result_id"],
                "date": registration["date"],
                "status": "completed-calibrated-figure10-reset-pair",
                "classification": registration["classification"],
                "registration": args.registration,
                "runtime_fingerprint": conventions.fingerprint,
                "persistent_projection_weight_scales": scales,
                "figure6_gates": figure6_gates,
                "intact": _condition_summary(intact),
                "disconnected_control": _condition_summary(control),
                "reset_assessment": asdict(assessment),
                "reset_gates": {
                    "pre_reset_winner": assessment.pre_reset_winner_pass,
                    "reset_chain": assessment.reset_chain_pass,
                    "winner_suppression": assessment.winner_suppression_pass,
                    "alternative_release": assessment.alternative_release_pass,
                },
                "reproduced_reset": assessment.reproduced_reset,
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
