"""Compare nonspecific-cell traces for fixed six- and seven-event mismatches."""

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
from smart_robustness.validation.figure7 import TopDownCurrentMode, run_figure7_condition


def _finite_pairs(values):
    return [[float(item) for item in row] for row in values]


def _current_samples(values):
    return [
        [float(time_ms), float(voltage_mV), str(source), float(current_pA)]
        for time_ms, voltage_mV, source, current_pA in values
    ]


def _gate_samples(values):
    return [
        [float(time_ms), float(voltage_mV), str(source), float(gate)]
        for time_ms, voltage_mV, source, gate in values
    ]


def _mismatch_summary(result):
    return {
        "relay_event_count": len(result.relay_spike_times_ms),
        "relay_active_indices": sorted(set(result.relay_spike_indices)),
        "trn_event_count": len(result.trn_spike_times_ms),
        "nonspecific_event_count": len(result.nonspecific_spike_times_ms),
        "nonspecific_spike_times_ms": list(result.nonspecific_spike_times_ms),
        "nonspecific_trn_gaba_peak": result.nonspecific_trn_gaba_peak,
        "nonspecific_trn_gaba_integral_ms": result.nonspecific_trn_gaba_integral_ms,
        "nonspecific_post_startup_trn_gaba_peak": (result.nonspecific_post_startup_trn_gaba_peak),
        "nonspecific_direct_input_current_range_pA": (
            result.nonspecific_direct_input_current_range_pA
        ),
        "nonspecific_trn_current_range_pA": result.nonspecific_trn_current_range_pA,
        "nonspecific_layer6ii_current_range_pA": (result.nonspecific_layer6ii_current_range_pA),
        "nonspecific_voltage_range_mV_by_compartment": [
            [str(compartment), float(minimum), float(maximum)]
            for compartment, minimum, maximum in (
                result.nonspecific_voltage_range_mV_by_compartment
            )
        ],
        "nonspecific_positive_soma_local_maxima_ms_mV": _finite_pairs(
            result.nonspecific_positive_soma_local_maxima_ms_mV
        ),
        "nonspecific_positive_detector_local_maxima_ms_mV": _finite_pairs(
            result.nonspecific_positive_detector_local_maxima_ms_mV
        ),
        "nonspecific_peak_current_samples_pA": _current_samples(
            result.nonspecific_peak_current_samples_pA
        ),
        "nonspecific_peak_gate_samples": _gate_samples(result.nonspecific_peak_gate_samples),
        "nonspecific_detector_voltage_range_mV": (result.nonspecific_detector_voltage_range_mV),
        "nonspecific_detector_threshold_upcrossings": (
            result.nonspecific_detector_threshold_upcrossings
        ),
        "nonspecific_detector_zero_downcrossings": (result.nonspecific_detector_zero_downcrossings),
        "nonspecific_detector_arm_transitions": (result.nonspecific_detector_arm_transitions),
        "nonspecific_detector_release_transitions": (
            result.nonspecific_detector_release_transitions
        ),
        "nonspecific_detector_final_armed": result.nonspecific_detector_final_armed,
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
    base_scales = {
        str(key): float(value)
        for key, value in training_profile["projection_weight_scales"].items()
    }
    protocol = profile["protocol"]
    comparator = profile["comparator"]
    outcomes = []
    for arm in registration["arms"]:
        persistent_scales = dict(base_scales)
        for item in arm["persistent_projection_scales"]:
            projection_id = item["projection_id"]
            if projection_id in persistent_scales:
                raise ValueError(f"{projection_id} overlaps an existing scale")
            persistent_scales[projection_id] = float(item["scale"])
        training = run_figure6_learning(
            conventions=conventions,
            protocol=Figure6LearningProtocol(
                monitored_populations=tuple(training_profile["monitored_populations"])
            ),
            projection_weight_scales=persistent_scales,
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
            raise RuntimeError(
                f"fresh Figure 6 prerequisite failed for {arm['name']}: {figure6_gates}"
            )
        mismatch = run_figure7_condition(
            condition=MatchCondition.MISMATCH,
            learned_weights=training.learned_weights,
            conventions=conventions,
            persistent_projection_weight_scales=persistent_scales,
            top_down_current_pA=float(protocol["top_down_current_pA"]),
            top_down_current_mode=TopDownCurrentMode(protocol["top_down_current_mode"]),
            top_down_cue_lead_ms=float(protocol["top_down_cue_lead_ms"]),
            duration_ms=float(protocol["duration_ms"]),
            dt_ms=float(protocol["dt_ms"]),
            equilibration_ms=float(protocol["equilibration_ms"]),
            comparator_top_k_targets=int(comparator["target_count"]),
            comparator_source_index=int(comparator["source_index"]),
            record_relay_diagnostics=True,
            brian=brian,
        )
        outcomes.append(
            {
                "name": arm["name"],
                "persistent_projection_weight_scales": persistent_scales,
                "figure6_gates": figure6_gates,
                "mismatch": _mismatch_summary(mismatch),
            }
        )

    print(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "id": registration["result_id"],
                "date": registration["date"],
                "status": "completed-read-only-mismatch-nonspecific-trace-compare",
                "classification": registration["classification"],
                "registration": args.registration,
                "runtime_fingerprint": conventions.fingerprint,
                "outcomes": outcomes,
                "model_or_protocol_change": False,
                "instrumentation_change_only": True,
                "candidate_reopened": False,
                "original_smart_reproduced": False,
                "baseline_promoted": False,
            },
            sort_keys=False,
        ),
        end="",
    )


if __name__ == "__main__":
    main()
