"""Capture and verify an isolated Figure 10 peripheral layer-6I replay."""

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
from smart_robustness.validation.figure10 import run_figure10_condition
from smart_robustness.validation.layer6i_replay import run_layer6i_replay


def _phase_count(values, split_ms: float, *, post: bool) -> int:
    return sum((value >= split_ms) is post for value in values)


def _mismatch_count(values, start_ms: float, end_ms: float | None = None) -> int:
    return sum(value >= start_ms and (end_ms is None or value < end_ms) for value in values)


def _source_identity(result) -> dict[str, object]:
    split = result.pre_match_duration_ms
    late = split + 100.0
    return {
        "layer4_pre_post_events": [
            int(result.layer4_counts(after_mismatch=False).sum()),
            int(result.layer4_counts(after_mismatch=True).sum()),
        ],
        "nonspecific_pre_post_events": [
            _phase_count(result.nonspecific_spike_times_ms, split, post=False),
            _phase_count(result.nonspecific_spike_times_ms, split, post=True),
        ],
        "layer5_pre_post_events": [
            _phase_count(result.layer5_spike_times_ms, split, post=False),
            _phase_count(result.layer5_spike_times_ms, split, post=True),
        ],
        "layer5_first100_late_mismatch_events": [
            _mismatch_count(result.layer5_spike_times_ms, split, late),
            _mismatch_count(result.layer5_spike_times_ms, late),
        ],
        "layer6i_pre_post_events": [
            _phase_count(result.layer6i_spike_times_ms, split, post=False),
            _phase_count(result.layer6i_spike_times_ms, split, post=True),
        ],
        "layer6i_first100_late_mismatch_events": [
            _mismatch_count(result.layer6i_spike_times_ms, split, late),
            _mismatch_count(result.layer6i_spike_times_ms, late),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True)
    parser.add_argument("--trace-output", required=True)
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
    cell_index = int(registration["trace"]["cell_index"])
    source = run_figure10_condition(
        top_down_current_pA=float(profile["protocol"]["top_down_current_pA"]),
        pre_match_duration_ms=float(protocol["pre_match_duration_ms"]),
        mismatch_duration_ms=float(protocol["mismatch_duration_ms"]),
        reset_pathway_enabled=True,
        learned_weights=training.learned_weights,
        persistent_projection_weight_scales=scales,
        comparator_top_k_targets=int(profile["comparator"]["target_count"]),
        comparator_source_index=int(profile["comparator"]["source_index"]),
        top_down_current_mode=profile["protocol"]["top_down_current_mode"],
        conventions=conventions,
        dt_ms=float(protocol["dt_ms"]),
        layer6i_replay_trace_output=args.trace_output,
        layer6i_replay_cell_index=cell_index,
        brian=brian,
    )
    identity = _source_identity(source)
    expected_identity = registration["required_connected_source_identity"]
    source_identity_pass = identity == expected_identity
    if not source_identity_pass:
        raise RuntimeError("connected layer-6I replay source changed from Artifact 634")

    source_cell_events = [
        float(time_ms - source.pre_match_duration_ms)
        for index, time_ms in zip(
            source.layer6i_spike_indices,
            source.layer6i_spike_times_ms,
            strict=True,
        )
        if index == cell_index and time_ms >= source.pre_match_duration_ms
    ]
    replay = run_layer6i_replay(
        args.trace_output,
        conventions=conventions,
        brian=brian,
    )
    max_state_error = max(error for _, error in replay.max_abs_error_by_variable)
    gate = registration["intact_replay_gate"]
    replay_gate_pass = (
        replay.exact_spike_train
        and replay.max_voltage_error_mV <= float(gate["maximum_voltage_error_mV"])
        and max_state_error <= float(gate["maximum_dimensionless_state_error"])
    )
    artifact = {
        "schema_version": 1,
        "id": registration["result_id"],
        "date": registration["date"],
        "status": (
            "lossless-layer6i-replay-verified"
            if replay_gate_pass
            else "layer6i-replay-identity-failed"
        ),
        "classification": registration["classification"],
        "registration": args.registration,
        "runtime_fingerprint": conventions.fingerprint,
        "fresh_figure6_gates": figure6_gates,
        "connected_source_identity": identity,
        "connected_source_identity_pass": source_identity_pass,
        "trace": {
            "path": source.layer6i_replay_trace_path,
            "sha256": source.layer6i_replay_trace_sha256,
            "cell_index": cell_index,
            "source_event_times_from_mismatch_ms": source_cell_events,
        },
        "intact_replay": asdict(replay),
        "intact_replay_max_state_error": max_state_error,
        "intact_replay_gate_pass": replay_gate_pass,
        "parameter_search_performed": False,
        "original_smart_reproduced": False,
        "baseline_promoted": False,
        "interpretation_boundary": registration["boundary"],
    }
    print(yaml.safe_dump(artifact, sort_keys=False), end="")


if __name__ == "__main__":
    main()
