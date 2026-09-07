"""Verify the selected effective nonspecific T density in the connected match."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
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


def _plain(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return _plain(asdict(value))
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Enum):
        return value.value
    return value


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
    training = run_figure6_learning(
        conventions=conventions,
        protocol=Figure6LearningProtocol(
            monitored_populations=tuple(training_profile["monitored_populations"])
        ),
        projection_weight_scales=scales,
        brian=brian,
    )
    expected_indices = tuple(int(i) for i in profile["figure6_gates"]["relay_active_indices"])
    relay_indices = training.result.population_spike_indices["thalamic_relay"]
    relay_counts = {index: relay_indices.count(index) for index in expected_indices}
    recruitment = assess_figure6_cortical_recruitment(training.result)
    timing = assess_figure6_top_down_timing(training.result)
    figure6_gates = {
        "relay_active_indices": set(relay_indices) == set(expected_indices),
        "relay_events_per_active_index": set(relay_counts.values()) == {
            int(profile["figure6_gates"]["relay_events_per_active_index"])
        },
        "relay_events": len(relay_indices)
        == int(profile["figure6_gates"]["relay_events"]),
        "cortical_chain_complete": recruitment.feedforward_chain_complete,
        "causal_learning_pair": timing.causal_pair_in_learning_window,
        "top_down_horizontal_contrast": (
            training.result.top_down_combined.horizontal_orientation_contrast > 0
        ),
    }
    figure6_pass = all(figure6_gates.values())

    match = None
    match_gates = {}
    if figure6_pass:
        protocol = profile["protocol"]
        match = run_figure7_condition(
            condition=MatchCondition.MATCH,
            learned_weights=training.learned_weights,
            conventions=conventions,
            persistent_projection_weight_scales=scales,
            top_down_current_pA=float(protocol["top_down_current_pA"]),
            top_down_current_mode=TopDownCurrentMode(
                protocol["top_down_current_mode"]
            ),
            top_down_cue_lead_ms=float(protocol["top_down_cue_lead_ms"]),
            duration_ms=float(protocol["duration_ms"]),
            dt_ms=float(protocol["dt_ms"]),
            equilibration_ms=float(protocol["equilibration_ms"]),
            brian=brian,
        )
        source_events = tuple(
            time
            for index, time in zip(
                match.category_spike_indices,
                match.category_spike_times_ms,
                strict=True,
            )
            if index == 40
        )
        match_gates = {
            "relay_active_indices": set(match.relay_spike_indices)
            == set(profile["match_gates"]["relay_active_indices"]),
            "relay_events": len(match.relay_spike_times_ms)
            == int(profile["match_gates"]["relay_events"]),
            "selected_category_event_available": bool(source_events),
            "top_down_current_terminated_on_first_selected_event": (
                match.top_down_current_termination_time_ms is not None
            ),
            "nonspecific_events": len(match.nonspecific_spike_times_ms)
            == int(profile["match_gates"]["nonspecific_events"]),
            "nonspecific_rate_hz": match.nonspecific_rate_hz
            == float(profile["match_gates"]["nonspecific_rate_hz"]),
            "no_reconstructed_comparator": match.comparator_transform is None,
        }
    match_pass = bool(match_gates) and all(match_gates.values())
    artifact = {
        "schema_version": 1,
        "id": registration["result_id"],
        "date": registration["date"],
        "status": (
            "connected-match-pass" if match_pass else
            "connected-match-failed" if figure6_pass else "figure6-failed"
        ),
        "classification": registration["classification"],
        "registration": args.registration,
        "runtime_fingerprint": conventions.fingerprint,
        "nonspecific_dendritic_calcium_density_scale": (
            conventions.nonspecific_dendritic_calcium_density_scale
        ),
        "figure6_result": training.result,
        "figure6_gates": figure6_gates,
        "figure6_pass": figure6_pass,
        "match_result": match,
        "match_gates": match_gates,
        "match_pass": match_pass,
        "mismatch_consulted": False,
        "promotable": False,
        "original_smart_reproduced": False,
        "baseline_promoted": False,
        "classification_boundary": registration["classification_boundary"],
    }
    print(yaml.safe_dump(_plain(artifact), sort_keys=False), end="")


if __name__ == "__main__":
    main()
