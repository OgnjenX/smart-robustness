"""Run the preregistered Figure 7 match for the recovered KInNeSS detector."""

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
from smart_robustness.validation.figure6 import Figure6LearningProtocol, run_figure6_learning
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


def _relay_times(result: Any) -> dict[str, list[float]]:
    indices = result.population_spike_indices["thalamic_relay"]
    times = result.population_spike_times_ms["thalamic_relay"]
    return {
        str(index): [
            time
            for source_index, time in zip(indices, times, strict=True)
            if source_index == index
        ]
        for index in (38, 39, 40, 41, 42)
    }


def _verify_training(result: Any, reference: dict[str, Any]) -> None:
    if result.convention_fingerprint != reference["runtime_fingerprint"]:
        raise ValueError("fresh Figure 6 runtime differs from Artifact 481")
    if _plain(result.population_spikes) != reference["population_spikes"]:
        raise ValueError("fresh Figure 6 population events differ from Artifact 481")
    if _relay_times(result) != reference["active_cell_times_ms"]["relay_horizontal"]:
        raise ValueError("fresh Figure 6 relay train differs from Artifact 481")
    expected_maps = reference["maps"]
    checks = {
        "bottom_up_horizontal_orientation_contrast": (
            result.bottom_up.horizontal_orientation_contrast
        ),
        "top_down_horizontal_orientation_contrast": (
            result.top_down_combined.horizontal_orientation_contrast
        ),
    }
    if checks != {
        key: expected_maps[key]
        for key in checks
    }:
        raise ValueError("fresh Figure 6 learned-map contrasts differ from Artifact 481")
    if max(result.top_down_combined.after) != expected_maps["top_down_combined"][
        "maximum_after"
    ]:
        raise ValueError("fresh Figure 6 learned-weight maximum differs from Artifact 481")


def _score_match(result: Any, required: dict[str, Any]) -> dict[str, bool]:
    expected = set(required["relay_active_indices"])
    counts = {
        index: result.relay_spike_indices.count(index)
        for index in expected
    }
    selected_events = [
        time
        for index, time in zip(
            result.category_spike_indices,
            result.category_spike_times_ms,
            strict=True,
        )
        if index == 40
    ]
    termination = result.top_down_current_termination_time_ms
    return {
        "relay_active_indices": set(result.relay_spike_indices) == expected,
        "minimum_relay_events_per_active_index": all(
            count >= int(required["minimum_relay_events_per_active_index"])
            for count in counts.values()
        ),
        "selected_category_event_available": bool(selected_events),
        "top_down_current_terminated_on_first_selected_event": (
            bool(selected_events) and termination == selected_events[0]
        ),
        "trn_events_present": bool(result.trn_spike_times_ms),
        "nonspecific_events": (
            len(result.nonspecific_spike_times_ms)
            == int(required["nonspecific_events"])
        ),
        "nonspecific_rate_hz": (
            result.nonspecific_rate_hz == float(required["nonspecific_rate_hz"])
        ),
        "figure7_target_duration": result.duration_ms == 100.0,
        "no_reconstructed_comparator": (
            result.comparator_transform in (None, "none")
            and result.comparator_relay_floor is None
            and result.comparator_target_count is None
        ),
        "no_calcium_ablation": not result.relay_calcium_ablated_at_stimulus,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)

    registration = yaml.safe_load(Path(args.registration).read_text())
    profile = yaml.safe_load(Path(registration["profile"]).read_text())
    training_profile = yaml.safe_load(Path(profile["training_profile"]).read_text())
    training_reference = yaml.safe_load(Path(profile["training_result"]).read_text())
    training_assessment = yaml.safe_load(Path(profile["training_assessment"]).read_text())
    if not training_assessment["assessment"]["preregistered_figure6_contract_passed"]:
        raise ValueError("Figure 6 assessment does not authorize recognition")

    base = yaml.safe_load(Path(training_profile["base_profile"]).read_text())
    conventions = replace(
        runtime_conventions_for_candidate(base["candidate"]),
        **training_profile["runtime_overrides"],
    )
    if conventions.fingerprint != registration["runtime_fingerprint"]:
        raise ValueError("registered runtime differs from executable profile")
    scales = {
        str(key): float(value)
        for key, value in training_profile["projection_weight_scales"].items()
    }

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    training = run_figure6_learning(
        conventions=conventions,
        protocol=Figure6LearningProtocol(
            monitored_populations=tuple(training_profile["monitored_populations"])
        ),
        projection_weight_scales=scales,
        brian=brian,
    )
    _verify_training(training.result, training_reference)

    protocol = profile["protocol"]
    result = run_figure7_condition(
        condition=MatchCondition.MATCH,
        learned_weights=training.learned_weights,
        conventions=conventions,
        persistent_projection_weight_scales=scales,
        top_down_current_pA=float(protocol["top_down_current_pA"]),
        top_down_current_mode=TopDownCurrentMode(protocol["top_down_current_mode"]),
        top_down_cue_lead_ms=float(protocol["top_down_cue_lead_ms"]),
        duration_ms=float(protocol["duration_ms"]),
        dt_ms=float(protocol["dt_ms"]),
        equilibration_ms=float(protocol["equilibration_ms"]),
        record_relay_diagnostics=True,
        record_interneuron_spikes=True,
        brian=brian,
    )
    gates = _score_match(result, profile["required_gates"])
    artifact = {
        "schema_version": 1,
        "registration": args.registration,
        "runtime_fingerprint": conventions.fingerprint,
        "training_repeat_verified": True,
        "training_reference": profile["training_result"],
        "weight_handoff": profile["weight_handoff"],
        "projection_weight_scales": scales,
        "match_result": result,
        "gates": gates,
        "match_prerequisites_pass": all(gates.values()),
        "mismatch_run_authorized": all(gates.values()),
        "original_smart_reproduced": False,
        "baseline_promoted": False,
        "interpretation_boundary": profile["boundary"],
    }
    with output.open("x") as stream:
        yaml.safe_dump(_plain(artifact), stream, sort_keys=False)
    print(
        f"match relay={len(result.relay_spike_indices)} "
        f"active={sorted(set(result.relay_spike_indices))} "
        f"trn={len(result.trn_spike_indices)} "
        f"nonspecific={len(result.nonspecific_spike_times_ms)} "
        f"pass={all(gates.values())}",
        flush=True,
    )


if __name__ == "__main__":
    main()
