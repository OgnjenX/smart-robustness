"""Replay intact Figure 7 input into an isolated nonspecific thalamic cell."""

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
from smart_robustness.validation.nonspecific_replay import run_nonspecific_replay


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


def _same_sequence(result: Any, control: dict[str, Any], prefix: str) -> bool:
    return (
        list(getattr(result, f"{prefix}_spike_indices"))
        == control[f"{prefix}_spike_indices"]
        and list(getattr(result, f"{prefix}_spike_times_ms"))
        == control[f"{prefix}_spike_times_ms"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True)
    parser.add_argument("--trace-output", required=True)
    args = parser.parse_args()

    registration = yaml.safe_load(Path(args.registration).read_text())
    profile = yaml.safe_load(Path(registration["profile"]).read_text())
    training_profile = yaml.safe_load(Path(profile["training_profile"]).read_text())
    training_reference = yaml.safe_load(Path(profile["training_result"]).read_text())
    control = yaml.safe_load(Path(registration["control_result"]).read_text())[
        "match_result"
    ]
    base_profile = yaml.safe_load(Path(training_profile["base_profile"]).read_text())
    conventions = replace(
        runtime_conventions_for_candidate(base_profile["candidate"]),
        **training_profile["runtime_overrides"],
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
    training_repeat = (
        training.result.population_spikes == training_reference["population_spikes"]
    )
    if not training_repeat:
        raise ValueError("fresh Figure 6 population events differ from reference")

    protocol = profile["protocol"]
    source = run_figure7_condition(
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
        nonspecific_replay_trace_output=args.trace_output,
        brian=brian,
    )
    source_identity = {
        prefix: _same_sequence(source, control, prefix)
        for prefix in ("relay", "trn", "category")
    }
    source_identity["nonspecific_times"] = (
        list(source.nonspecific_spike_times_ms)
        == control["nonspecific_spike_times_ms"]
    )
    source_repeat = all(source_identity.values())
    if not source_repeat:
        raise ValueError("connected trace source does not repeat artifact 520")

    intact = run_nonspecific_replay(
        args.trace_output, conventions=conventions, ablate_calcium=False, brian=brian
    )
    tolerance = float(
        registration["intact_replay_gate"]["maximum_voltage_error_mV"]
    )
    max_state_error = max(error for _, error in intact.max_abs_error_by_variable)
    intact_gate = (
        intact.exact_spike_train
        and intact.max_voltage_error_mV <= tolerance
        and max_state_error <= float(
            registration["intact_replay_gate"][
                "maximum_dimensionless_state_error"
            ]
        )
    )
    ablated = None
    if intact_gate:
        ablated = run_nonspecific_replay(
            args.trace_output,
            conventions=conventions,
            ablate_calcium=True,
            brian=brian,
        )

    late_boundary_ms = min(control["trn_spike_times_ms"])
    source_late = tuple(
        time for time in intact.source_spike_times_ms if time >= late_boundary_ms
    )
    ablated_late = () if ablated is None else tuple(
        time for time in ablated.replay_spike_times_ms if time >= late_boundary_ms
    )
    causal_support = bool(
        intact_gate and source_late and not ablated_late
    )
    artifact = {
        "schema_version": 1,
        "id": registration["result_id"],
        "date": registration["date"],
        "status": (
            "isolated-causal-support-not-candidate"
            if causal_support
            else "isolated-replay-inconclusive"
        ),
        "classification": registration["classification"],
        "registration": args.registration,
        "runtime_fingerprint": conventions.fingerprint,
        "fresh_figure6_repeat_verified": training_repeat,
        "connected_source_repeat": source_identity,
        "trace": {
            "path": source.nonspecific_replay_trace_path,
            "sha256": source.nonspecific_replay_trace_sha256,
            "source_event_count": len(source.nonspecific_spike_times_ms),
            "source_event_times_ms": source.nonspecific_spike_times_ms,
        },
        "intact_replay": intact,
        "intact_replay_max_state_error": max_state_error,
        "intact_replay_gate_pass": intact_gate,
        "calcium_ablated_replay": ablated,
        "late_event_boundary_ms": late_boundary_ms,
        "source_late_event_times_ms": source_late,
        "ablated_late_event_times_ms": ablated_late,
        "cell_autonomous_t_current_support": causal_support,
        "promotable": False,
        "original_smart_reproduced": False,
        "baseline_promoted": False,
        "interpretation_boundary": registration["classification_rule"],
    }
    print(yaml.safe_dump(_plain(artifact), sort_keys=False), end="")


if __name__ == "__main__":
    main()
