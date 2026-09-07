"""Localize excess nonspecific events without fitting the SMART baseline."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from smart_robustness.protocols import MatchCondition
from smart_robustness.validation.calibration import runtime_conventions_for_candidate
from smart_robustness.validation.figure6 import Figure6LearningProtocol, run_figure6_learning
from smart_robustness.validation.figure7 import TopDownCurrentMode, run_figure7_condition


def _summary(result: Any) -> dict[str, Any]:
    first_upstream = min(
        values[0]
        for values in (
            result.trn_spike_times_ms,
            result.relay_spike_times_ms,
            result.category_spike_times_ms,
        )
        if values
    )
    early = [
        float(value)
        for value in result.nonspecific_spike_times_ms
        if value < first_upstream
    ]
    return {
        "nonspecific_event_count": len(result.nonspecific_spike_times_ms),
        "nonspecific_rate_hz": result.nonspecific_rate_hz,
        "nonspecific_event_times_ms": list(result.nonspecific_spike_times_ms),
        "pre_upstream_nonspecific_event_times_ms": early,
        "pre_upstream_nonspecific_event_count": len(early),
        "post_upstream_nonspecific_event_count": (
            len(result.nonspecific_spike_times_ms) - len(early)
        ),
        "first_upstream_event_ms": float(first_upstream),
        "relay_event_count": len(result.relay_spike_times_ms),
        "relay_active_indices": sorted(set(result.relay_spike_indices)),
        "trn_event_count": len(result.trn_spike_times_ms),
        "category_event_count": len(result.category_spike_times_ms),
        "top_down_current_termination_time_ms": (
            result.top_down_current_termination_time_ms
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True)
    args = parser.parse_args()

    registration = yaml.safe_load(Path(args.registration).read_text())
    profile = yaml.safe_load(Path(registration["profile"]).read_text())
    training_profile = yaml.safe_load(Path(profile["training_profile"]).read_text())
    training_reference = yaml.safe_load(Path(profile["training_result"]).read_text())
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
    if training.result.convention_fingerprint != registration["runtime_fingerprint"]:
        raise ValueError("fresh Figure 6 runtime mismatch")
    if training.result.population_spikes != training_reference["population_spikes"]:
        raise ValueError("fresh Figure 6 population events differ from reference")

    protocol = profile["protocol"]
    outcomes: list[dict[str, Any]] = []
    for arm in registration["ablation_arms"]:
        result = run_figure7_condition(
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
            disabled_projection_ids=tuple(arm["disabled_projection_ids"]),
            brian=brian,
        )
        summary = _summary(result)
        outcomes.append(
            {
                "label": arm["label"],
                "disabled_projection_ids": arm["disabled_projection_ids"],
                "summary": summary,
            }
        )
        print(
            f"{arm['label']}: nonspecific="
            f"{summary['nonspecific_event_count']} "
            f"early/late={summary['pre_upstream_nonspecific_event_count']}/"
            f"{summary['post_upstream_nonspecific_event_count']}",
            flush=True,
        )

    control = registration["imported_control"]
    artifact = {
        "schema_version": 1,
        "id": registration["result_id"],
        "date": registration["date"],
        "status": "causal-localization-complete-not-candidate",
        "classification": "source-bundle-causal-ablation",
        "registration": args.registration,
        "runtime_fingerprint": conventions.fingerprint,
        "fresh_figure6_repeat_verified": True,
        "projection_weight_scales": scales,
        "imported_intact_control": control,
        "outcomes": outcomes,
        "promotable": False,
        "original_smart_reproduced": False,
        "baseline_promoted": False,
        "interpretation_boundary": registration["interpretation_boundary"],
    }
    print("---RESULT---", flush=True)
    print(yaml.safe_dump(artifact, sort_keys=False), end="")


if __name__ == "__main__":
    main()
