"""Test whether scalar TRN-to-relay gain can express overlap-only mismatch."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from smart_robustness.protocols import MatchCondition
from smart_robustness.validation.calibration import runtime_conventions_for_candidate
from smart_robustness.validation.figure6 import run_figure6_learning
from smart_robustness.validation.figure7 import (
    TopDownCurrentMode,
    expand_figure7_source_expectation_toward_bounds,
    run_figure7_condition,
)


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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        default="configs/calibration/figure7_mismatch_gaba_capacity_v1.yaml",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    profile = yaml.safe_load(Path(args.profile).read_text())
    base_profile = yaml.safe_load(Path(profile["base_profile"]).read_text())
    base = runtime_conventions_for_candidate(base_profile["candidate"])
    detector = profile["detector"]
    conventions = replace(
        base,
        trn_spike_event_coordinate="absolute_physical",
        trn_spike_event_threshold_mV=float(detector["arm_mV"]),
        trn_spike_event_release_mV=float(detector["release_mV"]),
        trn_spike_event_proximal_blend_fraction=None,
        **profile["runtime_overrides"],
    )
    baseline_scales = {
        str(key): float(value)
        for key, value in profile["baseline_trn_to_relay_gaba"]["scales"].items()
    }
    training = run_figure6_learning(
        conventions=conventions,
        projection_weight_scales=baseline_scales,
        brian=brian,
    )
    if training.result.population_spikes["thalamic_relay"] != 20:
        raise ValueError("capacity diagnostic handoff did not reproduce Figure 6")
    learned, applied_factor = expand_figure7_source_expectation_toward_bounds(
        training.learned_weights,
        headroom_fraction=float(profile["learned_state"]["selected_headroom_fraction"]),
        source_index=int(profile["learned_state"]["source_index"]),
    )

    protocol = profile["protocol"]
    overlap = int(profile["readout"]["overlap_index"])
    outcomes: list[dict[str, Any]] = []
    for gain in profile["dimension"]["grid"]:
        scales = {
            projection_id: baseline * float(gain)
            for projection_id, baseline in baseline_scales.items()
        }
        result = run_figure7_condition(
            condition=MatchCondition.MISMATCH,
            top_down_current_pA=float(protocol["top_down_current_pA"]),
            learned_weights=learned,
            conventions=conventions,
            duration_ms=float(protocol["duration_ms"]),
            dt_ms=float(protocol["dt_ms"]),
            persistent_projection_weight_scales=scales,
            top_down_current_mode=TopDownCurrentMode(
                protocol["top_down_current_mode"]
            ),
            top_down_cue_lead_ms=float(protocol["top_down_cue_lead_ms"]),
            equilibration_ms=float(protocol["equilibration_ms"]),
            brian=brian,
        )
        active = sorted(set(result.relay_spike_indices))
        outcomes.append(
            {
                "common_gain": float(gain),
                "scales": scales,
                "relay_events": len(result.relay_spike_times_ms),
                "relay_active_indices": active,
                "relay_event_counts_by_index": {
                    index: result.relay_spike_indices.count(index) for index in active
                },
                "trn_events": len(result.trn_spike_times_ms),
                "nonspecific_events": len(result.nonspecific_spike_times_ms),
                "nonspecific_rate_hz": result.nonspecific_rate_hz,
                "overlap_only": active == [overlap],
                "relay_silent": not active,
                "result": result,
            }
        )
        print(
            f"gain={gain} relay={len(result.relay_spike_times_ms)} "
            f"active={active} trn={len(result.trn_spike_times_ms)} "
            f"ns={len(result.nonspecific_spike_times_ms)}",
            flush=True,
        )

    overlap_only_gains = [
        outcome["common_gain"] for outcome in outcomes if outcome["overlap_only"]
    ]
    artifact = {
        "schema_version": 1,
        "id": Path(args.output).stem,
        "date": datetime.now(tz=UTC).date().isoformat(),
        "status": "complete",
        "profile": args.profile,
        "registration_artifact": profile["registration_artifact"],
        "base_candidate_fingerprint": base_profile["candidate_fingerprint"],
        "runtime_fingerprint": conventions.fingerprint,
        "holdouts_consulted": ["figure7_mismatch"],
        "diagnostic_only": True,
        "handoff_figure6_population_spikes": training.result.population_spikes,
        "applied_common_weight_factor": applied_factor,
        "outcomes": outcomes,
        "overlap_only_gains": overlap_only_gains,
        "assessment": {
            "scalar_gain_has_overlap_only_window": bool(overlap_only_gains),
            "candidate_promoted": False,
            "downstream_holdouts_unlocked": False,
        },
        "next_gate": profile["next_gate"],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(_plain(artifact), sort_keys=False))


if __name__ == "__main__":
    main()

