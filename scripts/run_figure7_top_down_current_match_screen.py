"""Screen the remaining registered Figure 7 top-down currents on clean match."""

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


def _pairs(values: tuple[tuple[int, Any], ...]) -> dict[int, Any]:
    return {int(index): value for index, value in values}


def _write(
    output: Path,
    *,
    profile_path: str,
    profile: dict[str, Any],
    base_profile: dict[str, Any],
    runtime_fingerprint: str,
    training_spikes: dict[str, int],
    outcomes: list[dict[str, Any]],
    complete: bool,
) -> None:
    survivors = [item["top_down_current_pA"] for item in outcomes if item["pass"]]
    artifact = {
        "schema_version": 1,
        "id": output.stem,
        "date": datetime.now(tz=UTC).date().isoformat(),
        "status": "complete" if complete else "running",
        "profile": profile_path,
        "registration_artifact": profile["registration_artifact"],
        "base_candidate_fingerprint": base_profile["candidate_fingerprint"],
        "runtime_fingerprint": runtime_fingerprint,
        "figure6_artifact": profile["figure6_artifact"],
        "holdouts_consulted": ["figure7_match"],
        "mismatch_consulted": False,
        "learned_state_handoff": profile["protocol"]["learned_state_handoff"],
        "handoff_figure6_population_spikes": training_spikes,
        "outcomes": outcomes,
        "survivor_currents_pA": survivors,
        "assessment": {
            "registered_candidate_count": len(
                profile["protocol"]["top_down_currents_pA"]
            ),
            "completed_candidate_count": len(outcomes),
            "exact_match_survivor_count": len(survivors),
            "advance_to_mismatch": bool(complete and survivors),
        },
        "next_gate": profile["next_gate"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(_plain(artifact), sort_keys=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        default="configs/calibration/figure7_top_down_current_match_reopen_v1.yaml",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    profile = yaml.safe_load(Path(args.profile).read_text())
    base_profile = yaml.safe_load(Path(profile["base_profile"]).read_text())
    figure6_artifact = yaml.safe_load(Path(profile["figure6_artifact"]).read_text())
    if not figure6_artifact["assessment"]["advance_to_figure7"]:
        raise ValueError("registered Figure 6 artifact does not authorize Figure 7")

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
    scales = {
        str(key): float(value)
        for key, value in profile["trn_to_relay_gaba"]["scales"].items()
    }
    training = run_figure6_learning(
        conventions=conventions,
        projection_weight_scales=scales,
        brian=brian,
    )
    if training.result.population_spikes["thalamic_relay"] != 20:
        raise ValueError("fresh handoff did not reproduce the Figure 6 relay train")

    protocol = profile["protocol"]
    current_mode = TopDownCurrentMode(
        protocol.get("top_down_current_mode", TopDownCurrentMode.SUSTAINED_EPOCH)
    )
    relay_source_indices = protocol.get("top_down_relay_source_indices")
    relay_source_indices = (
        None
        if relay_source_indices is None
        else frozenset(int(index) for index in relay_source_indices)
    )
    gate = profile["match_gate"]
    expected = tuple(int(index) for index in gate["relay_active_indices"])
    output = Path(args.output)
    outcomes: list[dict[str, Any]] = []
    for current_pA in protocol["top_down_currents_pA"]:
        result = run_figure7_condition(
            condition=MatchCondition.MATCH,
            top_down_current_pA=float(current_pA),
            learned_weights=training.learned_weights,
            conventions=conventions,
            duration_ms=float(protocol["duration_ms"]),
            dt_ms=float(protocol["dt_ms"]),
            record_relay_diagnostics=bool(protocol["record_relay_diagnostics"]),
            persistent_projection_weight_scales=scales,
            top_down_relay_source_indices=relay_source_indices,
            top_down_current_mode=current_mode,
            top_down_cue_lead_ms=float(protocol["top_down_cue_lead_ms"]),
            equilibration_ms=float(protocol["equilibration_ms"]),
            brian=brian,
        )
        relay_counts = {
            index: result.relay_spike_indices.count(index) for index in expected
        }
        event_counts = {
            index: result.trn_spike_indices.count(index)
            for index, _ in result.trn_detector_arm_transitions_by_index
        }
        upcrossings = _pairs(result.trn_detector_threshold_upcrossings_by_index)
        arms = _pairs(result.trn_detector_arm_transitions_by_index)
        releases = _pairs(result.trn_detector_release_transitions_by_index)
        fresh_cycles = bool(event_counts) and any(event_counts.values()) and all(
            event_counts[index]
            == upcrossings[index]
            == arms[index]
            == releases[index]
            for index in event_counts
        )
        first_cued_event_ms = next(
            (
                time
                for index, time in zip(
                    result.category_spike_indices,
                    result.category_spike_times_ms,
                    strict=True,
                )
                if index == 40
            ),
            None,
        )
        gates = {
            "top_down_relay_source_protocol": (
                result.top_down_relay_source_indices
                == (
                    None
                    if relay_source_indices is None
                    else tuple(sorted(relay_source_indices))
                )
            ),
            "top_down_current_protocol": (
                current_mode is TopDownCurrentMode.SUSTAINED_EPOCH
                or (
                    first_cued_event_ms is not None
                    and result.top_down_current_termination_time_ms
                    == first_cued_event_ms
                )
            ),
            "relay_active_indices": set(result.relay_spike_indices) == set(expected),
            "minimum_relay_events_per_active_index": all(
                count >= int(gate["minimum_relay_events_per_active_index"])
                for count in relay_counts.values()
            ),
            "trn_events": bool(result.trn_spike_times_ms),
            "sampled_trn_events_have_fresh_cycles": fresh_cycles,
            "figure7_target_duration": result.duration_ms == 100.0,
            "nonspecific_events": (
                len(result.nonspecific_spike_times_ms)
                == int(gate["nonspecific_events"])
            ),
            "nonspecific_40_hz": result.nonspecific_rate_hz
            == float(gate["nonspecific_rate_hz"]),
        }
        outcomes.append(
            {
                "top_down_current_pA": float(current_pA),
                "result": result,
                "relay_event_counts_by_index": relay_counts,
                "sampled_trn_event_counts_by_index": event_counts,
                "sampled_trn_threshold_upcrossings_by_index": upcrossings,
                "sampled_trn_arm_transitions_by_index": arms,
                "sampled_trn_release_transitions_by_index": releases,
                "gates": gates,
                "pass": all(gates.values()),
            }
        )
        _write(
            output,
            profile_path=args.profile,
            profile=profile,
            base_profile=base_profile,
            runtime_fingerprint=conventions.fingerprint,
            training_spikes=training.result.population_spikes,
            outcomes=outcomes,
            complete=False,
        )
        print(
            f"current={float(current_pA):g} pA "
            f"relay={len(result.relay_spike_times_ms)} "
            f"trn={len(result.trn_spike_times_ms)} "
            f"nonspecific={len(result.nonspecific_spike_times_ms)} "
            f"pass={all(gates.values())}",
            flush=True,
        )
    _write(
        output,
        profile_path=args.profile,
        profile=profile,
        base_profile=base_profile,
        runtime_fingerprint=conventions.fingerprint,
        training_spikes=training.result.population_spikes,
        outcomes=outcomes,
        complete=True,
    )


if __name__ == "__main__":
    main()
