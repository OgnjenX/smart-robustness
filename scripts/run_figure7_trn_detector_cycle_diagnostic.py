"""Localize why the calibrated Figure 7 TRN detector emits only one volley."""

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
from smart_robustness.validation.figure7 import run_figure7_condition


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        default="configs/calibration/figure7_trn_detector_cycle_diagnostic_v1.yaml",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    profile = yaml.safe_load(Path(args.profile).read_text())
    base_profile = yaml.safe_load(Path(profile["base_profile"]).read_text())
    base = runtime_conventions_for_candidate(base_profile["candidate"])
    fixed = profile["fixed_choices"]
    conventions = replace(
        base,
        trn_calcium_source_convention=str(
            fixed["trn_calcium_source_convention"]
        ),
        trn_dendritic_calcium_density_mS_cm2=float(
            fixed["trn_dendritic_calcium_density_mS_cm2"]
        ),
        trn_soma_proximal_axial_conductance_scale=float(
            fixed["trn_soma_proximal_axial_conductance_scale"]
        ),
        trn_spike_event_proximal_blend_fraction=float(
            fixed["trn_spike_event_proximal_blend_fraction"]
        ),
    )
    protocol = profile["protocol"]
    learned_state = str(fixed["learned_state"])
    if learned_state not in {
        "paper_constrained_figure6c_reference",
        "same_network_figure6_episode",
    }:
        raise ValueError(f"unsupported learned-state mode: {learned_state}")
    result = run_figure7_condition(
        condition=MatchCondition(str(fixed["condition"])),
        top_down_current_pA=float(fixed["top_down_current_pA"]),
        use_paper_constrained_reference=(
            learned_state == "paper_constrained_figure6c_reference"
        ),
        pretrain_with_figure6_episode=(
            learned_state == "same_network_figure6_episode"
        ),
        conventions=conventions,
        duration_ms=float(protocol["duration_ms"]),
        dt_ms=float(protocol["dt_ms"]),
        top_down_cue_lead_ms=float(protocol["top_down_cue_lead_ms"]),
        equilibration_ms=float(protocol["equilibration_ms"]),
        record_relay_diagnostics=True,
        brian=brian,
    )
    upcrossings = _pairs(result.trn_detector_threshold_upcrossings_by_index)
    zero_downcrossings = _pairs(result.trn_detector_zero_downcrossings_by_index)
    arm_transitions = _pairs(result.trn_detector_arm_transitions_by_index)
    release_transitions = _pairs(result.trn_detector_release_transitions_by_index)
    emitted_events = {
        index: result.trn_spike_indices.count(index)
        for index in protocol["diagnostic_indices"]
    }
    pre_stimulus_latched_release = all(
        upcrossings[index] == 0
        and release_transitions[index] == emitted_events[index]
        and emitted_events[index] > 0
        for index in upcrossings
    )
    if (
        learned_state == "same_network_figure6_episode"
        and all(count == 0 for count in emitted_events.values())
        and all(count == 0 for count in upcrossings.values())
    ):
        interpretation = "unified_candidate_has_no_evoked_trn_output"
    elif pre_stimulus_latched_release:
        interpretation = (
            "pre_training_latched_arm_survives_figure6_episode"
            if learned_state == "same_network_figure6_episode"
            else "pre_stimulus_latched_arm_released_by_stimulus"
        )
    elif all(count <= 1 for count in upcrossings.values()):
        interpretation = "no_post_event_regenerative_threshold_recrossing"
    elif any(
        upcrossings[index] > arm_transitions[index] for index in upcrossings
    ):
        interpretation = "detector_failed_to_rearm_despite_voltage_recrossing"
    elif any(
        arm_transitions[index] > release_transitions[index]
        for index in arm_transitions
    ):
        interpretation = "detector_rearmed_but_failed_to_return_below_release"
    elif any(
        release_transitions[index] > emitted_events[index]
        for index in release_transitions
    ):
        interpretation = "event_monitor_or_scheduling_inconsistency"
    else:
        interpretation = (
            "detector_cycles_and_upstream_event_timing_requires_localization"
        )

    artifact = {
        "schema_version": 1,
        "id": Path(args.output).stem,
        "date": datetime.now(tz=UTC).date().isoformat(),
        "status": "diagnostic-complete",
        "profile": args.profile,
        "candidate_fingerprint": base_profile["candidate_fingerprint"],
        "runtime_fingerprint": conventions.fingerprint,
        "protocol": protocol,
        "observed": {
            "relay_events": len(result.relay_spike_times_ms),
            "active_relay_indices": sorted(set(result.relay_spike_indices)),
            "trn_events": len(result.trn_spike_times_ms),
            "active_trn_cells": len(set(result.trn_spike_indices)),
            "nonspecific_events": len(result.nonspecific_spike_times_ms),
            "threshold_upcrossings_by_index": upcrossings,
            "zero_downcrossings_by_index": zero_downcrossings,
            "arm_transitions_by_index": arm_transitions,
            "release_transitions_by_index": release_transitions,
            "emitted_events_by_index": emitted_events,
            "pre_stimulus_latched_release_inferred": pre_stimulus_latched_release,
            "detector_voltage_range_mV_by_index": (
                result.trn_detector_voltage_range_mV_by_index
            ),
            "post_first_event_detector_voltage_range_mV_by_index": (
                result.trn_detector_post_first_event_voltage_range_mV_by_index
            ),
            "final_armed_by_index": result.trn_detector_final_armed_by_index,
        },
        "interpretation": interpretation,
        "result": result,
        "next_action": (
            "Use the registered interpretation to select an equation/source audit; "
            "do not tune an event threshold from this diagnostic."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(_plain(artifact), sort_keys=False))
    print(
        f"relay={len(result.relay_spike_times_ms)} "
        f"trn={len(result.trn_spike_times_ms)} "
        f"nonspecific={len(result.nonspecific_spike_times_ms)} "
        f"interpretation={interpretation}",
        flush=True,
    )


if __name__ == "__main__":
    main()
