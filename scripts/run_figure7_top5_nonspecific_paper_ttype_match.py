"""Test paper T-type kinetics in nonspecific thalamus through Figure 7 match."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from datetime import UTC, datetime
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
from smart_robustness.validation.figure7 import (
    TopDownCurrentMode,
    expand_figure7_source_expectation_toward_bounds,
    run_figure7_condition,
)

FIGURE6_MONITORED_POPULATIONS = (
    "thalamic_relay",
    "layer4_excitatory_v1",
    "layer23_excitatory_v1",
    "layer5_excitatory_v1",
    "layer6i_excitatory_v1",
    "layer6ii_excitatory_v1",
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
        default=(
            "configs/calibration/"
            "figure7_top5_nonspecific_paper_ttype_match_v1.yaml"
        ),
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
    runtime_expectations = profile.get(
        "runtime_expectations",
        {"nonspecific_calcium_kinetics_convention": "paper_2008"},
    )
    for field, expected in runtime_expectations.items():
        if getattr(conventions, field) != expected:
            raise ValueError(f"runtime field {field!r} does not match registration")

    scales = {
        str(key): float(value)
        for key, value in profile["trn_to_relay_gaba"]["scales"].items()
    }
    figure6_gate = profile["figure6_gate"]
    protocol = profile["protocol"]
    convergent_source_scope = protocol.get(
        "convergent_external_source_scope", "nonzero_pixels"
    )
    training = run_figure6_learning(
        conventions=conventions,
        protocol=Figure6LearningProtocol(
            monitored_populations=FIGURE6_MONITORED_POPULATIONS
        ),
        projection_weight_scales=scales,
        convergent_external_source_scope=convergent_source_scope,
        brian=brian,
    )
    training_result = training.result
    training_indices = training_result.population_spike_indices["thalamic_relay"]
    training_counts = {
        index: training_indices.count(index)
        for index in figure6_gate["relay_active_indices"]
    }
    recruitment = assess_figure6_cortical_recruitment(training_result)
    timing = assess_figure6_top_down_timing(training_result)
    figure6_gates = {
        "relay_active_indices": set(training_indices)
        == set(figure6_gate["relay_active_indices"]),
        "relay_events_per_active_index": set(training_counts.values())
        == {int(figure6_gate["relay_events_per_active_index"])},
        "relay_events": len(training_indices) == int(figure6_gate["relay_events"]),
        "cortical_chain_complete": recruitment.feedforward_chain_complete,
        "causal_learning_pair": timing.causal_pair_in_learning_window,
        "top_down_horizontal_contrast": (
            training_result.top_down_combined.horizontal_orientation_contrast > 0
        ),
    }
    figure6_pass = all(figure6_gates.values())

    match = None
    match_gates: dict[str, bool] = {}
    applied_factor = None
    if figure6_pass:
        source_index = int(profile["learned_state"]["source_index"])
        learned, applied_factor = expand_figure7_source_expectation_toward_bounds(
            training.learned_weights,
            headroom_fraction=float(
                profile["learned_state"]["headroom_fraction"]
            ),
            source_index=source_index,
        )
        match = run_figure7_condition(
            condition=MatchCondition.MATCH,
            top_down_current_pA=float(protocol["top_down_current_pA"]),
            learned_weights=learned,
            conventions=conventions,
            duration_ms=float(protocol["duration_ms"]),
            dt_ms=float(protocol["dt_ms"]),
            record_relay_diagnostics=bool(protocol["record_relay_diagnostics"]),
            persistent_projection_weight_scales=scales,
            comparator_top_k_targets=int(profile["comparator"]["target_count"]),
            comparator_source_index=source_index,
            top_down_current_mode=TopDownCurrentMode(
                protocol["top_down_current_mode"]
            ),
            top_down_cue_lead_ms=float(protocol["top_down_cue_lead_ms"]),
            equilibration_ms=float(protocol["equilibration_ms"]),
            convergent_external_source_scope=convergent_source_scope,
            brian=brian,
        )
        match_gate = profile["match_gate"]
        expected = tuple(int(index) for index in match_gate["relay_active_indices"])
        relay_counts = {
            index: match.relay_spike_indices.count(index) for index in expected
        }
        sampled = tuple(
            index for index, _ in match.trn_detector_arm_transitions_by_index
        )
        event_counts = {
            index: match.trn_spike_indices.count(index) for index in sampled
        }
        upcrossings = dict(match.trn_detector_threshold_upcrossings_by_index)
        arms = dict(match.trn_detector_arm_transitions_by_index)
        releases = dict(match.trn_detector_release_transitions_by_index)
        fresh_cycles = bool(event_counts) and any(event_counts.values()) and all(
            event_counts[index]
            == upcrossings[index]
            == arms[index]
            == releases[index]
            for index in event_counts
        )
        source_events = tuple(
            time
            for index, time in zip(
                match.cue_lead_category_spike_indices,
                match.cue_lead_category_spike_times_ms,
                strict=True,
            )
            if index == source_index
        )
        match_gates = {
            "one_selected_category_event_during_lead": len(source_events) == 1,
            "no_off_source_category_events_during_lead": all(
                index == source_index
                for index in match.cue_lead_category_spike_indices
            ),
            "no_relay_events_during_lead": not match.cue_lead_relay_spike_times_ms,
            "relay_active_indices": set(match.relay_spike_indices) == set(expected),
            "minimum_relay_events_per_active_index": all(
                count >= int(match_gate["minimum_relay_events_per_active_index"])
                for count in relay_counts.values()
            ),
            "trn_events": bool(match.trn_spike_times_ms),
            "nonspecific_events": len(match.nonspecific_spike_times_ms)
            == int(match_gate["nonspecific_events"]),
            "nonspecific_40_hz": match.nonspecific_rate_hz
            == float(match_gate["nonspecific_rate_hz"]),
            "figure7_target_duration": match.duration_ms == 100.0,
            "sampled_trn_events_have_fresh_cycles": fresh_cycles,
        }

    match_pass = bool(match_gates) and all(match_gates.values())
    status = (
        "match-pass"
        if match_pass
        else "match-failed" if figure6_pass else "figure6-failed"
    )
    artifact = {
        "schema_version": 1,
        "id": Path(args.output).stem,
        "date": datetime.now(tz=UTC).date().isoformat(),
        "status": status,
        "classification": "calibrated-reconstruction-source-sensitivity-not-baseline",
        "profile": args.profile,
        "registration_artifact": profile["registration_artifact"],
        "base_candidate_fingerprint": base_profile["candidate_fingerprint"],
        "runtime_fingerprint": conventions.fingerprint,
        "protocol_fingerprint": hashlib.sha256(
            json.dumps(profile["protocol"], sort_keys=True).encode()
        ).hexdigest(),
        "nonspecific_calcium_kinetics_convention": (
            conventions.nonspecific_calcium_kinetics_convention
        ),
        "runtime_discriminator": {
            field: getattr(conventions, field) for field in runtime_expectations
        },
        "convergent_external_source_scope": convergent_source_scope,
        "figure6_result": training_result,
        "figure6_relay_event_counts_by_index": training_counts,
        "figure6_cortical_recruitment": recruitment,
        "figure6_top_down_timing": timing,
        "figure6_gates": figure6_gates,
        "figure6_pass": figure6_pass,
        "applied_common_weight_factor": applied_factor,
        "match_result": match,
        "match_gates": match_gates,
        "match_pass": match_pass,
        "mismatch_consulted": False,
        "promotable": False,
        "reproduced": False,
        "advance_to_independent_match_verification": match_pass,
        "next_gate": profile["next_gate"],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(_plain(artifact), sort_keys=False))
    print(
        f"figure6_relay={len(training_indices)} figure6_pass={figure6_pass} "
        f"match_relay={len(match.relay_spike_times_ms) if match else '-'} "
        f"match_trn={len(match.trn_spike_times_ms) if match else '-'} "
        f"match_nonspecific={len(match.nonspecific_spike_times_ms) if match else '-'} "
        f"match_pass={match_pass}",
        flush=True,
    )


if __name__ == "__main__":
    main()
