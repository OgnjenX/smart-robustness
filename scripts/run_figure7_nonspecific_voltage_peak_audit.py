"""Audit nonspecific voltage peaks in the fixed top-five Figure 7 pair."""

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
    Figure7ConditionResult,
    TopDownCurrentMode,
    assess_figure7_reproduction,
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


def _condition_summary(result: Figure7ConditionResult) -> dict[str, Any]:
    return {
        "nonspecific_event_count": len(result.nonspecific_spike_times_ms),
        "nonspecific_event_times_ms": result.nonspecific_spike_times_ms,
        "positive_soma_local_maximum_count": len(
            result.nonspecific_positive_soma_local_maxima_ms_mV
        ),
        "positive_soma_local_maxima_ms_mV": (
            result.nonspecific_positive_soma_local_maxima_ms_mV
        ),
        "positive_detector_local_maximum_count": len(
            result.nonspecific_positive_detector_local_maxima_ms_mV
        ),
        "positive_detector_local_maxima_ms_mV": (
            result.nonspecific_positive_detector_local_maxima_ms_mV
        ),
        "detector_voltage_range_mV": result.nonspecific_detector_voltage_range_mV,
        "detector_threshold_upcrossings": (
            result.nonspecific_detector_threshold_upcrossings
        ),
        "detector_zero_downcrossings": (
            result.nonspecific_detector_zero_downcrossings
        ),
        "detector_arm_transitions": result.nonspecific_detector_arm_transitions,
        "detector_release_transitions": (
            result.nonspecific_detector_release_transitions
        ),
        "detector_final_armed": result.nonspecific_detector_final_armed,
        "relay_event_count": len(result.relay_spike_times_ms),
        "relay_active_indices": sorted(set(result.relay_spike_indices)),
        "trn_event_count": len(result.trn_spike_times_ms),
    }


def _classify_mismatch(
    result: Figure7ConditionResult, expected_events: int
) -> str:
    soma_peaks = len(result.nonspecific_positive_soma_local_maxima_ms_mV)
    upcrossings = result.nonspecific_detector_threshold_upcrossings
    emitted = len(result.nonspecific_spike_times_ms)
    if soma_peaks < expected_events:
        return "missing_cycles_absent_from_positive_somatic_waveform"
    if upcrossings is None or upcrossings < expected_events:
        return "additional_positive_cycles_are_subthreshold"
    if emitted < expected_events:
        return "detector_recovery_or_event_conversion_loses_cycles"
    return "official_mismatch_event_count_present"


def _shared_soma_threshold_analysis(
    match: Figure7ConditionResult,
    mismatch: Figure7ConditionResult,
    *,
    expected_match: int,
    expected_mismatch: int,
    current_threshold_mV: float,
) -> dict[str, Any]:
    match_amplitudes = tuple(
        amplitude
        for _, amplitude in match.nonspecific_positive_soma_local_maxima_ms_mV
    )
    mismatch_amplitudes = tuple(
        amplitude
        for _, amplitude in mismatch.nonspecific_positive_soma_local_maxima_ms_mV
    )
    thresholds = sorted(
        {0.0, current_threshold_mV, *match_amplitudes, *mismatch_amplitudes}
    )
    outcomes = tuple(
        (
            threshold,
            sum(amplitude > threshold for amplitude in match_amplitudes),
            sum(amplitude > threshold for amplitude in mismatch_amplitudes),
        )
        for threshold in thresholds
    )
    match_preserving = tuple(
        outcome for outcome in outcomes if outcome[1] == expected_match
    )
    return {
        "expected_counts": {"match": expected_match, "mismatch": expected_mismatch},
        "evaluated_threshold_count": len(thresholds),
        "current_threshold_mV": current_threshold_mV,
        "current_threshold_peak_counts": {
            "match": sum(
                amplitude > current_threshold_mV for amplitude in match_amplitudes
            ),
            "mismatch": sum(
                amplitude > current_threshold_mV for amplitude in mismatch_amplitudes
            ),
        },
        "exact_shared_threshold_exists": any(
            match_count == expected_match and mismatch_count == expected_mismatch
            for _, match_count, mismatch_count in outcomes
        ),
        "maximum_mismatch_peaks_while_preserving_match_count": max(
            (outcome[2] for outcome in match_preserving), default=None
        ),
        "interpretation": (
            "shared_somatic_threshold_cannot_recover_4_7_from_observed_peaks"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        default=(
            "configs/calibration/"
            "figure7_top5_nonspecific_voltage_peak_audit_v1.yaml"
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
    learned, applied_factor = expand_figure7_source_expectation_toward_bounds(
        training.learned_weights,
        headroom_fraction=float(profile["learned_state"]["headroom_fraction"]),
        source_index=int(profile["learned_state"]["source_index"]),
    )
    protocol = profile["protocol"]
    source_index = int(profile["learned_state"]["source_index"])
    target_count = int(profile["comparator"]["target_count"])
    results: dict[MatchCondition, Figure7ConditionResult] = {}
    for condition_name in protocol["conditions"]:
        condition = MatchCondition(condition_name)
        results[condition] = run_figure7_condition(
            condition=condition,
            top_down_current_pA=float(protocol["top_down_current_pA"]),
            learned_weights=learned,
            conventions=conventions,
            duration_ms=float(protocol["duration_ms"]),
            dt_ms=float(protocol["dt_ms"]),
            record_relay_diagnostics=True,
            persistent_projection_weight_scales=scales,
            comparator_top_k_targets=target_count,
            comparator_source_index=source_index,
            top_down_current_mode=TopDownCurrentMode(
                protocol["top_down_current_mode"]
            ),
            top_down_cue_lead_ms=float(protocol["top_down_cue_lead_ms"]),
            equilibration_ms=float(protocol["equilibration_ms"]),
            brian=brian,
        )

    match = results[MatchCondition.MATCH]
    mismatch = results[MatchCondition.MISMATCH]
    official_assessment = assess_figure7_reproduction(match, mismatch)
    expected_mismatch = int(profile["diagnostic"]["expected_mismatch_events"])
    expected_match = int(profile["diagnostic"]["expected_match_events"])
    classification = _classify_mismatch(mismatch, expected_mismatch)
    artifact = {
        "schema_version": 1,
        "id": Path(args.output).stem,
        "date": datetime.now(tz=UTC).date().isoformat(),
        "status": "complete-read-only-diagnostic",
        "classification": "calibrated-reconstruction-not-recovered-source",
        "profile": args.profile,
        "registration_artifact": profile["registration_artifact"],
        "base_candidate_fingerprint": base_profile["candidate_fingerprint"],
        "runtime_fingerprint": conventions.fingerprint,
        "parameter_changes": "none",
        "handoff_figure6_population_spikes": training.result.population_spikes,
        "applied_common_weight_factor": applied_factor,
        "match_result": match,
        "mismatch_result": mismatch,
        "match_diagnostic": _condition_summary(match),
        "mismatch_diagnostic": _condition_summary(mismatch),
        "diagnostic_classification": classification,
        "derived_shared_soma_threshold_analysis": _shared_soma_threshold_analysis(
            match,
            mismatch,
            expected_match=expected_match,
            expected_mismatch=expected_mismatch,
            current_threshold_mV=float(
                profile["diagnostic"]["detector_threshold_upcrossing_mV"]
            ),
        ),
        "official_assessment": official_assessment,
        "reproduced": official_assessment.reproduced,
        "next_gate": profile["next_gate"],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(_plain(artifact), sort_keys=False))
    print(
        "match events/positive-peaks/upcrossings="
        f"{len(match.nonspecific_spike_times_ms)}/"
        f"{len(match.nonspecific_positive_soma_local_maxima_ms_mV)}/"
        f"{match.nonspecific_detector_threshold_upcrossings}; "
        "mismatch="
        f"{len(mismatch.nonspecific_spike_times_ms)}/"
        f"{len(mismatch.nonspecific_positive_soma_local_maxima_ms_mV)}/"
        f"{mismatch.nonspecific_detector_threshold_upcrossings}; "
        f"classification={classification}",
        flush=True,
    )


if __name__ == "__main__":
    main()
