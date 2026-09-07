"""Summarize full-sheet TRN source drive for the unchanged fixed Figure 7 pair."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

from smart_robustness.protocols import MatchCondition
from smart_robustness.validation import figure7 as figure7_module
from smart_robustness.validation.calibration import runtime_conventions_for_candidate
from smart_robustness.validation.figure6 import (
    Figure6LearningProtocol,
    assess_figure6_cortical_recruitment,
    assess_figure6_top_down_timing,
    run_figure6_learning,
)
from smart_robustness.validation.figure7 import TopDownCurrentMode

CENTRAL = frozenset((22, 31, 38, 39, 40, 41, 42, 49, 58))
WHOLE = tuple(range(81))
PERIPHERAL = tuple(index for index in WHOLE if index not in CENTRAL)


def _pair_map(values):
    return {int(index): float(value) for index, value in values}


def _condition_data(result):
    counts = Counter(int(index) for index in result.trn_spike_indices)
    return {
        "events": {index: counts.get(index, 0) for index in WHOLE},
        "relay_ampa": _pair_map(result.trn_relay_ampa_integral_ms_by_index),
        "layer6ii_ampa": _pair_map(result.trn_layer6ii_ampa_integral_ms_by_index),
        "layer6ii_nmda": _pair_map(result.trn_layer6ii_nmda_integral_ms_by_index),
    }


def _region_summary(data, indices):
    return {
        "cell_count": len(indices),
        "trn_events": sum(data["events"][index] for index in indices),
        "relay_ampa_integral_ms": sum(data["relay_ampa"][index] for index in indices),
        "layer6ii_ampa_integral_ms": sum(data["layer6ii_ampa"][index] for index in indices),
        "layer6ii_nmda_integral_ms": sum(data["layer6ii_nmda"][index] for index in indices),
    }


def _order_counts(match, mismatch, indices, field):
    counts = {"match_greater": 0, "equal": 0, "mismatch_greater": 0}
    for index in indices:
        delta = match[field][index] - mismatch[field][index]
        if np.isclose(delta, 0.0, rtol=1e-12, atol=1e-12):
            counts["equal"] += 1
        elif delta > 0:
            counts["match_greater"] += 1
        else:
            counts["mismatch_greater"] += 1
    return counts


def _largest_peripheral_spike_excess(match, mismatch):
    rows = []
    for index in PERIPHERAL:
        spike_delta = mismatch["events"][index] - match["events"][index]
        rows.append(
            {
                "index": index,
                "mismatch_minus_match_trn_events": spike_delta,
                "mismatch_minus_match_relay_ampa_integral_ms": (
                    mismatch["relay_ampa"][index] - match["relay_ampa"][index]
                ),
                "mismatch_minus_match_layer6ii_ampa_integral_ms": (
                    mismatch["layer6ii_ampa"][index] - match["layer6ii_ampa"][index]
                ),
                "mismatch_minus_match_layer6ii_nmda_integral_ms": (
                    mismatch["layer6ii_nmda"][index] - match["layer6ii_nmda"][index]
                ),
            }
        )
    return sorted(
        rows,
        key=lambda row: (-row["mismatch_minus_match_trn_events"], row["index"]),
    )[:15]


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
    if figure7_module.FIGURE7_RELAY_DIAGNOSTIC_INDICES != tuple(sorted(CENTRAL)):
        raise ValueError("unexpected default Figure 7 diagnostic index set")
    figure7_module.FIGURE7_RELAY_DIAGNOSTIC_INDICES = WHOLE

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    persistent_scales = {
        str(key): float(value)
        for key, value in training_profile["projection_weight_scales"].items()
    }
    calibrated = registration["persistent_projection_scale"]
    if calibrated["projection_id"] in persistent_scales:
        raise ValueError("calibrated projection overlaps an existing scale")
    persistent_scales[calibrated["projection_id"]] = float(calibrated["scale"])

    training = run_figure6_learning(
        conventions=conventions,
        protocol=Figure6LearningProtocol(
            monitored_populations=tuple(training_profile["monitored_populations"])
        ),
        projection_weight_scales=persistent_scales,
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

    protocol = profile["protocol"]
    comparator = profile["comparator"]
    common = {
        "learned_weights": training.learned_weights,
        "conventions": conventions,
        "persistent_projection_weight_scales": persistent_scales,
        "top_down_current_pA": float(protocol["top_down_current_pA"]),
        "top_down_current_mode": TopDownCurrentMode(protocol["top_down_current_mode"]),
        "top_down_cue_lead_ms": float(protocol["top_down_cue_lead_ms"]),
        "duration_ms": float(protocol["duration_ms"]),
        "dt_ms": float(protocol["dt_ms"]),
        "equilibration_ms": float(protocol["equilibration_ms"]),
        "comparator_top_k_targets": int(comparator["target_count"]),
        "comparator_source_index": int(comparator["source_index"]),
        "record_relay_diagnostics": True,
        "brian": brian,
    }
    match_result = figure7_module.run_figure7_condition(condition=MatchCondition.MATCH, **common)
    mismatch_result = figure7_module.run_figure7_condition(
        condition=MatchCondition.MISMATCH, **common
    )
    match = _condition_data(match_result)
    mismatch = _condition_data(mismatch_result)
    regions = {"central": tuple(sorted(CENTRAL)), "peripheral": PERIPHERAL, "whole": WHOLE}
    print(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "id": registration["result_id"],
                "date": registration["date"],
                "status": "completed-read-only-full-sheet-trn-summary",
                "classification": registration["classification"],
                "registration": args.registration,
                "runtime_fingerprint": conventions.fingerprint,
                "figure6_gates": figure6_gates,
                "event_identity": {
                    "match": {
                        "relay_events": len(match_result.relay_spike_times_ms),
                        "trn_events": len(match_result.trn_spike_times_ms),
                        "nonspecific_events": len(match_result.nonspecific_spike_times_ms),
                    },
                    "mismatch": {
                        "relay_events": len(mismatch_result.relay_spike_times_ms),
                        "trn_events": len(mismatch_result.trn_spike_times_ms),
                        "nonspecific_events": len(mismatch_result.nonspecific_spike_times_ms),
                    },
                },
                "region_summaries": {
                    name: {
                        "match": _region_summary(match, indices),
                        "mismatch": _region_summary(mismatch, indices),
                    }
                    for name, indices in regions.items()
                },
                "peripheral_per_cell_order_counts": {
                    field: _order_counts(match, mismatch, PERIPHERAL, field)
                    for field in (
                        "events",
                        "relay_ampa",
                        "layer6ii_ampa",
                        "layer6ii_nmda",
                    )
                },
                "largest_peripheral_spike_excess": (
                    _largest_peripheral_spike_excess(match, mismatch)
                ),
                "nonspecific_trn_gaba_integral_ms": {
                    "match": match_result.nonspecific_trn_gaba_integral_ms,
                    "mismatch": mismatch_result.nonspecific_trn_gaba_integral_ms,
                },
                "model_or_protocol_change": False,
                "instrumentation_change_only": True,
                "candidate_reopened": False,
                "original_smart_reproduced": False,
                "baseline_promoted": False,
            },
            sort_keys=False,
        ),
        end="",
    )


if __name__ == "__main__":
    main()
