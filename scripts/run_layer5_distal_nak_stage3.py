"""Run registered Figure 7 and Figure 10 layer-5 distal Na/K comparisons."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import yaml

from smart_robustness import classic_sector
from smart_robustness.baseline import load_frozen_classic_baseline
from smart_robustness.models.mechanism_interventions import (
    LAYER5_CELL_CLASS,
    create_layer5_distal_nak_disabled_population,
)
from smart_robustness.models.selective import make_selective_population_factory
from smart_robustness.protocols import MatchCondition
from smart_robustness.validation import figure7 as figure7_module
from smart_robustness.validation.figure7 import TopDownCurrentMode
from smart_robustness.validation.figure10_search_cycle_spread import (
    VERTICAL_ALTERNATIVES,
    VERTICAL_TARGETS,
    build_projection036_variance_sector,
    run_figure10_search_cycle_spread_condition,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _condition_summary(result) -> dict[str, object]:
    return {
        "relay_events": len(result.relay_spike_times_ms),
        "relay_active_indices": sorted(set(result.relay_spike_indices)),
        "trn_events": len(result.trn_spike_times_ms),
        "nonspecific_events": len(result.nonspecific_spike_times_ms),
        "nonspecific_event_times_ms": list(result.nonspecific_spike_times_ms),
        "trn_to_nonspecific_gaba_integral_ms": (
            result.nonspecific_trn_gaba_integral_ms
        ),
    }


def _figure7_gates(
    match: dict[str, object],
    mismatch: dict[str, object],
    profile: dict[str, object],
) -> dict[str, bool]:
    gates = profile["figure7_gates"]
    match_indices = match["relay_active_indices"]
    mismatch_indices = mismatch["relay_active_indices"]
    return {
        "match_relay_active_indices": (
            match_indices == gates["match_relay_active_indices"]
        ),
        "match_relay_events": match["relay_events"] == gates["match_relay_events"],
        "mismatch_relay_allowed_indices": (
            mismatch_indices == gates["mismatch_relay_allowed_indices"]
        ),
        "match_more_active_relay_cells": (
            len(match_indices) > len(mismatch_indices)
        ),
        "match_more_trn_events": match["trn_events"] > mismatch["trn_events"],
        "match_more_trn_to_nonspecific_gaba": (
            match["trn_to_nonspecific_gaba_integral_ms"]
            > mismatch["trn_to_nonspecific_gaba_integral_ms"]
        ),
        "match_nonspecific_events": (
            match["nonspecific_events"] == gates["match_nonspecific_events"]
        ),
        "mismatch_nonspecific_events": (
            mismatch["nonspecific_events"] == gates["mismatch_nonspecific_events"]
        ),
    }


def _target_summary(result, *, release_ms: float, late_start_ms: float) -> dict:
    events = result.mismatch_layer4_events
    indices = np.asarray([item[0] for item in events], dtype=int)
    times = np.asarray([item[1] for item in events], dtype=float)
    pre_release_alternative_events = int(
        np.count_nonzero(
            (times < release_ms) & np.isin(indices, VERTICAL_ALTERNATIVES)
        )
    )
    rows = []
    for target in VERTICAL_TARGETS:
        target_times = times[(times >= late_start_ms) & (indices == target)]
        rows.append(
            {
                "target": target,
                "events": int(target_times.size),
                "first_event_ms": (
                    None if target_times.size == 0 else float(target_times[0])
                ),
                "last_event_ms": (
                    None if target_times.size == 0 else float(target_times[-1])
                ),
            }
        )
    counts = {row["target"]: row["events"] for row in rows}
    max_alternative = max(counts[target] for target in VERTICAL_ALTERNATIVES)
    replacement_winners = sorted(
        target
        for target in VERTICAL_ALTERNATIVES
        if counts[target] == max_alternative and counts[target] > counts[40]
    )
    return {
        "pre_layer4_events": result.pre_layer4_events,
        "pre_layer4_active_indices": list(result.pre_layer4_active_indices),
        "aggregate_winner_events": result.winner_post_events,
        "aggregate_alternative_events": result.alternative_events,
        "pre_release_alternative_events": pre_release_alternative_events,
        "late_reset_targets": rows,
        "late_old_center_events": counts[40],
        "late_max_alternative_events": max_alternative,
        "late_alternative_margin": max_alternative - counts[40],
        "replacement_winner_candidates": replacement_winners,
        "nonspecific_post_events": result.nonspecific_post_events,
        "layer5_post_events": result.layer5_post_events,
        "layer6i_post_events": result.layer6i_post_events,
        "projection035_post_release_integral_pA_ms": [
            list(item) for item in result.projection035_post_release_integral_pA_ms
        ],
    }


def _figure10_gates(intact, control, intact_summary, control_summary) -> dict[str, bool]:
    return {
        "identical_nonempty_prestate": (
            intact.pre_layer4_events > 0
            and intact.pre_layer4_events == control.pre_layer4_events
            and intact.pre_layer4_active_indices == control.pre_layer4_active_indices
        ),
        "alternatives_quiet_before_release": (
            intact_summary["pre_release_alternative_events"] == 0
            and control_summary["pre_release_alternative_events"] == 0
        ),
        "full_input_delivery": all(
            value > 0
            for result in (intact, control)
            for _, value in result.projection035_post_release_integral_pA_ms
        ),
        "intact_reset_chain": (
            intact.nonspecific_post_events > 0
            and intact.layer5_post_events > 0
            and intact.layer6i_post_events > 0
        ),
        "aggregate_winner_suppression": (
            intact.winner_post_events < control.winner_post_events
        ),
        "intact_replacement_winner": bool(
            intact_summary["replacement_winner_candidates"]
        ),
        "reset_favors_replacement_margin": (
            intact_summary["late_alternative_margin"]
            > control_summary["late_alternative_margin"]
        ),
    }


def _run_stage3_repetition(
    *,
    population_factory,
    learned_weights,
    conventions,
    scales,
    profile,
    repetition: int,
    brian,
) -> dict[str, object]:
    figure7_protocol = profile["figure7_protocol"]
    comparator = profile["comparator"]
    figure7_common = {
        "learned_weights": learned_weights,
        "conventions": conventions,
        "persistent_projection_weight_scales": scales,
        "top_down_current_pA": float(figure7_protocol["top_down_current_pA"]),
        "top_down_current_mode": TopDownCurrentMode(
            figure7_protocol["top_down_current_mode"]
        ),
        "top_down_cue_lead_ms": float(figure7_protocol["top_down_cue_lead_ms"]),
        "duration_ms": float(figure7_protocol["duration_ms"]),
        "dt_ms": float(figure7_protocol["dt_ms"]),
        "equilibration_ms": float(figure7_protocol["equilibration_ms"]),
        "comparator_top_k_targets": int(comparator["target_count"]),
        "comparator_source_index": int(comparator["source_index"]),
        "record_relay_diagnostics": True,
        "population_factory": population_factory,
        "brian": brian,
    }
    original_builder = classic_sector.build_first_order_connected_sector
    try:
        classic_sector.build_first_order_connected_sector = (
            build_projection036_variance_sector
        )
        match_result = figure7_module.run_figure7_condition(
            condition=MatchCondition.MATCH,
            **figure7_common,
        )
        mismatch_result = figure7_module.run_figure7_condition(
            condition=MatchCondition.MISMATCH,
            **figure7_common,
        )
    finally:
        classic_sector.build_first_order_connected_sector = original_builder
    match = _condition_summary(match_result)
    mismatch = _condition_summary(mismatch_result)
    figure7_gates = _figure7_gates(match, mismatch, profile)
    outcome: dict[str, object] = {
        "repetition": repetition,
        "figure7": {
            "match": match,
            "mismatch": mismatch,
            "gates": figure7_gates,
            "pass": all(figure7_gates.values()),
        },
        "figure10": None,
        "stage3_pass": False,
    }
    if not all(figure7_gates.values()):
        return outcome

    figure10_protocol = profile["figure10_protocol"]
    release_ms = float(figure10_protocol["release_after_mismatch_ms"])
    common = {
        "top_down_current_pA": float(figure10_protocol["top_down_current_pA"]),
        "pre_match_duration_ms": float(
            figure10_protocol["pre_match_duration_ms"]
        ),
        "mismatch_duration_ms": float(figure10_protocol["mismatch_duration_ms"]),
        "release_after_mismatch_ms": release_ms,
        "learned_weights": learned_weights,
        "persistent_projection_weight_scales": scales,
        "persistent_projection_delays_ms": {},
        "comparator_top_k_targets": int(comparator["target_count"]),
        "comparator_source_index": int(comparator["source_index"]),
        "top_down_current_mode": figure10_protocol["top_down_current_mode"],
        "conventions": conventions,
        "dt_ms": float(figure10_protocol["dt_ms"]),
        "population_factory": population_factory,
        "brian": brian,
    }
    intact = run_figure10_search_cycle_spread_condition(
        reset_pathway_enabled=True,
        **common,
    )
    control = run_figure10_search_cycle_spread_condition(
        reset_pathway_enabled=False,
        **common,
    )
    intact_summary = _target_summary(intact, release_ms=release_ms, late_start_ms=75.0)
    control_summary = _target_summary(
        control,
        release_ms=release_ms,
        late_start_ms=75.0,
    )
    figure10_gates = _figure10_gates(
        intact,
        control,
        intact_summary,
        control_summary,
    )
    outcome["figure10"] = {
        "intact": intact_summary,
        "disconnected_control": control_summary,
        "gates": figure10_gates,
        "pass": all(figure10_gates.values()),
    }
    outcome["stage3_pass"] = all(figure10_gates.values())
    return outcome


def _without_repetition(outcome: dict[str, object]) -> dict[str, object]:
    return {name: value for name, value in outcome.items() if name != "repetition"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        default="configs/baselines/classic_smart_calibrated_v1.yaml",
    )
    parser.add_argument(
        "--study",
        default="configs/robustness/layer5_distal_nak_ablation_v1.yaml",
    )
    parser.add_argument(
        "--figure6-result",
        default="results/layer5-distal-nak-figure6-924.yaml",
    )
    parser.add_argument(
        "--figure6-assessment",
        default=(
            "docs/validation-results/"
            "mechanism-layer5-distal-nak-figure6-assessment-924.yaml"
        ),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    baseline = load_frozen_classic_baseline(args.baseline)
    study = yaml.safe_load(Path(args.study).read_text())
    assessment = yaml.safe_load(Path(args.figure6_assessment).read_text())
    figure6_path = Path(args.figure6_result)
    if _sha256(figure6_path) != assessment["raw_result"]["sha256"]:
        raise ValueError("Figure 6 result differs from its sealed assessment")
    if not assessment["decision"]["stage3_match_mismatch_and_reset_authorized"]:
        raise ValueError("Figure 6 assessment does not authorize stage 3")
    figure6 = yaml.safe_load(figure6_path.read_text())
    exact_reruns = int(study["execution"]["exact_reruns_per_arm"])
    profile_path = baseline.repository_root / yaml.safe_load(
        Path(args.baseline).read_text()
    )["implementation"]["profile"]["path"]
    profile = yaml.safe_load(profile_path.read_text())
    factories = {
        "classic_control": None,
        "layer5_distal_nak_disabled": make_selective_population_factory(
            create_layer5_distal_nak_disabled_population,
            {LAYER5_CELL_CLASS},
        ),
    }

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arms: dict[str, dict[str, object]] = {}
    scales = dict(baseline.projection_weight_scales)
    for arm_name, population_factory in factories.items():
        source_trials = figure6["arms"][arm_name]["trials"]
        if len(source_trials) != exact_reruns:
            raise ValueError(f"{arm_name} does not contain both sealed Figure 6 runs")
        learned_weight_keys = (
            "bottom_up_weights",
            "top_down_wide_weights",
            "top_down_narrow_weights",
        )
        if any(
            source_trials[0][key] != source_trials[1][key]
            for key in learned_weight_keys
        ):
            raise ValueError(f"{arm_name} Figure 6 learned weights are not exact")
        learned_weights = {
            "modeldb112923.projection.035": source_trials[0]["bottom_up_weights"],
            "modeldb112923.projection.005": source_trials[0]["top_down_wide_weights"],
            "modeldb112923.projection.007": source_trials[0]["top_down_narrow_weights"],
        }
        outcomes = []
        for repetition in range(exact_reruns):
            outcomes.append(
                _run_stage3_repetition(
                    population_factory=population_factory,
                    learned_weights=learned_weights,
                    conventions=baseline.runtime_conventions(),
                    scales=scales,
                    profile=profile,
                    repetition=repetition,
                    brian=brian,
                )
            )
            exact_repeat = len(outcomes) < 2 or (
                _without_repetition(outcomes[0])
                == _without_repetition(outcomes[-1])
            )
            arms[arm_name] = {
                "outcomes": outcomes,
                "exact_repeat": exact_repeat,
                "all_repetitions_pass": bool(
                    len(outcomes) == exact_reruns
                    and exact_repeat
                    and all(item["stage3_pass"] for item in outcomes)
                ),
            }
            output_path.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "status": "running-layer5-distal-nak-stage3",
                        "baseline_manifest": args.baseline,
                        "baseline_manifest_fingerprint": baseline.manifest_fingerprint,
                        "runtime_fingerprint": baseline.runtime_fingerprint,
                        "study": args.study,
                        "figure6_result_sha256": assessment["raw_result"]["sha256"],
                        "network_outcome_used_for_parameter_selection": False,
                        "arms": arms,
                    },
                    sort_keys=False,
                )
            )

    classic_pass = bool(arms["classic_control"]["all_repetitions_pass"])
    ablation_pass = bool(
        arms["layer5_distal_nak_disabled"]["all_repetitions_pass"]
    )
    output_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "status": "completed-layer5-distal-nak-stage3",
                "baseline_manifest": args.baseline,
                "baseline_manifest_fingerprint": baseline.manifest_fingerprint,
                "runtime_fingerprint": baseline.runtime_fingerprint,
                "study": args.study,
                "figure6_result_sha256": assessment["raw_result"]["sha256"],
                "network_outcome_used_for_parameter_selection": False,
                "classic_stage3_pass": classic_pass,
                "ablation_stage3_pass": ablation_pass,
                "stage4_authorized": bool(classic_pass and ablation_pass),
                "arms": arms,
            },
            sort_keys=False,
        )
    )


if __name__ == "__main__":
    main()
