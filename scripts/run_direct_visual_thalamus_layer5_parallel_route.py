"""Run the sealed parallel-route Figure 7 and conditional Figure 10 assay."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import yaml
from run_layer5_distal_nak_stage3 import (
    _condition_summary,
    _figure7_gates,
    _figure10_gates,
    _target_summary,
    _without_repetition,
)

from smart_robustness import classic_sector
from smart_robustness.baseline import load_frozen_classic_baseline
from smart_robustness.models.direct_visual_thalamus_layer5 import (
    make_direct_visual_thalamus_layer5_sector_builder,
)
from smart_robustness.protocols import MatchCondition
from smart_robustness.validation import figure7 as figure7_module
from smart_robustness.validation import figure10_search_cycle_spread as figure10_module
from smart_robustness.validation.figure7 import TopDownCurrentMode
from smart_robustness.validation.figure10_search_cycle_spread import (
    build_projection036_variance_sector,
)

ARM_RATIOS = {
    "ratio_0": 0.0,
    "ratio_0p03125": 0.03125,
    "ratio_0p0625": 0.0625,
    "ratio_0p125": 0.125,
    "ratio_0p25": 0.25,
    "ratio_0p5": 0.5,
    "ratio_1": 1.0,
}
ARM_ORDER = tuple(ARM_RATIOS)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_repetition(
    *,
    ratio: float,
    learned_weights,
    conventions,
    scales,
    profile,
    repetition: int,
    brian,
) -> dict[str, object]:
    figure7_protocol = profile["figure7_protocol"]
    comparator = profile["comparator"]
    common7 = {
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
        "brian": brian,
    }
    original7 = classic_sector.build_first_order_connected_sector
    try:
        classic_sector.build_first_order_connected_sector = (
            make_direct_visual_thalamus_layer5_sector_builder(
                strength_ratio=ratio,
                base_builder=build_projection036_variance_sector,
            )
        )
        match_result = figure7_module.run_figure7_condition(
            condition=MatchCondition.MATCH,
            **common7,
        )
        mismatch_result = figure7_module.run_figure7_condition(
            condition=MatchCondition.MISMATCH,
            **common7,
        )
    finally:
        classic_sector.build_first_order_connected_sector = original7
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
        "behavioral_pass": False,
    }
    if not all(figure7_gates.values()):
        return outcome

    figure10_protocol = profile["figure10_protocol"]
    release_ms = float(figure10_protocol["release_after_mismatch_ms"])
    common10 = {
        "top_down_current_pA": float(figure10_protocol["top_down_current_pA"]),
        "pre_match_duration_ms": float(figure10_protocol["pre_match_duration_ms"]),
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
        "brian": brian,
    }
    original10 = figure10_module._build_first_order_connected_sector
    try:
        figure10_module._build_first_order_connected_sector = (
            make_direct_visual_thalamus_layer5_sector_builder(
                strength_ratio=ratio,
                base_builder=original10,
            )
        )
        intact = figure10_module.run_figure10_search_cycle_spread_condition(
            reset_pathway_enabled=True,
            **common10,
        )
        control = figure10_module.run_figure10_search_cycle_spread_condition(
            reset_pathway_enabled=False,
            **common10,
        )
    finally:
        figure10_module._build_first_order_connected_sector = original10
    intact_summary = _target_summary(
        intact,
        release_ms=release_ms,
        late_start_ms=75.0,
    )
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
    outcome["behavioral_pass"] = all(figure10_gates.values())
    return outcome


def _identity(args, baseline, figure6_sha256: str) -> dict[str, object]:
    return {
        "baseline_manifest": args.baseline,
        "baseline_manifest_fingerprint": baseline.manifest_fingerprint,
        "runtime_fingerprint": baseline.runtime_fingerprint,
        "study": args.study,
        "registration": args.registration,
        "figure6_result_sha256": figure6_sha256,
        "network_outcome_used_for_parameter_selection": False,
    }


def _load_arms(output: Path, *, identity: dict[str, object]) -> dict[str, object]:
    if not output.exists():
        return {}
    raw = yaml.safe_load(output.read_text())
    if any(raw.get(key) != value for key, value in identity.items()):
        raise ValueError("existing parallel-route result does not match this sealed run")
    arms = raw.get("arms", {})
    if not isinstance(arms, dict) or set(arms) - set(ARM_ORDER):
        raise ValueError("existing parallel-route result contains unknown arms")
    return arms


def _arm_summary(
    ratio: float,
    outcomes: list[dict[str, object]],
    exact_reruns: int,
) -> dict[str, object]:
    exact_repeat = len(outcomes) < 2 or all(
        _without_repetition(item) == _without_repetition(outcomes[0])
        for item in outcomes[1:]
    )
    return {
        "strength_ratio": ratio,
        "outcomes": outcomes,
        "exact_repeat": exact_repeat,
        "figure7_all_repetitions_pass": bool(
            len(outcomes) == exact_reruns
            and exact_repeat
            and all(item["figure7"]["pass"] for item in outcomes)
        ),
        "behavioral_all_repetitions_pass": bool(
            len(outcomes) == exact_reruns
            and exact_repeat
            and all(item["behavioral_pass"] for item in outcomes)
        ),
    }


def _write(
    output: Path,
    *,
    status: str,
    identity: dict[str, object],
    arms: dict[str, object],
    complete: bool,
) -> None:
    payload = {"schema_version": 1, "status": status, **identity, "arms": arms}
    if complete:
        payload.update(
            {
                "all_exact_repeats": all(
                    arms[name]["exact_repeat"] for name in ARM_ORDER
                ),
                "figure7_passing_ratios": [
                    ARM_RATIOS[name]
                    for name in ARM_ORDER
                    if arms[name]["figure7_all_repetitions_pass"]
                ],
                "full_behavioral_passing_ratios": [
                    ARM_RATIOS[name]
                    for name in ARM_ORDER
                    if arms[name]["behavioral_all_repetitions_pass"]
                ],
            }
        )
    output.write_text(yaml.safe_dump(payload, sort_keys=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        default="configs/baselines/classic_smart_calibrated_v1.yaml",
    )
    parser.add_argument(
        "--study",
        default=(
            "configs/robustness/"
            "direct_visual_thalamus_layer5_parallel_route_v1.yaml"
        ),
    )
    parser.add_argument(
        "--registration",
        default=(
            "docs/validation-results/"
            "post2008-direct-visual-thalamus-layer5-parallel-route-registration-941.yaml"
        ),
    )
    parser.add_argument(
        "--figure6-result",
        default="results/direct-visual-thalamus-layer5-figure6-940.yaml",
    )
    parser.add_argument(
        "--figure6-assessment",
        default=(
            "docs/validation-results/"
            "post2008-direct-visual-thalamus-layer5-figure6-assessment-940.yaml"
        ),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    study = yaml.safe_load(Path(args.study).read_text())
    if tuple(study["execution"]["run_order"]) != ARM_ORDER:
        raise ValueError("runner arm order differs from the sealed study")
    exact_reruns = int(study["execution"]["exact_reruns_per_arm"])
    if exact_reruns != 2:
        raise ValueError("the sealed study requires two exact reruns per arm")
    assessment = yaml.safe_load(Path(args.figure6_assessment).read_text())
    figure6_path = Path(args.figure6_result)
    figure6_sha256 = _sha256(figure6_path)
    if figure6_sha256 != assessment["raw_result"]["sha256"]:
        raise ValueError("Figure 6 result differs from sealed artifact 940")
    figure6 = yaml.safe_load(figure6_path.read_text())
    baseline = load_frozen_classic_baseline(args.baseline)
    profile_path = baseline.repository_root / yaml.safe_load(
        Path(args.baseline).read_text()
    )["implementation"]["profile"]["path"]
    profile = yaml.safe_load(profile_path.read_text())

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    identity = _identity(args, baseline, figure6_sha256)
    arms = _load_arms(output, identity=identity)
    scales = dict(baseline.projection_weight_scales)
    for arm_index, arm_name in enumerate(ARM_ORDER):
        if any(name in arms for name in ARM_ORDER[arm_index + 1 :]) and arm_name not in arms:
            raise ValueError("resume result violates the sealed arm order")
        source_trials = figure6["arms"][arm_name]["trials"]
        if len(source_trials) != exact_reruns:
            raise ValueError(f"{arm_name} lacks two sealed Figure 6 trials")
        keys = (
            "bottom_up_weights",
            "top_down_wide_weights",
            "top_down_narrow_weights",
        )
        if any(source_trials[0][key] != source_trials[1][key] for key in keys):
            raise ValueError(f"{arm_name} Figure 6 learned weights are not exact")
        learned_weights = {
            "modeldb112923.projection.035": source_trials[0]["bottom_up_weights"],
            "modeldb112923.projection.005": source_trials[0]["top_down_wide_weights"],
            "modeldb112923.projection.007": source_trials[0]["top_down_narrow_weights"],
        }
        outcomes = list(arms.get(arm_name, {}).get("outcomes", []))
        if len(outcomes) > exact_reruns:
            raise ValueError(f"invalid saved outcome count for {arm_name}")
        for repetition in range(len(outcomes), exact_reruns):
            outcomes.append(
                _run_repetition(
                    ratio=ARM_RATIOS[arm_name],
                    learned_weights=learned_weights,
                    conventions=baseline.runtime_conventions(),
                    scales=scales,
                    profile=profile,
                    repetition=repetition,
                    brian=brian,
                )
            )
            arms[arm_name] = _arm_summary(
                ARM_RATIOS[arm_name],
                outcomes,
                exact_reruns,
            )
            _write(
                output,
                status="running-direct-visual-thalamus-layer5-parallel-route",
                identity=identity,
                arms=arms,
                complete=False,
            )
    _write(
        output,
        status="completed-direct-visual-thalamus-layer5-parallel-route",
        identity=identity,
        arms=arms,
        complete=True,
    )


if __name__ == "__main__":
    main()
