"""Execute the sealed exact-null inhibitory-routing behavioral validation."""

from __future__ import annotations

import argparse
import fcntl
import os
import platform
import tempfile
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
from smart_robustness.models.inhibitory_routing import (
    InhibitoryRoutingMode,
    make_inhibitory_routing_sector_builder,
)
from smart_robustness.protocols import MatchCondition
from smart_robustness.validation import figure7 as figure7_module
from smart_robustness.validation import figure10_search_cycle_spread as figure10_module
from smart_robustness.validation.active_apical_recording import file_sha256
from smart_robustness.validation.figure7 import TopDownCurrentMode
from smart_robustness.validation.figure10_search_cycle_spread import (
    build_projection036_variance_sector,
)

ARM_ORDER = ("unwrapped_control", "wrapped_legacy_aggregate")
REPETITIONS = 2


def checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".inhibitory-null-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            yaml.safe_dump(payload, stream, sort_keys=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def verify_seal(path: Path) -> dict:
    seal = yaml.safe_load(path.read_text())
    if seal["status"] != "sealed-before-null-behavioral-outcomes":
        raise ValueError("an exact-null behavioral execution seal is required")
    for filename, digest in seal["files"].items():
        if file_sha256(Path(filename)) != digest:
            raise ValueError(f"sealed file changed: {filename}")
    return seal


def _builder(*, wrapped: bool, base_builder):
    if not wrapped:
        return base_builder
    return make_inhibitory_routing_sector_builder(
        mode=InhibitoryRoutingMode.LEGACY_AGGREGATE,
        base_builder=base_builder,
    )


def _run_repetition(
    *,
    wrapped: bool,
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
        classic_sector.build_first_order_connected_sector = _builder(
            wrapped=wrapped,
            base_builder=build_projection036_variance_sector,
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
        figure10_module._build_first_order_connected_sector = _builder(
            wrapped=wrapped,
            base_builder=original10,
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
    intact_summary = _target_summary(intact, release_ms=release_ms, late_start_ms=75.0)
    control_summary = _target_summary(control, release_ms=release_ms, late_start_ms=75.0)
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


def _arm_summary(outcomes: list[dict[str, object]]) -> dict[str, object]:
    exact_repeat = len(outcomes) < 2 or all(
        _without_repetition(item) == _without_repetition(outcomes[0])
        for item in outcomes[1:]
    )
    return {
        "outcomes": outcomes,
        "exact_repeat": exact_repeat,
        "all_behavioral_gates_pass": bool(
            len(outcomes) == REPETITIONS
            and exact_repeat
            and all(item["behavioral_pass"] for item in outcomes)
        ),
    }


def execute(output: Path, seal_path: Path) -> None:
    import brian2 as brian

    seal = verify_seal(seal_path)
    baseline = load_frozen_classic_baseline(seal["baseline_manifest"])
    figure6_path = Path(seal["figure6_result"])
    figure6 = yaml.safe_load(figure6_path.read_text())
    source_trials = figure6["arms"]["ratio_0"]["trials"]
    if len(source_trials) != 2:
        raise ValueError("sealed ratio-zero Figure 6 result lacks two trials")
    keys = ("bottom_up_weights", "top_down_wide_weights", "top_down_narrow_weights")
    if any(source_trials[0][key] != source_trials[1][key] for key in keys):
        raise ValueError("sealed ratio-zero Figure 6 weights are not exact repeats")
    learned_weights = {
        "modeldb112923.projection.035": source_trials[0]["bottom_up_weights"],
        "modeldb112923.projection.005": source_trials[0]["top_down_wide_weights"],
        "modeldb112923.projection.007": source_trials[0]["top_down_narrow_weights"],
    }
    profile_manifest = yaml.safe_load(Path(seal["baseline_manifest"]).read_text())
    profile = yaml.safe_load(
        (baseline.repository_root / profile_manifest["implementation"]["profile"]["path"]).read_text()
    )
    identity = {
        "seal_sha256": file_sha256(seal_path),
        "baseline_manifest_fingerprint": baseline.manifest_fingerprint,
        "runtime_fingerprint": baseline.runtime_fingerprint,
        "figure6_result_sha256": file_sha256(figure6_path),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "brian2": brian.__version__,
    }
    payload = {
        "schema_version": 1,
        "status": "running-inhibitory-routing-null",
        "identity": identity,
        "arms": {},
        "cross_arm_exact": None,
        "biological_routing_authorized": False,
    }
    if output.exists():
        payload = yaml.safe_load(output.read_text())
        if payload["identity"] != identity:
            raise ValueError("checkpoint identity mismatch")
    if set(payload["arms"]) - set(ARM_ORDER):
        raise ValueError("checkpoint contains an unregistered arm")
    if payload["status"] == "completed-inhibitory-routing-null":
        if set(payload["arms"]) != set(ARM_ORDER):
            raise ValueError("incomplete checkpoint marked complete")
        return

    brian.prefs.codegen.target = "numpy"
    for arm_index, arm_name in enumerate(ARM_ORDER):
        if any(name in payload["arms"] for name in ARM_ORDER[arm_index + 1 :]) and arm_name not in payload["arms"]:
            raise ValueError("resume checkpoint violates registered arm order")
        outcomes = list(payload["arms"].get(arm_name, {}).get("outcomes", []))
        if len(outcomes) > REPETITIONS:
            raise ValueError("checkpoint has too many repetitions")
        for repetition in range(len(outcomes), REPETITIONS):
            outcomes.append(
                _run_repetition(
                    wrapped=arm_name == "wrapped_legacy_aggregate",
                    learned_weights=learned_weights,
                    conventions=baseline.runtime_conventions(),
                    scales=dict(baseline.projection_weight_scales),
                    profile=profile,
                    repetition=repetition,
                    brian=brian,
                )
            )
            payload["arms"][arm_name] = _arm_summary(outcomes)
            checkpoint(output, payload)
    control = payload["arms"]["unwrapped_control"]["outcomes"]
    wrapped = payload["arms"]["wrapped_legacy_aggregate"]["outcomes"]
    payload["cross_arm_exact"] = all(
        _without_repetition(left) == _without_repetition(right)
        for left, right in zip(control, wrapped, strict=True)
    )
    payload["status"] = "completed-inhibitory-routing-null"
    payload["biological_routing_authorized"] = bool(
        payload["cross_arm_exact"]
        and all(payload["arms"][name]["all_behavioral_gates_pass"] for name in ARM_ORDER)
    )
    checkpoint(output, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--seal",
        type=Path,
        default=Path(
            "docs/validation-results/post2008-inhibitory-routing-null-seal-972.yaml"
        ),
    )
    args = parser.parse_args()
    lock_path = args.output.with_suffix(args.output.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("an identical inhibitory-null runner is active") from error
        execute(args.output, args.seal)


if __name__ == "__main__":
    main()
