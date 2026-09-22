"""Run sealed SST-like Stage 4A learning and first-order progression."""

from __future__ import annotations

import argparse
import fcntl
import os
import platform
import tempfile
from pathlib import Path
from typing import Any

import run_l5_sst_like_behavioral_grid as stage3_runner
import yaml
from run_layer5_distal_nak_stage3 import _without_repetition

from smart_robustness import classic_sector
from smart_robustness.baseline import load_frozen_classic_baseline
from smart_robustness.models.sst_like_feedback import make_l5_sst_like_sector_builder
from smart_robustness.validation.active_apical_recording import file_sha256
from smart_robustness.validation.figure6 import (
    BOTTOM_UP_PROJECTION_ID,
    TOP_DOWN_NARROW_PROJECTION_ID,
    TOP_DOWN_WIDE_PROJECTION_ID,
    Figure6LearningProtocol,
    assess_figure6_cortical_recruitment,
    assess_figure6_top_down_timing,
    run_figure6_learning,
)
from smart_robustness.validation.figure10_search_cycle_spread import (
    build_projection036_variance_sector,
)

REPETITIONS = 2
DEFAULT_SEAL = Path(
    "docs/validation-results/post2008-l5-sst-like-stage4a-seal-1003.yaml"
)
ELIGIBLE_POINT_IDS = (
    "delay1p0-resource0p125",
    "delay1p0-resource0p25",
    "delay3p0-resource0p25",
    "delay3p0-resource0p5",
    "delay7p0-resource0p25",
    "delay7p0-resource0p5",
    "delay7p0-resource1p0",
)


def registered_points() -> list[dict[str, float | int | str]]:
    """Return every Stage-3 survivor in its inherited prospective order."""

    by_id = {point["point_id"]: point for point in stage3_runner.registered_points()}
    return [
        {**by_id[point_id], "stage4a_index": index}
        for index, point_id in enumerate(ELIGIBLE_POINT_IDS)
    ]


def checkpoint(path: Path, payload: dict[str, Any]) -> None:
    """Atomically persist one completed repetition or point classification."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".l5-sst-stage4a-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            yaml.safe_dump(payload, stream, sort_keys=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def verify_seal(path: Path) -> dict[str, Any]:
    """Verify the separate pre-outcome Stage-4A execution seal."""

    seal = yaml.safe_load(path.read_text())
    if seal["status"] != "sealed-before-stage4a-network-outcomes":
        raise ValueError("a Stage-4A execution seal is required")
    for filename, digest in seal["files"].items():
        if file_sha256(Path(filename)) != digest:
            raise ValueError(f"sealed file changed: {filename}")
    if "scripts/run_l5_sst_like_stage4a.py" not in seal["files"]:
        raise ValueError("Stage-4A runner is missing from execution seal")
    return seal


def _point_builder(point: dict[str, Any]):
    return make_l5_sst_like_sector_builder(
        base_builder=build_projection036_variance_sector,
        total_conductance_nS=float(point["total_conductance_nS"]),
        delay_ms=float(point["delay_ms"]),
    )


def _figure6_summary(training: Any, expected_relay: set[int]) -> dict[str, Any]:
    relay_indices = training.result.population_spike_indices["thalamic_relay"]
    relay_counts = {index: relay_indices.count(index) for index in expected_relay}
    recruitment = assess_figure6_cortical_recruitment(training.result)
    timing = assess_figure6_top_down_timing(training.result)
    gates = {
        "relay_active_indices": set(relay_indices) == expected_relay,
        "relay_events_per_active_index": set(relay_counts.values()) == {4},
        "relay_events": len(relay_indices) == 20,
        "cortical_chain_complete": recruitment.feedforward_chain_complete,
        "causal_learning_pair": timing.causal_pair_in_learning_window,
        "top_down_horizontal_contrast": (
            training.result.top_down_combined.horizontal_orientation_contrast > 0
        ),
    }
    return {
        "gates": gates,
        "pass": all(gates.values()),
        "population_spike_counts": training.result.population_spikes,
        "population_spike_indices": training.result.population_spike_indices,
        "population_spike_times_ms": training.result.population_spike_times_ms,
        "relay_active_indices": sorted(set(relay_indices)),
        "relay_event_count": len(relay_indices),
        "bottom_up_weights": training.learned_weights[BOTTOM_UP_PROJECTION_ID],
        "top_down_wide_weights": training.learned_weights[
            TOP_DOWN_WIDE_PROJECTION_ID
        ],
        "top_down_narrow_weights": training.learned_weights[
            TOP_DOWN_NARROW_PROJECTION_ID
        ],
        "top_down_horizontal_contrast": (
            training.result.top_down_combined.horizontal_orientation_contrast
        ),
    }


def run_registered_repetition(
    *,
    point: dict[str, Any],
    repetition: int,
    conventions: Any,
    scales: dict[str, float],
    profile: dict[str, Any],
    training_profile: dict[str, Any],
    brian: Any,
) -> dict[str, Any]:
    """Run fresh transformed learning and the inherited first-order gates."""

    expected_relay = set(profile["figure6_gates"]["relay_active_indices"])
    original_builder = classic_sector.build_first_order_connected_sector
    try:
        classic_sector.build_first_order_connected_sector = _point_builder(point)
        training = run_figure6_learning(
            conventions=conventions,
            protocol=Figure6LearningProtocol(
                monitored_populations=tuple(training_profile["monitored_populations"])
            ),
            projection_weight_scales=scales,
            brian=brian,
        )
    finally:
        classic_sector.build_first_order_connected_sector = original_builder

    figure6 = _figure6_summary(training, expected_relay)
    learned_weights = {
        BOTTOM_UP_PROJECTION_ID: figure6["bottom_up_weights"],
        TOP_DOWN_WIDE_PROJECTION_ID: figure6["top_down_wide_weights"],
        TOP_DOWN_NARROW_PROJECTION_ID: figure6["top_down_narrow_weights"],
    }
    behavioral = stage3_runner.run_registered_repetition(
        total_conductance_nS=float(point["total_conductance_nS"]),
        delay_ms=float(point["delay_ms"]),
        learned_weights=learned_weights,
        conventions=conventions,
        scales=scales,
        profile=profile,
        repetition=repetition,
        brian=brian,
    )
    progression_pass = bool(figure6["pass"] and behavioral["behavioral_pass"])
    return {
        "repetition": repetition,
        "figure6": figure6,
        "figure7": behavioral["figure7"],
        "figure10": behavioral["figure10"],
        "progression_pass": progression_pass,
    }


def classify_point(outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply exact-repeat and complete Stage-4A conjunction rules."""

    if len(outcomes) != REPETITIONS:
        raise ValueError("Stage-4A classification requires exactly two repetitions")
    normalized = [_without_repetition(outcome) for outcome in outcomes]
    exact_repeat = normalized[0] == normalized[1]
    all_progression_gates_pass = all(
        outcome.get("progression_pass") is True for outcome in outcomes
    )
    return {
        "exact_repeat": exact_repeat,
        "all_progression_gates_pass": all_progression_gates_pass,
        "classification": (
            "learning_first_order_survival"
            if exact_repeat and all_progression_gates_pass
            else "failure"
        ),
    }


def validate_checkpoint_points(points: list[dict[str, Any]]) -> None:
    """Reject extra, reordered, skipped or prematurely classified points."""

    expected = registered_points()
    if len(points) > len(expected):
        raise ValueError("checkpoint contains unregistered Stage-4A points")
    identity_keys = (
        "stage4a_index",
        "point_index",
        "point_id",
        "delay_ms",
        "resource_fraction",
        "total_conductance_nS",
    )
    for index, point in enumerate(points):
        if any(point.get(key) != expected[index][key] for key in identity_keys):
            raise ValueError("checkpoint violates Stage-4A point order or identity")
        outcomes = point.get("outcomes")
        if not isinstance(outcomes, list) or len(outcomes) > REPETITIONS:
            raise ValueError("checkpoint has an invalid repetition inventory")
        for repetition, outcome in enumerate(outcomes):
            if outcome.get("repetition") != repetition:
                raise ValueError("checkpoint violates repetition order")
        classified = point.get("classification") is not None
        if classified and len(outcomes) != REPETITIONS:
            raise ValueError("checkpoint classifies an incomplete point")
        if index < len(points) - 1 and not classified:
            raise ValueError("checkpoint advances beyond an incomplete point")


def execute(output: Path, seal_path: Path) -> None:
    import brian2 as brian

    seal = verify_seal(seal_path)
    baseline = load_frozen_classic_baseline(seal["baseline_manifest"])
    manifest = yaml.safe_load(Path(seal["baseline_manifest"]).read_text())
    profile_path = baseline.repository_root / manifest["implementation"]["profile"]["path"]
    profile = yaml.safe_load(profile_path.read_text())
    training_profile = yaml.safe_load(
        (baseline.repository_root / profile["training_profile"]).read_text()
    )
    identity = {
        "seal_sha256": file_sha256(seal_path),
        "stage4_registration_sha256": seal["stage4_registration_sha256"],
        "stage3_assessment_sha256": seal["stage3_assessment_sha256"],
        "baseline_manifest_fingerprint": baseline.manifest_fingerprint,
        "runtime_fingerprint": baseline.runtime_fingerprint,
        "registered_points": registered_points(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "brian2": brian.__version__,
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "running-l5-sst-like-stage4a",
        "identity": identity,
        "points": [],
        "all_points_reported": False,
        "classification_counts": {},
        "network_execution": True,
        "frozen_baseline_modified": False,
    }
    if output.exists():
        payload = yaml.safe_load(output.read_text())
        if payload.get("identity") != identity:
            raise ValueError("checkpoint identity mismatch")
    points = payload.get("points")
    if not isinstance(points, list):
        raise TypeError("checkpoint point inventory is invalid")
    validate_checkpoint_points(points)
    if payload.get("status") == "completed-l5-sst-like-stage4a":
        if len(points) != len(registered_points()):
            raise ValueError("incomplete checkpoint marked complete")
        return

    brian.prefs.codegen.target = "numpy"
    conventions = baseline.runtime_conventions()
    scales = dict(baseline.projection_weight_scales)
    expected_points = registered_points()
    for point_index, registered in enumerate(expected_points):
        if point_index == len(points):
            point: dict[str, Any] = {
                **registered,
                "outcomes": [],
                "exact_repeat": None,
                "all_progression_gates_pass": None,
                "classification": None,
            }
            points.append(point)
        else:
            point = points[point_index]
        outcomes = point["outcomes"]
        for repetition in range(len(outcomes), REPETITIONS):
            outcomes.append(
                run_registered_repetition(
                    point=point,
                    repetition=repetition,
                    conventions=conventions,
                    scales=scales,
                    profile=profile,
                    training_profile=training_profile,
                    brian=brian,
                )
            )
            checkpoint(output, payload)
        point.update(classify_point(outcomes))
        checkpoint(output, payload)

    counts: dict[str, int] = {}
    for point in points:
        name = str(point["classification"])
        counts[name] = counts.get(name, 0) + 1
    payload["all_points_reported"] = len(points) == len(expected_points)
    payload["classification_counts"] = dict(sorted(counts.items()))
    payload["status"] = "completed-l5-sst-like-stage4a"
    checkpoint(output, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seal", type=Path, default=DEFAULT_SEAL)
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(output.suffix + ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("an identical L5 SST-like Stage-4A runner is active") from error
        execute(output, args.seal.resolve())


if __name__ == "__main__":
    main()
