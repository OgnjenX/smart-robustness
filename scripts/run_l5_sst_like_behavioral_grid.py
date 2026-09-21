"""Run the sealed nonzero L5 SST-like Stage-3 behavioral grid."""

from __future__ import annotations

import argparse
import fcntl
import os
import platform
import tempfile
from pathlib import Path
from typing import Any

import run_inhibitory_routing_null as legacy_runner
import yaml
from run_layer4_pv_like_somatic_routing import (
    load_legacy_reference,
    recursive_differences,
)
from run_layer5_distal_nak_stage3 import _without_repetition

from smart_robustness.baseline import load_frozen_classic_baseline
from smart_robustness.models.sst_like_feedback import make_l5_sst_like_sector_builder
from smart_robustness.validation.active_apical_recording import file_sha256

RESOURCE_ANCHOR_NS = 193.28648801211204
RESOURCE_FRACTIONS = (0.125, 0.25, 0.5, 1.0)
DELAYS_MS = (1.0, 3.0, 7.0)
REPETITIONS = 2
DEFAULT_SEAL = Path(
    "docs/validation-results/post2008-l5-sst-like-behavioral-grid-seal-999.yaml"
)


def registered_points() -> list[dict[str, float | int | str]]:
    """Return the immutable delay-major Stage-3 coordinate sequence."""

    points: list[dict[str, float | int | str]] = []
    for delay_ms in DELAYS_MS:
        for fraction in RESOURCE_FRACTIONS:
            fraction_label = str(fraction).replace(".", "p")
            delay_label = str(delay_ms).replace(".", "p")
            points.append(
                {
                    "point_index": len(points),
                    "point_id": f"delay{delay_label}-resource{fraction_label}",
                    "delay_ms": delay_ms,
                    "resource_fraction": fraction,
                    "total_conductance_nS": RESOURCE_ANCHOR_NS * fraction,
                }
            )
    return points


def checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".l5-sst-stage3-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            yaml.safe_dump(payload, stream, sort_keys=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def verify_seal(path: Path) -> dict[str, Any]:
    seal = yaml.safe_load(path.read_text())
    if seal["status"] != "sealed-before-stage3-network-outcomes":
        raise ValueError("a Stage-3 behavioral execution seal is required")
    for filename, digest in seal["files"].items():
        if file_sha256(Path(filename)) != digest:
            raise ValueError(f"sealed file changed: {filename}")
    if "scripts/run_l5_sst_like_behavioral_grid.py" not in seal["files"]:
        raise ValueError("runner is missing from execution seal")
    return seal


def run_registered_repetition(
    *,
    total_conductance_nS: float,
    delay_ms: float,
    **kwargs: Any,
) -> dict[str, object]:
    """Run one inherited behavioral repetition with one registered route."""

    original_builder_selector = legacy_runner._builder

    def point_builder_selector(*, wrapped: bool, base_builder: Any):
        if not wrapped:
            raise ValueError("the SST-like Stage-3 runner requires its wrapper path")
        return make_l5_sst_like_sector_builder(
            base_builder=base_builder,
            total_conductance_nS=total_conductance_nS,
            delay_ms=delay_ms,
        )

    try:
        legacy_runner._builder = point_builder_selector
        return legacy_runner._run_repetition(wrapped=True, **kwargs)
    finally:
        legacy_runner._builder = original_builder_selector


def classify_point(
    outcomes: list[dict[str, Any]], legacy_reference: dict[str, Any]
) -> dict[str, Any]:
    """Apply the registered exact-repeat, behavior and legacy rules."""

    if len(outcomes) != REPETITIONS:
        raise ValueError("point classification requires exactly two repetitions")
    normalized = [_without_repetition(outcome) for outcome in outcomes]
    exact_repeat = normalized[0] == normalized[1]
    legacy_exact = all(outcome == legacy_reference for outcome in normalized)
    all_behavioral_gates_pass = all(
        outcome.get("behavioral_pass") is True for outcome in outcomes
    )
    if not exact_repeat or not all_behavioral_gates_pass:
        classification = "failure"
    elif legacy_exact:
        classification = "exact_survival"
    else:
        classification = "robust_changed_survival"
    return {
        "exact_repeat": exact_repeat,
        "legacy_exact": legacy_exact,
        "all_behavioral_gates_pass": all_behavioral_gates_pass,
        "differences_from_legacy": recursive_differences(
            legacy_reference, normalized[0]
        ),
        "classification": classification,
    }


def classify_region(points: list[dict[str, Any]]) -> str:
    """Apply the preregistered complete-region classification."""

    if len(points) != len(registered_points()):
        return "indeterminate"
    classes = [point.get("classification") for point in points]
    if any(value not in {"exact_survival", "robust_changed_survival", "failure"} for value in classes):
        return "indeterminate"
    if all(value == "exact_survival" for value in classes):
        return "invariant_region"
    if all(value in {"exact_survival", "robust_changed_survival"} for value in classes):
        return "robust_region"
    if all(value == "failure" for value in classes):
        return "no_survival_in_registered_region"
    return "mixed_region"


def validate_checkpoint_points(points: list[dict[str, Any]]) -> None:
    """Reject extra, reordered or malformed checkpoint coordinates."""

    expected = registered_points()
    if len(points) > len(expected):
        raise ValueError("checkpoint contains unregistered grid points")
    identity_keys = (
        "point_index",
        "point_id",
        "delay_ms",
        "resource_fraction",
        "total_conductance_nS",
    )
    for index, point in enumerate(points):
        if any(point.get(key) != expected[index][key] for key in identity_keys):
            raise ValueError("checkpoint violates registered point order or identity")
        outcomes = point.get("outcomes")
        if not isinstance(outcomes, list) or len(outcomes) > REPETITIONS:
            raise ValueError("checkpoint has an invalid repetition inventory")
        for repetition, outcome in enumerate(outcomes):
            if outcome.get("repetition") != repetition:
                raise ValueError("checkpoint violates registered repetition order")
        classified = point.get("classification") is not None
        if classified and len(outcomes) != REPETITIONS:
            raise ValueError("checkpoint classifies an incomplete point")
        if index < len(points) - 1 and not classified:
            raise ValueError("checkpoint advances beyond an incomplete point")


def _learned_weights(path: Path) -> dict[str, Any]:
    figure6 = yaml.safe_load(path.read_text())
    source_trials = figure6["arms"]["ratio_0"]["trials"]
    keys = ("bottom_up_weights", "top_down_wide_weights", "top_down_narrow_weights")
    if len(source_trials) != REPETITIONS or any(
        source_trials[0][key] != source_trials[1][key] for key in keys
    ):
        raise ValueError("sealed ratio-zero Figure-6 weights are not exact repeats")
    return {
        "modeldb112923.projection.035": source_trials[0]["bottom_up_weights"],
        "modeldb112923.projection.005": source_trials[0]["top_down_wide_weights"],
        "modeldb112923.projection.007": source_trials[0]["top_down_narrow_weights"],
    }


def execute(output: Path, seal_path: Path) -> None:
    import brian2 as brian

    seal = verify_seal(seal_path)
    baseline = load_frozen_classic_baseline(seal["baseline_manifest"])
    legacy_path = Path(seal["legacy_result"])
    legacy_reference = load_legacy_reference(
        legacy_path, seal["legacy_result_sha256"]
    )
    figure6_path = Path(seal["figure6_result"])
    learned_weights = _learned_weights(figure6_path)
    manifest = yaml.safe_load(Path(seal["baseline_manifest"]).read_text())
    profile = yaml.safe_load(
        (baseline.repository_root / manifest["implementation"]["profile"]["path"]).read_text()
    )
    identity = {
        "seal_sha256": file_sha256(seal_path),
        "baseline_manifest_fingerprint": baseline.manifest_fingerprint,
        "runtime_fingerprint": baseline.runtime_fingerprint,
        "legacy_result_sha256": file_sha256(legacy_path),
        "figure6_result_sha256": file_sha256(figure6_path),
        "registered_points": registered_points(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "brian2": brian.__version__,
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "running-l5-sst-like-behavioral-grid",
        "identity": identity,
        "points": [],
        "region_classification": "indeterminate",
        "all_points_reported": False,
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
    if payload.get("status") == "completed-l5-sst-like-behavioral-grid":
        if classify_region(points) != payload.get("region_classification"):
            raise ValueError("completed checkpoint has an invalid region classification")
        return

    brian.prefs.codegen.target = "numpy"
    expected_points = registered_points()
    for point_index, registered in enumerate(expected_points):
        if point_index == len(points):
            point: dict[str, Any] = {
                **registered,
                "outcomes": [],
                "exact_repeat": None,
                "legacy_exact": None,
                "all_behavioral_gates_pass": None,
                "differences_from_legacy": None,
                "classification": None,
            }
            points.append(point)
        else:
            point = points[point_index]
        outcomes = point["outcomes"]
        for repetition in range(len(outcomes), REPETITIONS):
            outcomes.append(
                run_registered_repetition(
                    total_conductance_nS=float(point["total_conductance_nS"]),
                    delay_ms=float(point["delay_ms"]),
                    learned_weights=learned_weights,
                    conventions=baseline.runtime_conventions(),
                    scales=dict(baseline.projection_weight_scales),
                    profile=profile,
                    repetition=repetition,
                    brian=brian,
                )
            )
            checkpoint(output, payload)
        point.update(classify_point(outcomes, legacy_reference))
        checkpoint(output, payload)

    payload["all_points_reported"] = len(points) == len(expected_points)
    payload["region_classification"] = classify_region(points)
    payload["status"] = "completed-l5-sst-like-behavioral-grid"
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
            raise RuntimeError("an identical L5 SST-like Stage-3 runner is active") from error
        execute(output, args.seal.resolve())


if __name__ == "__main__":
    main()
