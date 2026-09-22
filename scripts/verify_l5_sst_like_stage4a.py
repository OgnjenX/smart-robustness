"""Independently verify a completed sealed L5 SST-like Stage-4A result."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import run_l5_sst_like_stage4a as runner
import yaml

from smart_robustness.validation.active_apical_recording import file_sha256


def _failed_gates(outcome: dict[str, Any], figure: str) -> list[str]:
    result = outcome.get(figure)
    if not isinstance(result, dict):
        return []
    gates = result.get("gates")
    if not isinstance(gates, dict):
        return []
    return [name for name, passed in gates.items() if passed is False]


def verify_result(result_path: Path, seal_path: Path) -> dict[str, Any]:
    """Return independently recomputed evidence or reject an invalid result."""

    seal = runner.verify_seal(seal_path)
    raw = result_path.read_bytes()
    result = yaml.safe_load(raw)
    if not isinstance(result, dict):
        raise TypeError("result payload must be a mapping")
    if result.get("status") != "completed-l5-sst-like-stage4a":
        raise ValueError("result is not terminal")
    if result.get("all_points_reported") is not True:
        raise ValueError("result does not report every registered point")
    if result.get("network_execution") is not True:
        raise ValueError("result is not a network execution")
    if result.get("frozen_baseline_modified") is not False:
        raise ValueError("result does not preserve the frozen baseline")

    identity = result.get("identity")
    if not isinstance(identity, dict):
        raise TypeError("result identity must be a mapping")
    expected_identity = {
        "seal_sha256": file_sha256(seal_path),
        "stage4_registration_sha256": seal["stage4_registration_sha256"],
        "stage3_assessment_sha256": seal["stage3_assessment_sha256"],
        "baseline_manifest_fingerprint": seal["baseline_manifest_fingerprint"],
        "runtime_fingerprint": seal["runtime_fingerprint"],
        "registered_points": runner.registered_points(),
    }
    for name, expected in expected_identity.items():
        if identity.get(name) != expected:
            raise ValueError(f"result identity mismatch: {name}")

    points = result.get("points")
    if not isinstance(points, list):
        raise TypeError("result points must be a list")
    runner.validate_checkpoint_points(points)
    if len(points) != len(runner.registered_points()):
        raise ValueError("result point inventory is incomplete")

    point_evidence: list[dict[str, Any]] = []
    for point in points:
        independently_computed = runner.classify_point(point["outcomes"])
        for name, expected in independently_computed.items():
            if point.get(name) != expected:
                raise ValueError(
                    f"stored point assessment mismatch: {point['point_id']}.{name}"
                )
        point_evidence.append(
            {
                "point_id": point["point_id"],
                "delay_ms": point["delay_ms"],
                "resource_fraction": point["resource_fraction"],
                "total_conductance_nS": point["total_conductance_nS"],
                "classification": point["classification"],
                "exact_repeat": point["exact_repeat"],
                "all_progression_gates_pass": point[
                    "all_progression_gates_pass"
                ],
                "figure6_failed_gates": [
                    _failed_gates(outcome, "figure6") for outcome in point["outcomes"]
                ],
                "figure7_failed_gates": [
                    _failed_gates(outcome, "figure7") for outcome in point["outcomes"]
                ],
                "figure10_failed_gates": [
                    _failed_gates(outcome, "figure10") for outcome in point["outcomes"]
                ],
            }
        )

    counts = Counter(point["classification"] for point in points)
    classification_counts = dict(sorted(counts.items()))
    if result.get("classification_counts") != classification_counts:
        raise ValueError("stored classification counts mismatch")
    return {
        "raw_result": str(result_path),
        "raw_result_sha256": file_sha256(result_path),
        "raw_result_bytes": len(raw),
        "seal": str(seal_path),
        "seal_sha256": file_sha256(seal_path),
        "completed_points": len(points),
        "completed_repetitions": sum(len(point["outcomes"]) for point in points),
        "classification_counts": classification_counts,
        "all_points_exact_repeat": all(point["exact_repeat"] for point in points),
        "all_points_survive_stage4a": all(
            point["classification"] == "learning_first_order_survival"
            for point in points
        ),
        "point_evidence": point_evidence,
        "independent_verification_passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--seal", type=Path, default=runner.DEFAULT_SEAL)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    evidence = verify_result(args.result.resolve(), args.seal.resolve())
    rendered = yaml.safe_dump(evidence, sort_keys=False)
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.write_text(rendered)


if __name__ == "__main__":
    main()
