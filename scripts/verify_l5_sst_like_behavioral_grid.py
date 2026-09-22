"""Independently verify a completed sealed L5 SST-like Stage-3 result."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import run_l5_sst_like_behavioral_grid as runner
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
    if result.get("status") != "completed-l5-sst-like-behavioral-grid":
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
        "baseline_manifest_fingerprint": seal["baseline_manifest_fingerprint"],
        "runtime_fingerprint": seal["runtime_fingerprint"],
        "legacy_result_sha256": seal["legacy_result_sha256"],
        "figure6_result_sha256": seal["figure6_result_sha256"],
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

    legacy_reference = runner.load_legacy_reference(
        Path(seal["legacy_result"]), seal["legacy_result_sha256"]
    )
    point_evidence: list[dict[str, Any]] = []
    for point in points:
        outcomes = point["outcomes"]
        independently_computed = runner.classify_point(outcomes, legacy_reference)
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
                "legacy_exact": point["legacy_exact"],
                "all_behavioral_gates_pass": point["all_behavioral_gates_pass"],
                "figure7_failed_gates": [
                    _failed_gates(outcome, "figure7") for outcome in outcomes
                ],
                "figure10_failed_gates": [
                    _failed_gates(outcome, "figure10") for outcome in outcomes
                ],
                "differences_from_legacy": point["differences_from_legacy"],
            }
        )

    region_classification = runner.classify_region(points)
    if result.get("region_classification") != region_classification:
        raise ValueError("stored region classification mismatch")
    counts = Counter(point["classification"] for point in points)
    return {
        "raw_result": str(result_path),
        "raw_result_sha256": file_sha256(result_path),
        "raw_result_bytes": len(raw),
        "seal": str(seal_path),
        "seal_sha256": file_sha256(seal_path),
        "completed_points": len(points),
        "completed_repetitions": sum(len(point["outcomes"]) for point in points),
        "classification_counts": dict(sorted(counts.items())),
        "region_classification": region_classification,
        "all_points_exact_repeat": all(point["exact_repeat"] for point in points),
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
