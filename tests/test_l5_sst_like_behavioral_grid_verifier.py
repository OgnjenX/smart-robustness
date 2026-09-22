from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_l5_sst_like_behavioral_grid.py"


def load_verifier():
    spec = importlib.util.spec_from_file_location("l5_sst_stage3_verifier", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SCRIPT.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def completed_payload(verifier, *, classification: str = "exact_survival") -> dict:
    points = []
    for registered in verifier.runner.registered_points():
        outcomes = [
            {
                "repetition": repetition,
                "behavioral_pass": True,
                "figure7": {"pass": True, "gates": {"match": True}},
                "figure10": {"pass": True, "gates": {"reset": True}},
            }
            for repetition in range(2)
        ]
        points.append(
            {
                **registered,
                "outcomes": outcomes,
                "exact_repeat": True,
                "legacy_exact": classification == "exact_survival",
                "all_behavioral_gates_pass": True,
                "differences_from_legacy": [],
                "classification": classification,
            }
        )
    return {
        "status": "completed-l5-sst-like-behavioral-grid",
        "identity": {},
        "points": points,
        "region_classification": (
            "invariant_region" if classification == "exact_survival" else "robust_region"
        ),
        "all_points_reported": True,
        "network_execution": True,
        "frozen_baseline_modified": False,
    }


def install_fakes(monkeypatch, verifier, payload: dict) -> None:
    seal = {
        "baseline_manifest_fingerprint": "baseline",
        "runtime_fingerprint": "runtime",
        "legacy_result_sha256": "legacy",
        "figure6_result_sha256": "figure6",
        "legacy_result": "legacy.yaml",
    }
    monkeypatch.setattr(verifier.runner, "verify_seal", lambda path: seal)
    monkeypatch.setattr(verifier, "file_sha256", lambda path: f"hash:{Path(path).name}")
    monkeypatch.setattr(
        verifier.runner, "load_legacy_reference", lambda path, digest: {"legacy": True}
    )

    expected_in_order = [
        {
            name: point[name]
            for name in (
                "exact_repeat",
                "legacy_exact",
                "all_behavioral_gates_pass",
                "differences_from_legacy",
                "classification",
            )
        }
        for point in payload["points"]
    ]
    calls = iter(expected_in_order)

    def classify(outcomes, legacy):
        return next(calls)

    monkeypatch.setattr(verifier.runner, "classify_point", classify)


def write_payload(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "result.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


def test_verifier_recomputes_complete_registered_result(monkeypatch, tmp_path: Path) -> None:
    verifier = load_verifier()
    payload = completed_payload(verifier)
    install_fakes(monkeypatch, verifier, payload)
    payload["identity"] = {
        "seal_sha256": "hash:seal.yaml",
        "baseline_manifest_fingerprint": "baseline",
        "runtime_fingerprint": "runtime",
        "legacy_result_sha256": "legacy",
        "figure6_result_sha256": "figure6",
        "registered_points": verifier.runner.registered_points(),
    }
    result = verifier.verify_result(write_payload(tmp_path, payload), tmp_path / "seal.yaml")
    assert result["independent_verification_passed"] is True
    assert result["completed_points"] == 12
    assert result["completed_repetitions"] == 24
    assert result["classification_counts"] == {"exact_survival": 12}
    assert result["region_classification"] == "invariant_region"
    assert result["all_points_exact_repeat"] is True


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.update(status="running"), "not terminal"),
        (
            lambda payload: payload["points"].pop(),
            "point inventory is incomplete",
        ),
        (
            lambda payload: payload["points"][0].update(exact_repeat=False),
            "stored point assessment mismatch",
        ),
        (
            lambda payload: payload.update(region_classification="mixed_region"),
            "stored region classification mismatch",
        ),
    ],
)
def test_verifier_rejects_incomplete_or_inconsistent_results(
    monkeypatch, tmp_path: Path, mutation, message: str
) -> None:
    verifier = load_verifier()
    payload = completed_payload(verifier)
    install_fakes(monkeypatch, verifier, payload)
    payload["identity"] = {
        "seal_sha256": "hash:seal.yaml",
        "baseline_manifest_fingerprint": "baseline",
        "runtime_fingerprint": "runtime",
        "legacy_result_sha256": "legacy",
        "figure6_result_sha256": "figure6",
        "registered_points": verifier.runner.registered_points(),
    }
    mutation(payload)
    with pytest.raises(ValueError, match=message):
        verifier.verify_result(write_payload(tmp_path, payload), tmp_path / "seal.yaml")
