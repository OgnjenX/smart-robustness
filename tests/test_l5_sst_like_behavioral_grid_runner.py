from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_l5_sst_like_behavioral_grid.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("l5_sst_stage3_runner", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SCRIPT.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def outcome(*, repetition: int, passed: bool = True, value: int = 1) -> dict:
    return {
        "repetition": repetition,
        "behavioral_pass": passed,
        "figure7": {"pass": passed, "value": value},
        "figure10": {"pass": passed} if passed else None,
    }


def test_registered_points_are_complete_delay_major_and_exact() -> None:
    runner = load_runner()
    points = runner.registered_points()
    assert len(points) == 12
    assert [(point["delay_ms"], point["resource_fraction"]) for point in points] == [
        (delay, fraction)
        for delay in (1.0, 3.0, 7.0)
        for fraction in (0.125, 0.25, 0.5, 1.0)
    ]
    assert all(
        point["total_conductance_nS"]
        == runner.RESOURCE_ANCHOR_NS * point["resource_fraction"]
        for point in points
    )


def test_registered_repetition_injects_only_requested_point(monkeypatch) -> None:
    runner = load_runner()
    original_selector = runner.legacy_runner._builder
    captured = {}

    def fake_run_repetition(*, wrapped, marker):
        assert wrapped is True

        def base_builder(**kwargs):
            captured["base_kwargs"] = kwargs
            return "sector"

        builder = runner.legacy_runner._builder(wrapped=True, base_builder=base_builder)
        captured["builder"] = builder
        return {"repetition": marker, "behavioral_pass": True}

    def fake_wrapper(*, base_builder, total_conductance_nS, delay_ms):
        captured["base_builder"] = base_builder
        captured["resource"] = total_conductance_nS
        captured["delay"] = delay_ms
        return lambda **kwargs: base_builder(**kwargs)

    monkeypatch.setattr(runner.legacy_runner, "_run_repetition", fake_run_repetition)
    monkeypatch.setattr(runner, "make_l5_sst_like_sector_builder", fake_wrapper)
    result = runner.run_registered_repetition(
        total_conductance_nS=48.32162200302801,
        delay_ms=3.0,
        marker=7,
    )
    assert result == {"repetition": 7, "behavioral_pass": True}
    assert captured["resource"] == 48.32162200302801
    assert captured["delay"] == 3.0
    assert runner.legacy_runner._builder is original_selector


def test_point_classifications_are_prospective() -> None:
    runner = load_runner()
    legacy = runner._without_repetition(outcome(repetition=0))
    exact = runner.classify_point(
        [outcome(repetition=0), outcome(repetition=1)], legacy
    )
    changed = runner.classify_point(
        [outcome(repetition=0, value=2), outcome(repetition=1, value=2)], legacy
    )
    failed_gate = runner.classify_point(
        [outcome(repetition=0, passed=False), outcome(repetition=1, passed=False)],
        legacy,
    )
    failed_repeat = runner.classify_point(
        [outcome(repetition=0), outcome(repetition=1, value=2)], legacy
    )
    assert exact["classification"] == "exact_survival"
    assert changed["classification"] == "robust_changed_survival"
    assert changed["differences_from_legacy"]
    assert failed_gate["classification"] == "failure"
    assert failed_repeat["classification"] == "failure"


@pytest.mark.parametrize(
    ("classes", "expected"),
    [
        (["exact_survival"] * 12, "invariant_region"),
        (["exact_survival"] * 11 + ["robust_changed_survival"], "robust_region"),
        (["exact_survival"] * 11 + ["failure"], "mixed_region"),
        (["failure"] * 12, "no_survival_in_registered_region"),
        (["exact_survival"] * 11, "indeterminate"),
    ],
)
def test_region_classification(classes: list[str], expected: str) -> None:
    runner = load_runner()
    points = [{"classification": value} for value in classes]
    assert runner.classify_region(points) == expected


def test_checkpoint_validation_rejects_reorder_and_skipped_repetition() -> None:
    runner = load_runner()
    points = runner.registered_points()
    first = {
        **points[0],
        "outcomes": [outcome(repetition=0), outcome(repetition=1)],
        "classification": "exact_survival",
    }
    second = {
        **points[1],
        "outcomes": [outcome(repetition=0)],
        "classification": None,
    }
    runner.validate_checkpoint_points([first, second])
    with pytest.raises(ValueError, match="order or identity"):
        runner.validate_checkpoint_points([second])
    malformed = dict(second)
    malformed["outcomes"] = [outcome(repetition=1)]
    with pytest.raises(ValueError, match="repetition order"):
        runner.validate_checkpoint_points([first, malformed])
    incomplete_first = {
        **points[0],
        "outcomes": [outcome(repetition=0)],
        "classification": None,
    }
    completed_second = {
        **points[1],
        "outcomes": [outcome(repetition=0), outcome(repetition=1)],
        "classification": "exact_survival",
    }
    with pytest.raises(ValueError, match="incomplete point"):
        runner.validate_checkpoint_points([incomplete_first, completed_second])


def test_checkpoint_after_second_repetition_before_classification_is_resumable() -> None:
    runner = load_runner()
    point = {
        **runner.registered_points()[0],
        "outcomes": [outcome(repetition=0), outcome(repetition=1)],
        "classification": None,
    }
    runner.validate_checkpoint_points([point])


def test_runner_rejects_unsealed_status(tmp_path: Path) -> None:
    runner = load_runner()
    seal = tmp_path / "seal.yaml"
    seal.write_text(yaml.safe_dump({"status": "draft", "files": {}}))
    with pytest.raises(ValueError, match="execution seal"):
        runner.verify_seal(seal)
