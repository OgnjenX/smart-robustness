"""Continuation controls use synthetic outcomes and execute no network assay."""

from __future__ import annotations

import importlib.util
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml


def load_script(filename):
    path = Path(__file__).parents[1] / "scripts" / filename
    spec = importlib.util.spec_from_file_location(filename.removesuffix(".py"), path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


@pytest.fixture
def setup_runner(monkeypatch, tmp_path):
    runner = load_script("run_l5_sst_like_stage4b_continuation.py")
    original = runner.parent.stage4a.registered_points()
    points = []
    for point in original[:5]:
        outcomes = [{"repetition": r, "figure14_pass": True, "events": 3} for r in range(2)]
        points.append(
            {
                **point,
                "stage4a_classification": "learning_first_order_survival",
                "outcomes": outcomes,
                **runner.parent.classify_point(outcomes),
            }
        )
    source = {"identity": {"protocol": {}}, "points": points}
    registration = {
        "result": str(tmp_path / "result.yaml"),
        "old_environment": {"platform": "old"},
        "new_environment": {"platform": "new"},
    }
    learning = {"points": [{"classification": "learning_first_order_survival"} for _ in original]}
    learning["points"][-1]["classification"] = "failure"
    baseline = SimpleNamespace(
        runtime_conventions=lambda: "conventions", projection_weight_scales={}
    )
    monkeypatch.setattr(runner, "verify_seal", lambda path: None)
    monkeypatch.setattr(
        runner, "load_inputs", lambda: (registration, source, baseline, learning, {})
    )
    monkeypatch.setattr(runner, "continuation_identity", lambda reg, seal: {"sealed": True})
    monkeypatch.setattr(runner, "_check_outcome", lambda outcome, protocol: None)
    monkeypatch.setattr(runner.bridge, "environment", lambda: registration["new_environment"])
    monkeypatch.setattr(runner.parent, "learned_weights_for_repetition", lambda d, i, r: (i, r))
    calls = []

    def fake_run(**kwargs):
        calls.append(
            (kwargs["point"]["stage4a_index"], kwargs["repetition"], kwargs["learned_weights"])
        )
        return {"repetition": kwargs["repetition"], "figure14_pass": True, "events": 3}

    monkeypatch.setattr(runner.parent, "run_registered_repetition", fake_run)
    return runner, registration, source, learning, calls


def test_continuation_runs_only_four_remaining_repetitions_and_preserves_failure(setup_runner):
    runner, registration, source, _learning, calls = setup_runner
    original = deepcopy(source)
    output = Path(registration["result"])
    runner.execute(output, Path("seal"))
    result = yaml.safe_load(output.read_text())
    assert calls == [(5, 0, (5, 0)), (5, 1, (5, 1)), (6, 0, (6, 0)), (6, 1, (6, 1))]
    assert source == original
    assert result["points"][:5] == original["points"]
    assert result["points"][-1]["stage4a_classification"] == "failure"
    assert result["status"] == "completed-stage4b-continuation"
    assert len(result["execution_environments"]) == 14
    assert all(
        x["environment"] == {"platform": "old"} for x in result["execution_environments"][:10]
    )
    assert all(
        x["environment"] == {"platform": "new"} for x in result["execution_environments"][10:]
    )
    runner.execute(output, Path("seal"))
    assert len(calls) == 4


def test_checkpoint_resumes_after_one_new_repetition(setup_runner, monkeypatch):
    runner, registration, _source, _learning, calls = setup_runner
    output = Path(registration["result"])
    real = runner.parent.run_registered_repetition

    def stop_second(**kwargs):
        if kwargs["repetition"] == 1:
            raise RuntimeError("interrupted")
        return real(**kwargs)

    monkeypatch.setattr(runner.parent, "run_registered_repetition", stop_second)
    with pytest.raises(RuntimeError, match="interrupted"):
        runner.execute(output, Path("seal"))
    checkpoint = yaml.safe_load(output.read_text())
    assert sum(len(p["outcomes"]) for p in checkpoint["points"]) == 11
    monkeypatch.setattr(runner.parent, "run_registered_repetition", real)
    runner.execute(output, Path("seal"))
    assert [c[:2] for c in calls] == [(5, 0), (5, 1), (6, 0), (6, 1)]


@pytest.mark.parametrize("corruption", ["original", "environment"])
def test_resume_rejects_changed_old_outcome_or_environment(setup_runner, corruption):
    runner, registration, _source, _learning, calls = setup_runner
    output = Path(registration["result"])
    runner.execute(output, Path("seal"))
    result = yaml.safe_load(output.read_text())
    if corruption == "original":
        result["points"][0]["outcomes"][0]["events"] = 4
    else:
        result["execution_environments"][0]["environment"] = {"platform": "new"}
    output.write_text(yaml.safe_dump(result))
    with pytest.raises(ValueError, match="changed"):
        runner.execute(output, Path("seal"))
    assert len(calls) == 4


def test_exact_repeat_failure_stops_before_last_point(setup_runner, monkeypatch):
    runner, registration, _source, _learning, calls = setup_runner
    real = runner.parent.run_registered_repetition

    def unequal(**kwargs):
        outcome = real(**kwargs)
        outcome["events"] += kwargs["repetition"]
        return outcome

    monkeypatch.setattr(runner.parent, "run_registered_repetition", unequal)
    with pytest.raises(RuntimeError, match="exact-repeat"):
        runner.execute(Path(registration["result"]), Path("seal"))
    assert [c[:2] for c in calls] == [(5, 0), (5, 1)]
    assert (
        yaml.safe_load(Path(registration["result"]).read_text())["points"][-1]["classification"]
        == "engineering_stop"
    )


def test_independent_verifier_rejects_false_classification(monkeypatch):
    verifier = load_script("verify_l5_sst_like_stage4b_continuation.py")
    monkeypatch.setattr(verifier, "_check_outcome", lambda outcome, protocol: {})
    point = {
        "point_id": "example",
        "outcomes": [{"repetition": r, "figure14_pass": False} for r in range(2)],
        "exact_repeat": True,
        "both_figure14_gate_sets_pass": True,
        "classification": "figure14_survival",
    }
    with pytest.raises(ValueError, match="contradicts"):
        verifier.check_point(point, {})
    point.update(both_figure14_gate_sets_pass=False, classification="figure14_failure")
    assert verifier.check_point(point, {})["classification"] == "figure14_failure"
