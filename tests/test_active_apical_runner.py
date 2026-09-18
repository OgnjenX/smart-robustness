"""Test runner safeguards without executing biological response simulations."""

import runpy
from pathlib import Path

import numpy as np
import pytest
import yaml


@pytest.fixture
def runner():
    return runpy.run_path(str(Path(__file__).parents[1] / "scripts/run_active_apical_isolated.py"))


def test_atomic_checkpoint_replacement(runner, tmp_path):
    path = tmp_path / "checkpoint.yaml"
    runner["checkpoint"](path, {"stage": 1})
    runner["checkpoint"](path, {"stage": 2})
    assert yaml.safe_load(path.read_text()) == {"stage": 2}
    assert not list(tmp_path.glob(".apical-checkpoint-*"))


def test_unsealed_execution_rejected(runner, tmp_path):
    path = tmp_path / "seal.yaml"
    path.write_text(yaml.safe_dump({"status": "draft"}))
    with pytest.raises(ValueError, match="seal"):
        runner["verify_seal"](path)


def test_seal_detects_changed_file_and_missing_runner(runner, tmp_path):
    source = tmp_path / "source.py"
    source.write_text("value = 1\n")
    seal = tmp_path / "seal.yaml"
    data = {
        "status": "sealed-before-isolated-outcomes",
        "files": {str(source): runner["file_sha256"](source)},
    }
    seal.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="runner"):
        runner["verify_seal"](seal)
    source.write_text("value = 2\n")
    with pytest.raises(ValueError, match="changed"):
        runner["verify_seal"](seal)


def test_complete_synthetic_execution_resume_and_tamper_detection(runner, tmp_path, monkeypatch):
    """No biological integration: controlled constant traces exercise the full pipeline."""
    calls = []

    def fake_simulate(*, dt_ms, rest_only=False, **kwargs):
        calls.append((kwargs["arm"], dt_ms, rest_only))
        duration = 100.0 if rest_only else 500.0
        t = 900.0 + np.arange(round(duration / dt_ms)) * dt_ms
        arrays = {
            "time_ms": t,
            "spike_time_ms": np.array([]),
            "spike_cell_index": np.array([], dtype=np.int32),
            "dt_ms": np.array(dt_ms),
        }
        for compartment in ("soma", "proximal_dendrite", "distal_dendrite"):
            arrays[f"v_{compartment}_mV"] = np.full((24, len(t)), -65.0)
        return arrays

    execute = runner["execute"]
    monkeypatch.setitem(execute.__globals__, "simulate", fake_simulate)
    monkeypatch.setitem(
        execute.__globals__,
        "verify_seal",
        lambda _: {"baseline_manifest": "configs/baselines/classic_smart_calibrated_v1.yaml"},
    )
    seal = tmp_path / "synthetic-seal.yaml"
    seal.write_text("synthetic-test-only")
    output = tmp_path / "synthetic.yaml"
    execute(output, seal)
    data = yaml.safe_load(output.read_text())
    assert len(calls) == 13
    assert data["status"] == "completed-isolated-apical"
    assert all(data["assessment"]["exact_raw_repeats"].values())
    assert not data["assessment"]["isolated_promotion_gates_pass"]
    execute(output, seal)
    assert len(calls) == 13  # completed checkpoints are verified, not rerun
    first = next(iter(data["runs"].values()))
    first["metrics"][0]["soma_spike_count"] += 1
    output.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="metrics"):
        execute(output, seal)
