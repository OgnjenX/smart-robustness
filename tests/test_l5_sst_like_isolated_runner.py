from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import yaml

from smart_robustness.validation.l5_sst_like_isolated import (
    CLAMP_MV,
    EMISSIONS_MS,
    RESOURCE_ANCHOR_NS,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_l5_sst_like_isolated_recruitment.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("l5_sst_like_isolated_runner", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_arrays(*, total_conductance_nS: float, delay_ms: float, dt_ms: float):
    time = np.array([0.0, 1.0, 2.0])
    return {
        "time_ms": time,
        "source_spike_times_ms": EMISSIONS_MS.copy(),
        "source_spike_indices": np.zeros(5, dtype=np.int64),
        "target_spike_times_ms": np.array([], dtype=float),
        "target_spike_indices": np.array([], dtype=np.int64),
        "distal_gate": np.array([0.0, 1.0, 0.0]),
        "distal_synaptic_current_pA": np.array([0.0, -15.0, 0.0]),
        "v_soma_mV": np.full(3, CLAMP_MV),
        "v_proximal_dendrite_mV": np.full(3, CLAMP_MV),
        "v_distal_dendrite_mV": np.full(3, CLAMP_MV),
        "delivery_count": np.array([5], dtype=np.int64),
        "requested_total_conductance_nS": np.array(total_conductance_nS),
        "realized_total_conductance_nS": np.array(total_conductance_nS),
        "resource_fraction": np.array(total_conductance_nS / RESOURCE_ANCHOR_NS),
        "delay_ms": np.array(delay_ms),
        "dt_ms": np.array(dt_ms),
    }


def test_registered_run_order_covers_exactly_48_conditions() -> None:
    runner = load_runner()
    runs = list(runner.registered_runs())
    keys = {
        runner.run_key(
            delay_ms=run["delay_ms"],
            resource_fraction=run["resource_fraction"],
            dt_ms=run["dt_ms"],
            repetition=run["repetition"],
        )
        for run in runs
    }
    assert len(runs) == len(keys) == 48
    assert runs[0]["delay_ms"] == 1.0
    assert runs[0]["resource_fraction"] == 0.125
    assert runs[-1]["delay_ms"] == 7.0
    assert runs[-1]["resource_fraction"] == 1.0


def test_runner_rejects_unsealed_status(tmp_path: Path) -> None:
    runner = load_runner()
    seal = tmp_path / "seal.yaml"
    seal.write_text(yaml.safe_dump({"status": "draft", "files": {}}))
    with pytest.raises(ValueError, match="execution seal"):
        runner.verify_seal(seal)


def test_synthetic_complete_resume_and_tamper_detection(tmp_path: Path, monkeypatch):
    runner = load_runner()
    calls = []

    def fake_simulate(**kwargs):
        calls.append(kwargs.copy())
        kwargs.pop("baseline")
        return fake_arrays(**kwargs)

    monkeypatch.setattr(runner, "simulate", fake_simulate)
    monkeypatch.setattr(
        runner,
        "verify_seal",
        lambda _: {"baseline_manifest": "configs/baselines/classic_smart_calibrated_v1.yaml"},
    )
    seal = tmp_path / "seal.yaml"
    seal.write_text("synthetic-only")
    output = tmp_path / "result.yaml"
    runner.execute(output, seal)
    data = yaml.safe_load(output.read_text())
    assert len(calls) == 48
    assert data["status"] == "completed-l5-sst-like-isolated-recruitment"
    assert data["assessment"]["isolated_promotion_gates_pass"] is True
    assert all(data["assessment"]["exact_raw_repeats"].values())
    assert all(
        gate["pass"] for gate in data["assessment"]["numerical_convergence"].values()
    )
    runner.execute(output, seal)
    assert len(calls) == 48
    first = next(iter(data["runs"].values()))
    first["metrics"]["delivery_count"] = 4
    output.write_text(yaml.safe_dump(data, sort_keys=False))
    with pytest.raises(ValueError, match="metrics"):
        runner.execute(output, seal)
