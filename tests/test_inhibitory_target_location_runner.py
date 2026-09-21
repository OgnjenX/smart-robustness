from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_inhibitory_target_location_isolated.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("inhibitory_target_runner", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_keys_cover_exact_registered_grid() -> None:
    runner = load_runner()
    keys = {
        runner.run_key(protocol, arm, dt_ms, repetition)
        for protocol in runner.PROTOCOLS
        for arm in runner.ARMS
        for dt_ms in runner.DT_MS
        for repetition in range(2)
    }
    assert len(keys) == 24


def test_runner_rejects_unsealed_status(tmp_path: Path) -> None:
    runner = load_runner()
    seal = tmp_path / "seal.yaml"
    seal.write_text(
        yaml.safe_dump(
            {
                "status": "implementation-complete-not-sealed",
                "files": {},
            }
        )
    )
    with pytest.raises(ValueError, match="execution seal"):
        runner.verify_seal(seal)
