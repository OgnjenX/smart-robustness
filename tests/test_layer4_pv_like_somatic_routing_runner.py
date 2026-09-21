from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

from smart_robustness.baseline import load_frozen_classic_baseline

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_layer4_pv_like_somatic_routing.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("layer4_pv_routing_runner", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SCRIPT.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def test_structural_summary_proves_one_factor_and_exact_resource() -> None:
    runner = load_runner()
    baseline = load_frozen_classic_baseline(
        ROOT / "configs/baselines/classic_smart_calibrated_v1.yaml"
    )
    summary = runner.structural_summary(baseline.runtime_conventions())
    assert summary["changed_projection_ids"] == [
        "modeldb112923.projection.036"
    ]
    assert summary["old_compartment"] == "proximal_dendrite"
    assert summary["new_compartment"] == "soma"
    assert summary["total_port_conductance_exact"] is True
    assert summary["all_non_port_parameters_equal"] is True


def test_transformed_builder_injects_only_registered_population_factory() -> None:
    runner = load_runner()
    calls = []

    def base_builder(**kwargs):
        calls.append(kwargs)
        return "sector"

    builder = runner.transformed_builder(base_builder)
    assert builder(example=4) == "sector"
    assert calls[0]["example"] == 4
    assert callable(calls[0]["population_factory"])
    with pytest.raises(TypeError, match="keyword arguments"):
        builder(1)
    with pytest.raises(ValueError, match="nested"):
        builder(population_factory=lambda **_: None)


def test_recursive_differences_reports_every_changed_leaf() -> None:
    runner = load_runner()
    differences = runner.recursive_differences(
        {"a": 1, "b": {"c": [2, 3]}},
        {"a": 4, "b": {"c": [2, 5]}},
    )
    assert differences == [
        {"path": "a", "legacy": 1, "transformed": 4},
        {"path": "b.c[1]", "legacy": 3, "transformed": 5},
    ]


def test_runner_rejects_unsealed_status(tmp_path: Path) -> None:
    runner = load_runner()
    seal = tmp_path / "seal.yaml"
    seal.write_text(yaml.safe_dump({"status": "draft", "files": {}}))
    with pytest.raises(ValueError, match="execution seal"):
        runner.verify_seal(seal)
