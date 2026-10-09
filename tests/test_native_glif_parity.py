"""Fixed gates/fixtures and checkpoint tests do not execute downloaded native code."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from smart_robustness.validation.native_glif_parity import compare, exact_repeat, fixtures, stimuli
from smart_robustness.validation.native_glif_state_machine import simulate_glif


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    name = "glif_parity_runner_test"
    spec = importlib.util.spec_from_file_location(name, Path("scripts/run_sst_vip_glif_parity.py"))
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def test_fixed_fixture_and_stimulus_counts():
    cases = fixtures()
    assert len(cases) == 10 and len({c["identity"] for c in cases}) == 10
    assert {c["family"] for c in cases} == {"LIF", "LIF-R", "LIF-ASC", "LIF-R-ASC", "LIF-R-ASC-A"}
    for case in cases:
        currents = stimuli(case["parameters"])
        assert list(currents) == ["zero", "subthreshold", "step", "pulses"]
        assert all(v.shape == (2000,) for v in currents.values())
        assert np.all(currents["step"][:200] == 0)
        assert np.all(currents["pulses"][100:200] == 0)


def test_gate_rejects_spike_nan_and_infinite_corruptions():
    p = fixtures()[0]["parameters"]
    data, _ = simulate_glif(p, np.zeros(10))
    assert compare(data, data)["passed"] and exact_repeat(data, data)
    for key, value in (("voltage", np.nan), ("threshold", np.inf), ("voltage", 1.0)):
        changed = {k: v.copy() for k, v in data.items()}
        changed[key][0] = value
        assert not compare(data, changed)["passed"]
    changed = {k: v.copy() for k, v in data.items()}
    changed["spike_time_steps"] = np.array([1])
    assert not compare(data, changed)["passed"]


def test_exception_is_recorded_not_a_pass(runner):
    def broken(*args):
        raise RuntimeError("native stand-in failed")

    report, _ = runner.evaluate_case(fixtures()[0], np.zeros(10), broken)
    assert not report["passed"]
    assert set(report["errors"]) == {"native0", "native1"}


def test_bad_reset_is_not_a_pass_even_with_matching_arrays(runner):
    def stopped(p, current):
        data, _ = simulate_glif(p, current)
        return data, {"bad_reset_stop": True}

    report, _ = runner.evaluate_case(fixtures()[0], np.zeros(10), stopped)
    assert report["comparison"]["passed"] and not report["passed"]


def test_checkpoint_resume_does_not_execute_reference(runner, tmp_path):
    case = fixtures()[0]
    current = np.zeros(10)
    saved = runner.run_case(
        0,
        case,
        "unit-only",
        current,
        directory=tmp_path,
        context={"seal": "test"},
        reference=simulate_glif,
    )

    def forbidden(*args):
        pytest.fail("resume must not rerun reference")

    repeated = runner.run_case(
        0,
        case,
        "unit-only",
        current,
        directory=tmp_path,
        context={"seal": "test"},
        reference=forbidden,
    )
    assert repeated == saved
    path = tmp_path / "case-0000.yaml"
    modified = yaml.safe_load(path.read_text())
    modified["report"]["passed"] = not modified["report"]["passed"]
    with path.open("w") as handle:
        yaml.safe_dump(modified, handle)
    with pytest.raises(ValueError, match="report"):
        runner.run_case(
            0,
            case,
            "unit-only",
            current,
            directory=tmp_path,
            context={"seal": "test"},
            reference=forbidden,
        )


def test_changed_context_and_orphan_arrays_rejected(runner, tmp_path):
    case = fixtures()[0]
    current = np.zeros(10)
    runner.run_case(
        0,
        case,
        "unit-only",
        current,
        directory=tmp_path,
        context={"seal": "test"},
        reference=simulate_glif,
    )
    with pytest.raises(ValueError, match="checkpoint"):
        runner.run_case(
            0,
            case,
            "unit-only",
            current,
            directory=tmp_path,
            context={"seal": "other"},
            reference=simulate_glif,
        )
    with (tmp_path / "case-0001.npz").open("xb") as handle:
        handle.write(b"preserve orphan")
    with pytest.raises(ValueError, match="orphan"):
        runner.run_case(
            1,
            case,
            "unit-only",
            current,
            directory=tmp_path,
            context={"seal": "test"},
            reference=simulate_glif,
        )
