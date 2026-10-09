"""Synthetic-only continuation contracts; no replay of the preserved network result."""

from __future__ import annotations

import fcntl
import importlib
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from smart_robustness.validation import l5_sst_like_stage4d as contract
from smart_robustness.validation.higher_order import Figure16Protocol

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    return (
        importlib.import_module("run_l5_sst_like_stage4d_continuation"),
        importlib.import_module("verify_l5_sst_like_stage4d_continuation"),
    )


def test_amended_raw_check_preserves_normalized_and_gate_checks(modules):
    _, verifier = modules
    from test_l5_sst_like_stage4d_runner import synthetic_outcome

    data = synthetic_outcome()
    verifier.check_outcome(data)
    value = deepcopy(data)
    raw = value["cross_correlations"][-1]["raw"]
    index = int(np.argmin(np.abs(raw)))
    raw[index] += 1e-10
    with pytest.raises(ValueError, match="raw"):
        verifier.original.check_outcome(value)
    assert (
        verifier.check_outcome(value)["numerical_diagnostics"][-1]["original_dimensional_raw_pass"]
        is False
    )
    for mutation in ("raw", "normalized", "gate", "time", "seed"):
        value = deepcopy(data)
        if mutation == "raw":
            value["cross_correlations"][-1]["raw"][500] += 0.1
        elif mutation == "normalized":
            value["cross_correlations"][-1]["normalized"][500] += 1e-4
        elif mutation == "gate":
            value["gates"][contract.GATES[2]] = not value["gates"][contract.GATES[2]]
        elif mutation == "time":
            value["sample_times_ms"][0] = 0
        else:
            value["fields"]["v2"]["seed"] = 18
        with pytest.raises(ValueError):
            verifier.check_outcome(value)


def synthetic_inputs(runner):
    points = runner.original.parent.parent.stage4a.registered_points()
    learning = {"points": [{"classification": "learning_first_order_survival"} for _ in points]}
    learning["points"][-1]["classification"] = "failure"
    spectra = {"points": [{"classification": "figure14_survival"} for _ in points]}
    spectra["points"][0]["classification"] = "figure14_failure"
    synchrony = {"points": [{"classification": "figure15_survival"} for _ in points]}
    first = {
        **points[0],
        "classification": None,
        "outcomes": [{"repetition": 0, "gates": dict.fromkeys(contract.GATES, True)}],
        "earlier_classifications": runner.original.earlier_for(0, learning, spectra, synchrony),
    }
    return {
        "source": {"points": [first]},
        "learning": learning,
        "spectra": spectra,
        "synchrony": synchrony,
        "baseline": SimpleNamespace(
            runtime_conventions=lambda: "runtime", projection_weight_scales={}
        ),
        "protocol": Figure16Protocol(),
    }


def test_resume_copies_source_verbatim_retains_failures_and_terminal_noop(
    modules, monkeypatch, tmp_path
):
    runner, _ = modules
    inputs = synthetic_inputs(runner)
    pristine = deepcopy(inputs["source"])
    output = tmp_path / "continuation.yaml"
    monkeypatch.setattr(runner, "RESULT", output)
    monkeypatch.setattr(runner, "verify_seal", lambda p: {})
    monkeypatch.setattr(runner, "load_inputs", lambda s: inputs)
    monkeypatch.setattr(runner, "identity_for", lambda *a: {"synthetic": True})
    monkeypatch.setattr(runner, "check_outcome", lambda o: {"diagnostic": "synthetic"})
    monkeypatch.setattr(
        runner.original.parent.parent, "learned_weights_for_repetition", lambda l, i, r: [i, r]
    )
    calls = []
    interrupted = False

    def fake(**kw):
        nonlocal interrupted
        i, r = kw["point"]["stage4a_index"], kw["repetition"]
        if not interrupted:
            interrupted = True
            raise RuntimeError("interrupted")
        assert kw["learned_weights"] == [i, r]
        calls.append((i, r))
        return {"repetition": r, "gates": dict.fromkeys(contract.GATES, True)}

    monkeypatch.setattr(runner.original, "run_repetition", fake)
    with pytest.raises(RuntimeError, match="interrupted"):
        runner.execute(output, tmp_path / "seal")
    assert yaml.safe_load(output.read_text())["points"] == inputs["source"]["points"]
    runner.execute(output, tmp_path / "seal")
    assert calls == [(i, r) for i in range(7) for r in range(2) if (i, r) != (0, 0)]
    saved = yaml.safe_load(output.read_text())
    assert saved["status"] == "completed-stage4d-continuation"
    assert saved["points"][0]["joint_classification"] == "bounded_partial_survival"
    assert saved["points"][-1]["joint_classification"] == "learning_or_first_order_failure"
    runner.execute(output, tmp_path / "seal")
    assert len(calls) == 13 and inputs["source"] == pristine
    for tamper in ("source", "counts", "diagnostics", "earlier"):
        value = deepcopy(saved)
        if tamper == "source":
            value["points"][0]["outcomes"][0]["changed"] = True
        elif tamper == "counts":
            value["joint_classification_counts"] = {}
        elif tamper == "diagnostics":
            value["numerical_diagnostics"] = []
        else:
            value["points"][0]["earlier_classifications"]["stage4b"] = "figure14_survival"
        with pytest.raises(ValueError):
            runner.validate_payload(value, {"synthetic": True}, inputs)


@pytest.mark.parametrize("which", ["source", "output"])
def test_both_result_locks_prevent_concurrent_execution(modules, monkeypatch, tmp_path, which):
    runner, _ = modules
    source, output = tmp_path / "source.yaml", tmp_path / "result.yaml"
    monkeypatch.setattr(runner, "SOURCE", source)
    monkeypatch.setattr("sys.argv", ["runner", "--output", str(output)])
    path = source if which == "source" else output
    with path.with_suffix(".yaml.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            runner.main()
    assert not output.exists() and not source.exists()


def test_missing_seal_cannot_authorize_replay(modules, tmp_path):
    runner, _ = modules
    seal = tmp_path / "seal.yaml"
    seal.write_text("status: not-sealed\n")
    with pytest.raises(ValueError, match="seal"):
        runner.verify_seal(seal)


def test_rejected_new_outcome_is_saved_and_never_rerun(modules, monkeypatch, tmp_path):
    runner, _ = modules
    inputs = synthetic_inputs(runner)
    output = tmp_path / "rejected.yaml"
    monkeypatch.setattr(runner, "RESULT", output)
    monkeypatch.setattr(runner, "verify_seal", lambda p: {})
    monkeypatch.setattr(runner, "load_inputs", lambda s: inputs)
    monkeypatch.setattr(runner, "identity_for", lambda *a: {})
    monkeypatch.setattr(
        runner.original.parent.parent, "learned_weights_for_repetition", lambda *a: {}
    )
    calls = []
    monkeypatch.setattr(
        runner.original,
        "run_repetition",
        lambda **kw: calls.append(kw) or {"repetition": 1, "rejected": True},
    )

    def check(outcome):
        if outcome.get("rejected"):
            raise ValueError("numerical rejection")
        return {}

    monkeypatch.setattr(runner, "check_outcome", check)
    for _ in range(2):
        with pytest.raises(ValueError, match="numerical rejection"):
            runner.execute(output, tmp_path / "seal")
    saved = yaml.safe_load(output.read_text())
    assert len(calls) == 1
    assert saved["points"][0]["outcomes"][0] == inputs["source"]["points"][0]["outcomes"][0]
    assert saved["points"][0]["outcomes"][1] == {"repetition": 1, "rejected": True}
