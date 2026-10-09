"""Zero-network tests for checkpoint orchestration and independent reconstruction."""

from __future__ import annotations

import fcntl
import importlib
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from smart_robustness.analysis.cross_correlation import figure16_cross_correlations
from smart_robustness.validation import l5_sst_like_stage4d as contract
from smart_robustness.validation.higher_order import Figure16Protocol

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    return (
        importlib.import_module("run_l5_sst_like_stage4d"),
        importlib.import_module("verify_l5_sst_like_stage4d"),
    )


def synthetic_outcome(*, silent=False):
    rng = np.random.default_rng(19)
    first = rng.normal(size=1000) if not silent else np.zeros(1000)
    second = np.roll(first, 7) + rng.normal(size=1000) * 0.2 if not silent else first
    fields = {}
    for area, seed, signal in (("v1", 16, second), ("v2", 17, first)):
        matrix = np.array([signal + 5.0, signal - 5.0]).tolist()
        fields[area] = {
            "seed": seed,
            "fingerprint": "synthetic",
            "potential_uV": matrix,
            "current_source_density_uV_per_um": matrix,
            "inferior_300um_tip_depth_um": [0, 25],
            "superior_300um_tip_depth_um": [1175, 1200],
            "inferior_300um_potential_uV": matrix,
            "superior_300um_potential_uV": matrix,
        }
    # Compute the same regional means as the assay, not the original signals:
    # adding/subtracting DC above introduces a small floating-point difference.
    x = np.mean(fields["v2"]["inferior_300um_potential_uV"], axis=0)
    y = np.mean(fields["v1"]["superior_300um_potential_uV"], axis=0)
    correlations = figure16_cross_correlations(
        x, y, 1000, bands_hz=tuple(map(tuple, contract.BANDS))
    )
    peaks = [c.peak_absolute_normalized for c in correlations]
    return {
        "repetition": 0,
        "learned_state_provenance": "simulated-learned-weight-snapshot",
        "sample_times_ms": list(range(1000, 2000)),
        "fields": fields,
        "cross_correlations": [
            {
                "band_hz": list(c.band_hz),
                "lag_ms": c.lag_ms.tolist(),
                "raw": c.raw.tolist(),
                "normalized": c.normalized.tolist(),
                "peak_absolute_normalized": c.peak_absolute_normalized,
            }
            for c in correlations
        ],
        "gates": dict(zip(contract.GATES, [True, True, max(peaks[:4]) > peaks[4]], strict=True)),
    }


@pytest.mark.parametrize("silent", [True, False])
def test_direct_circular_reconstruction_agrees_with_assay(modules, silent):
    _, verifier = modules
    verifier.check_outcome(synthetic_outcome(silent=silent))


@pytest.mark.parametrize("tamper", ["time", "field", "peak", "gate", "geometry", "provenance"])
def test_independent_verifier_rejects_tampering(modules, tamper):
    _, verifier = modules
    data = synthetic_outcome()
    if tamper == "time":
        data["sample_times_ms"][0] = 0
    elif tamper == "field":
        data["fields"]["v2"]["inferior_300um_potential_uV"][0][0] += 1
    elif tamper == "peak":
        data["cross_correlations"][0]["peak_absolute_normalized"] += 0.1
    elif tamper == "gate":
        data["gates"][contract.GATES[2]] = not data["gates"][contract.GATES[2]]
    elif tamper == "geometry":
        data["fields"]["v2"]["seed"] = 18
    else:
        data["learned_state_provenance"] = "paper-constrained-figure6c-reference"
    with pytest.raises(ValueError):
        verifier.check_outcome(data)


def fixtures(runner):
    registered = runner.parent.parent.stage4a.registered_points()
    learning = {"points": [{"classification": "learning_first_order_survival"} for _ in registered]}
    learning["points"][-1]["classification"] = "failure"
    spectra = {"points": [{"classification": "figure14_survival"} for _ in registered]}
    spectra["points"][0]["classification"] = "figure14_failure"
    synchrony = {"points": [{"classification": "figure15_survival"} for _ in registered]}
    return learning, spectra, synchrony


def test_resume_retains_all_points_weights_and_terminal_noop(modules, monkeypatch, tmp_path):
    runner, verifier = modules
    learning, spectra, synchrony = fixtures(runner)
    output = tmp_path / "synthetic.yaml"
    monkeypatch.setattr(runner, "RESULT", output)
    monkeypatch.setattr(runner, "verify_seal", lambda path: {})
    monkeypatch.setattr(runner, "identity_for", lambda *args: {"synthetic": True})
    baseline = SimpleNamespace(runtime_conventions=lambda: "runtime", projection_weight_scales={})
    monkeypatch.setattr(
        runner,
        "load_inputs",
        lambda seal: (learning, spectra, synchrony, baseline, {}, Figure16Protocol()),
    )
    monkeypatch.setattr(
        runner.parent.parent, "learned_weights_for_repetition", lambda l, i, r: [i, r]
    )
    # Checkpoint orchestration is tested independently of full-field reconstruction.
    monkeypatch.setattr(verifier, "check_outcome", lambda o: {})
    calls = []
    interrupted = False

    def fake(**kwargs):
        nonlocal interrupted
        index, repetition = kwargs["point"]["stage4a_index"], kwargs["repetition"]
        if index == 0 and repetition == 1 and not interrupted:
            interrupted = True
            raise RuntimeError("interrupted")
        assert kwargs["learned_weights"] == [index, repetition]
        calls.append((index, repetition))
        return {"repetition": repetition, "gates": dict.fromkeys(contract.GATES, True)}

    monkeypatch.setattr(runner, "run_repetition", fake)
    with pytest.raises(RuntimeError, match="interrupted"):
        runner.execute(output, tmp_path / "seal")
    assert len(yaml.safe_load(output.read_text())["points"][0]["outcomes"]) == 1
    runner.execute(output, tmp_path / "seal")
    assert calls == [(i, r) for i in range(7) for r in range(2)]
    saved = yaml.safe_load(output.read_text())
    assert saved["status"] == "completed-stage4d"
    assert saved["points"][0]["joint_classification"] == "bounded_partial_survival"
    assert saved["points"][-1]["joint_classification"] == "learning_or_first_order_failure"
    runner.execute(output, tmp_path / "seal")
    assert len(calls) == 14
    tampered = deepcopy(saved)
    tampered["points"][0]["earlier_classifications"]["stage4b"] = "figure14_survival"
    with pytest.raises(ValueError, match="earlier"):
        runner.validate_payload(tampered, {"synthetic": True}, learning, spectra, synchrony)
    tampered = deepcopy(saved)
    tampered["points"] = tampered["points"][:6]
    with pytest.raises(ValueError, match="terminal"):
        runner.validate_payload(tampered, {"synthetic": True}, learning, spectra, synchrony)


def test_unsealed_runner_cannot_execute(modules, tmp_path):
    runner, _ = modules
    seal = tmp_path / "unsealed.yaml"
    seal.write_text("status: not-sealed\n")
    with pytest.raises(ValueError, match="seal"):
        runner.verify_seal(seal)


def test_single_process_lock_rejects_concurrent_execution(modules, monkeypatch, tmp_path):
    runner, _ = modules
    output = tmp_path / "locked.yaml"
    monkeypatch.setattr("sys.argv", ["runner", "--output", str(output)])
    with output.with_suffix(".yaml.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            runner.main()
    assert not output.exists()


def test_rejected_outcome_is_preserved_and_not_rerun(modules, monkeypatch, tmp_path):
    runner, verifier = modules
    learning, spectra, synchrony = fixtures(runner)
    output = tmp_path / "stopped.yaml"
    monkeypatch.setattr(runner, "RESULT", output)
    monkeypatch.setattr(runner, "verify_seal", lambda path: {})
    monkeypatch.setattr(runner, "identity_for", lambda *args: {})
    baseline = SimpleNamespace(runtime_conventions=lambda: "runtime", projection_weight_scales={})
    monkeypatch.setattr(
        runner,
        "load_inputs",
        lambda seal: (learning, spectra, synchrony, baseline, {}, Figure16Protocol()),
    )
    monkeypatch.setattr(runner.parent.parent, "learned_weights_for_repetition", lambda *a: {})
    calls = []
    monkeypatch.setattr(
        runner,
        "run_repetition",
        lambda **kw: calls.append(kw) or {"repetition": 0, "invalid_raw_evidence": True},
    )

    def reject(outcome):
        raise ValueError("invalid fields")

    monkeypatch.setattr(verifier, "check_outcome", reject)
    for _ in range(2):
        with pytest.raises(ValueError, match="invalid fields"):
            runner.execute(output, tmp_path / "seal")
    assert len(calls) == 1
    saved = yaml.safe_load(output.read_text())
    assert saved["points"][0]["outcomes"] == [{"repetition": 0, "invalid_raw_evidence": True}]


def test_run_repetition_wires_full_network_and_retains_raw_fields(modules, monkeypatch):
    runner, verifier = modules
    data = synthetic_outcome()
    fields = {k: SimpleNamespace(**v) for k, v in data["fields"].items()}
    candidate = SimpleNamespace(
        protocol=Figure16Protocol(),
        sample_times_ms=data["sample_times_ms"],
        learned_state_provenance=data["learned_state_provenance"],
        v1_field=fields["v1"],
        v2_field=fields["v2"],
    )
    calls = []
    monkeypatch.setattr(
        runner.tempfile, "mkdtemp", lambda **kw: "/private/tmp/synthetic-not-created"
    )
    monkeypatch.setattr(
        runner, "run_figure16_candidate", lambda **kw: calls.append(kw) or candidate
    )
    monkeypatch.setattr(contract, "full_network_builder", lambda p: "full-catalog-builder")
    kwargs = {
        "point": {},
        "repetition": 0,
        "learned_weights": {"weights": [1]},
        "conventions": "runtime",
        "scales": {},
        "protocol": Figure16Protocol(),
    }
    outcome = runner.run_repetition(**kwargs)
    verifier.check_outcome(outcome)
    assert calls[0]["geometry_seed"] == 16
    assert calls[0]["projection036_variance_topology"] is True
    assert calls[0]["network_builder"] == "full-catalog-builder"
    assert calls[0]["learned_weights"] == kwargs["learned_weights"]
    candidate.v1_field.potential_uV[0][0] = float("nan")
    stopped = runner.run_repetition(**kwargs)
    assert stopped["gates"][contract.GATES[1]] is False
    assert stopped["cross_correlations"] == []
    with pytest.raises(ValueError, match="finiteness"):
        verifier.check_outcome(stopped)
