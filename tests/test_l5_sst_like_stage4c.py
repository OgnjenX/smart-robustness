"""Zero-network-outcome contract tests for the sealed Stage-4C implementation."""

from __future__ import annotations

import importlib.util
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from smart_robustness.analysis.figure15 import figure15_layer4_synchrony

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def script_import_path(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))


def load(name):
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def synthetic_outcome(repetition=0, period=20.0):
    times = np.arange(10, 1000, period)
    indices = [39] * len(times) + [40] * len(times)
    spikes = list(times) + list(times + 1)
    synchrony = figure15_layer4_synchrony(indices, spikes)
    peak = float(synchrony.gamma_peak_hz)
    return {
        "repetition": repetition,
        "layer4_spike_indices": indices,
        "layer4_spike_times_ms": [float(t) for t in spikes],
        "first_cell_spikes": len(times),
        "second_cell_spikes": len(times),
        "gamma_peak_hz": peak,
        "numeric_44hz_diagnostic_pass": abs(peak - 44) <= 5,
        "gates": {
            "both_cells_have_at_least_two_spikes": True,
            "peak_in_published_gamma_band": True,
        },
        "figure15_source_identifiable_pass": True,
    }


def test_protocol_is_locked_and_numeric_diagnostic_not_survival_gate():
    runner = load("run_l5_sst_like_stage4c")
    reg = yaml.safe_load((ROOT / runner.REGISTRATION).read_text())
    parent = yaml.safe_load((ROOT / runner.PARENT_PROTOCOL).read_text())
    protocol = runner.validate_protocol(reg, parent)
    assert protocol["first_cell_index"] == 39 and protocol["second_cell_index"] == 40
    assert protocol["target_hz"] == 44
    for key, value in [
        ("pair", [38, 40]),
        ("dt_ms", 0.02),
        ("histogram_bin_ms", 2),
        ("input_learning_state", "route-free-weights"),
    ]:
        changed = deepcopy(reg)
        changed["stage4c_figure15"][key] = value
        with pytest.raises(ValueError):
            runner.validate_protocol(changed, parent)
    changed = deepcopy(parent)
    changed["figure15_protocol"]["spectrum_input"] = "displayed-lags-only"
    with pytest.raises(ValueError):
        runner.validate_protocol(reg, changed)
    o = synthetic_outcome()
    assert o["numeric_44hz_diagnostic_pass"] is False
    assert (
        runner.classify_point([o, {**o, "repetition": 1}])["classification"] == "figure15_survival"
    )


@pytest.mark.parametrize("period", [20.0, 22.0, 25.0, 35.0])
def test_independent_direct_fft_agrees_for_synthetic_trains(period):
    verifier = load("verify_l5_sst_like_stage4c")
    verifier.check_outcome(synthetic_outcome(period=period))


def test_verifier_rejects_tampering_and_handles_silent_pair():
    verifier = load("verify_l5_sst_like_stage4c")
    for key, value in [
        ("gamma_peak_hz", 44.0),
        ("first_cell_spikes", 0),
        ("numeric_44hz_diagnostic_pass", True),
        ("figure15_source_identifiable_pass", False),
    ]:
        changed = synthetic_outcome()
        changed[key] = value
        with pytest.raises(ValueError):
            verifier.check_outcome(changed)
    empty = {
        "repetition": 0,
        "layer4_spike_indices": [],
        "layer4_spike_times_ms": [],
        "first_cell_spikes": 0,
        "second_cell_spikes": 0,
        "gamma_peak_hz": None,
        "numeric_44hz_diagnostic_pass": False,
        "gates": {
            "both_cells_have_at_least_two_spikes": False,
            "peak_in_published_gamma_band": False,
        },
        "figure15_source_identifiable_pass": False,
    }
    assert len(verifier.check_outcome(empty)["failed_gates"]) == 2


def test_repetition_uses_corresponding_weights_and_restores_builder(monkeypatch):
    runner = load("run_l5_sst_like_stage4c")
    original = runner.classic_sector.build_first_order_connected_sector
    monkeypatch.setattr(runner.parent.stage4a, "_point_builder", lambda p: "route-builder")
    calls = []
    outcome = synthetic_outcome()

    def fake(**kwargs):
        calls.append(kwargs)
        assert runner.classic_sector.build_first_order_connected_sector == "route-builder"
        return SimpleNamespace(
            network_result=SimpleNamespace(
                layer4_spike_indices=outcome["layer4_spike_indices"],
                layer4_spike_times_ms=outcome["layer4_spike_times_ms"],
            ),
            synchrony=SimpleNamespace(
                first_spike_count=50, second_spike_count=50, gamma_peak_hz=outcome["gamma_peak_hz"]
            ),
        )

    monkeypatch.setattr(runner, "run_figure15_condition", fake)
    profile = {
        "figure7_protocol": {
            "top_down_current_pA": 800,
            "top_down_current_mode": "until_cued_cell_first_event",
            "top_down_cue_lead_ms": 0,
            "equilibration_ms": 0,
        }
    }
    weights = {"projection": [123]}
    result = runner.run_repetition(
        point={},
        repetition=1,
        learned_weights=weights,
        conventions="runtime",
        scales={},
        profile=profile,
        protocol={},
    )
    assert calls[0]["learned_weights"] is weights
    assert result["repetition"] == 1
    assert runner.classic_sector.build_first_order_connected_sector is original

    def fail(**kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(runner, "run_figure15_condition", fail)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        runner.run_repetition(
            point={},
            repetition=0,
            learned_weights=weights,
            conventions=None,
            scales={},
            profile=profile,
            protocol={},
        )
    assert runner.classic_sector.build_first_order_connected_sector is original


def fixtures(runner):
    learning = {"points": []}
    spectra = {"points": []}
    points = []
    for index, point in enumerate(runner.parent.stage4a.registered_points()):
        a = "failure" if index == 6 else "learning_first_order_survival"
        b = "figure14_failure" if index < 4 else "figure14_survival"
        learning["points"].append({"classification": a})
        spectra["points"].append({"classification": b})
        outcomes = [synthetic_outcome(r) for r in range(2)]
        points.append(
            {
                **point,
                "stage4a_classification": a,
                "stage4b_classification": b,
                "outcomes": outcomes,
                **runner.classify_point(outcomes),
            }
        )
    return learning, spectra, points


def test_checkpoint_retains_all_failures_and_rejects_false_identity_or_counts():
    runner = load("run_l5_sst_like_stage4c")
    learning, spectra, points = fixtures(runner)
    payload = {
        "schema_version": 1,
        "status": "completed-stage4c",
        "identity": {},
        "points": points,
        "network_execution": True,
        "frozen_baseline_modified": False,
        "all_points_reported": True,
        "classification_counts": {"figure15_survival": 7},
    }
    runner.validate_payload(payload, {}, learning, spectra)
    for changed in [
        dict(payload, points=points[:6]),
        dict(payload, identity={"wrong": True}),
        dict(payload, classification_counts={"figure15_survival": 6}),
    ]:
        with pytest.raises(ValueError):
            runner.validate_payload(changed, {}, learning, spectra)
    changed = deepcopy(payload)
    changed["points"][-1]["stage4a_classification"] = "rescued"
    with pytest.raises(ValueError, match="earlier classification"):
        runner.validate_payload(changed, {}, learning, spectra)
    changed = deepcopy(payload)
    changed["points"][0]["outcomes"][1]["layer4_spike_times_ms"][0] += 0.1
    with pytest.raises(ValueError, match="classification"):
        runner.validate_payload(changed, {}, learning, spectra)


def test_missing_or_changed_seal_prevents_execution(tmp_path):
    runner = load("run_l5_sst_like_stage4c")
    seal = tmp_path / "seal.yaml"
    seal.write_text("status: not-sealed\n")
    with pytest.raises(ValueError, match="seal"):
        runner.verify_seal(seal)


def test_execute_resumes_once_and_terminal_repeat_is_noop(tmp_path, monkeypatch):
    runner = load("run_l5_sst_like_stage4c")
    learning, spectra, _points = fixtures(runner)
    output = tmp_path / "result.yaml"
    monkeypatch.setattr(runner, "RESULT", output)
    monkeypatch.setattr(runner, "verify_seal", lambda path: {})
    monkeypatch.setattr(runner, "identity_for", lambda *args: {"synthetic": True})
    baseline = SimpleNamespace(runtime_conventions=lambda: "runtime", projection_weight_scales={})
    monkeypatch.setattr(runner, "load_inputs", lambda seal: (learning, spectra, baseline, {}, {}))
    monkeypatch.setattr(
        runner.parent, "learned_weights_for_repetition", lambda l, i, r: {"marker": [i, r]}
    )
    calls = []
    interrupted = False

    def fake(**kwargs):
        nonlocal interrupted
        index = kwargs["point"]["stage4a_index"]
        repetition = kwargs["repetition"]
        if index == 0 and repetition == 1 and not interrupted:
            interrupted = True
            raise RuntimeError("interrupted")
        assert kwargs["learned_weights"] == {"marker": [index, repetition]}
        calls.append((index, repetition))
        return synthetic_outcome(repetition)

    monkeypatch.setattr(runner, "run_repetition", fake)
    with pytest.raises(RuntimeError, match="interrupted"):
        runner.execute(output, tmp_path / "seal")
    assert len(yaml.safe_load(output.read_text())["points"][0]["outcomes"]) == 1
    runner.execute(output, tmp_path / "seal")
    assert calls == [(i, r) for i in range(7) for r in range(2)]
    payload = yaml.safe_load(output.read_text())
    assert payload["status"] == "completed-stage4c"
    assert payload["points"][-1]["stage4a_classification"] == "failure"
    runner.execute(output, tmp_path / "seal")
    assert len(calls) == 14


def test_exact_repeat_failure_is_engineering_stop_not_scientific_failure():
    runner = load("run_l5_sst_like_stage4c")
    first, second = synthetic_outcome(0), synthetic_outcome(1)
    second["layer4_spike_times_ms"][0] += 0.1
    assert runner.classify_point([first, second])["classification"] == "engineering_stop"
    first["figure15_source_identifiable_pass"] = False
    second = {**deepcopy(first), "repetition": 1}
    assert runner.classify_point([first, second])["classification"] == "figure15_failure"
