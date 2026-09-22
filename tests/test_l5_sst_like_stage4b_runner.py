from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_l5_sst_like_stage4b.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("l5_sst_stage4b_runner", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SCRIPT.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def test_protocol_is_parent_locked() -> None:
    runner = load_runner()
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/post2008-l5-sst-like-stage4-progression-registration-1002.yaml"
        ).read_text()
    )
    parent = yaml.safe_load(
        (
            ROOT / "docs/validation-results/calibrated-figure14-holdout-registration-807.yaml"
        ).read_text()
    )
    assert runner.validate_protocol(registration, parent) == {
        "duration_ms": 1000.0,
        "dt_ms": 0.01,
        "histogram_bin_ms": 1.0,
        "hamming_window_ms": 200.0,
    }
    changed = yaml.safe_load(yaml.safe_dump(parent))
    changed["figure14_protocol"]["gamma_band_hz"] = [25.0, 70.0]
    with pytest.raises(ValueError, match="analysis band"):
        runner.validate_protocol(registration, changed)
    altered = yaml.safe_load(yaml.safe_dump(registration))
    altered["stage4b_figure14"]["gates"] = runner.GATE_NAMES[:2]
    with pytest.raises(ValueError, match="gates changed"):
        runner.validate_protocol(altered, parent)


def test_every_stage4a_coordinate_is_retained_including_failure() -> None:
    runner = load_runner()
    points = runner.stage4a.registered_points()
    assert len(points) == 7
    assert points[-1]["point_id"] == "delay7p0-resource1p0"
    assert [point["stage4a_index"] for point in points] == list(range(7))


def test_learned_weights_use_same_point_and_repetition() -> None:
    runner = load_runner()
    points = runner.stage4a.registered_points()
    input_points = []
    for point_index, point in enumerate(points):
        outcomes = []
        for repetition in range(2):
            marker = point_index * 10 + repetition
            outcomes.append(
                {
                    "repetition": repetition,
                    "figure6": {
                        "bottom_up_weights": [marker + 0.1],
                        "top_down_wide_weights": [marker + 0.2],
                        "top_down_narrow_weights": [marker + 0.3],
                    },
                }
            )
        input_points.append({**point, "outcomes": outcomes})
    result = {"points": input_points}
    assert runner.learned_weights_for_repetition(result, 6, 1) == {
        "modeldb112923.projection.035": [61.1],
        "modeldb112923.projection.005": [61.2],
        "modeldb112923.projection.007": [61.3],
    }
    result["points"][6]["point_id"] = "wrong"
    with pytest.raises(ValueError, match="point identity"):
        runner.learned_weights_for_repetition(result, 6, 1)


def test_repetition_uses_fixed_conditions_and_restores_builder(monkeypatch) -> None:
    runner = load_runner()
    original = runner.classic_sector.build_first_order_connected_sector
    calls = []
    point = runner.stage4a.registered_points()[0]
    monkeypatch.setattr(runner.stage4a, "_point_builder", lambda value: "wrapped-builder")

    def fake_run(*, condition, **kwargs):
        calls.append((condition, kwargs, runner.classic_sector.build_first_order_connected_sector))
        frequency = 55.0 if condition is runner.MatchCondition.MATCH else 10.0
        gamma = 2.0 if condition is runner.MatchCondition.MATCH else 1.0
        return SimpleNamespace(
            network_result=SimpleNamespace(
                v1_cortical_spike_times_ms=(1.0, 2.0),
                relay_spike_times_ms=(1.0,),
                trn_spike_times_ms=(),
                nonspecific_spike_times_ms=(),
            ),
            spectrum=SimpleNamespace(
                dominant_frequency_hz=frequency,
                low_power=0.2,
                middle_caption_power=0.3,
                middle_methods_power=0.4,
                gamma_power=gamma,
            ),
        )

    monkeypatch.setattr(runner, "run_figure14_condition", fake_run)
    monkeypatch.setattr(
        runner,
        "assess_figure14_spectra",
        lambda match, mismatch: SimpleNamespace(
            match_gamma_dominant=True,
            mismatch_lower_frequency_dominant=True,
            mismatch_gamma_reduced=True,
        ),
    )
    weights = {"modeldb112923.projection.035": [1.0]}
    profile = {
        "figure7_protocol": {
            "top_down_current_pA": 800.0,
            "top_down_current_mode": "until_cued_cell_first_event",
            "top_down_cue_lead_ms": 0.0,
            "equilibration_ms": 0.0,
        },
        "comparator": {"target_count": 5, "source_index": 40},
    }
    result = runner.run_registered_repetition(
        point=point,
        repetition=1,
        learned_weights=weights,
        conventions="conventions",
        scales={"projection": 1.0},
        profile=profile,
        protocol={
            "duration_ms": 1000.0,
            "dt_ms": 0.01,
            "histogram_bin_ms": 1.0,
            "hamming_window_ms": 200.0,
        },
        brian="brian",
    )
    assert [condition for condition, _, _ in calls] == [
        runner.MatchCondition.MATCH,
        runner.MatchCondition.MISMATCH,
    ]
    assert all(builder == "wrapped-builder" for _, _, builder in calls)
    assert all(kwargs["learned_weights"] is weights for _, kwargs, _ in calls)
    assert all(kwargs["duration_ms"] == 1000.0 for _, kwargs, _ in calls)
    assert runner.classic_sector.build_first_order_connected_sector is original
    assert result["figure14_pass"] is True
    assert result["match"]["cortical_spike_times_ms"] == [1.0, 2.0]


def test_classification_and_checkpoint_contract() -> None:
    runner = load_runner()
    points = runner.stage4a.registered_points()
    first = {**points[0], "outcomes": [], "classification": None}
    runner.validate_checkpoint_points([first])
    with pytest.raises(ValueError, match="order or identity"):
        runner.validate_checkpoint_points([{**points[1], "outcomes": []}])
    with pytest.raises(ValueError, match="repetition order"):
        runner.validate_checkpoint_points([{**points[0], "outcomes": [{"repetition": 1}]}])
    outcome0 = {"repetition": 0, "figure14_pass": True, "marker": 1}
    outcome1 = {"repetition": 1, "figure14_pass": True, "marker": 1}
    assert runner.classify_point([outcome0, outcome1])["classification"] == "figure14_survival"
    failed = {"repetition": 1, "figure14_pass": False, "marker": 1}
    assert runner.classify_point([outcome0, failed])["classification"] == "engineering_stop"
    both_failed = [
        {"repetition": 0, "figure14_pass": False, "marker": 1},
        {"repetition": 1, "figure14_pass": False, "marker": 1},
    ]
    assert runner.classify_point(both_failed)["classification"] == "figure14_failure"
    with pytest.raises(ValueError, match="exactly two"):
        runner.classify_point([outcome0])


def test_runner_rejects_unsealed_status(tmp_path: Path) -> None:
    runner = load_runner()
    seal = tmp_path / "seal.yaml"
    seal.write_text(yaml.safe_dump({"status": "draft", "files": {}}))
    with pytest.raises(ValueError, match="execution seal"):
        runner.verify_seal(seal)
