from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from smart_robustness.baseline import load_frozen_classic_baseline

ROOT = Path(__file__).parents[1]
MANIFEST = ROOT / "configs/baselines/classic_smart_calibrated_v1.yaml"


def test_frozen_baseline_reconstructs_the_promoted_runtime_and_scales() -> None:
    baseline = load_frozen_classic_baseline(MANIFEST)

    assert baseline.name == "classic-smart-calibrated-behavioral-v1"
    assert baseline.implementation_commit == (
        "f6dc9de948586597d13d544a50e2b1bf48ca5514"
    )
    assert baseline.runtime_conventions().fingerprint == baseline.runtime_fingerprint
    assert baseline.projection036_variance_topology
    assert dict(baseline.projection_weight_scales) == {
        "modeldb112923.projection.000": 0.01,
        "modeldb112923.projection.001": 0.01,
        "modeldb112923.projection.004": 0.03,
        "modeldb112923.projection.008": 0.75,
        "modeldb112923.projection.011": 0.9375,
        "modeldb112923.projection.025": 8.0,
        "modeldb112923.projection.026": 0.5,
    }
    assert not baseline.exact_numerical_reproduction_claimed


def test_frozen_baseline_preserves_passes_and_failed_numeric_holdout() -> None:
    manifest = yaml.safe_load(MANIFEST.read_text())
    expected = manifest["expected_behavior"]

    assert expected["figure6_learning"] == "pass"
    assert expected["figure7_match_mismatch"]["nonspecific_events_match_mismatch"] == [
        4,
        7,
    ]
    assert expected["figure10_reset"]["intact_control_old_winner_events"] == [53, 63]
    assert expected["figure14_spectrum"]["match_peak_hz"] == 55.0
    assert expected["figure14_spectrum"]["mismatch_peak_hz"] == 10.0
    assert expected["figure15_local_synchrony"]["source_identifiable_gamma_phenotype"] == (
        "pass"
    )
    assert expected["figure15_local_synchrony"]["preregistered_numeric_gate"] == "fail"
    assert expected["figure15_local_synchrony"]["reconstructed_peak_hz"] == pytest.approx(
        53.026513256628306
    )
    assert expected["figure16_higher_order"]["lower_to_gamma_peak_ratio"] == pytest.approx(
        4.824415614511934
    )
    assert manifest["baseline_frozen"]
    assert not manifest["exact_numerical_reproduction_claimed"]
    assert manifest["robustness_experiment_contract"][
        "no_further_classic_parameter_calibration"
    ]


def test_frozen_baseline_rejects_changed_evidence(tmp_path: Path) -> None:
    manifest = yaml.safe_load(MANIFEST.read_text())
    manifest["implementation"]["profile"]["sha256"] = "0" * 64
    changed = tmp_path / "configs/baselines/changed.yaml"
    changed.parent.mkdir(parents=True)
    changed.write_text(yaml.safe_dump(manifest, sort_keys=False))

    with pytest.raises(ValueError, match="frozen evidence differs"):
        load_frozen_classic_baseline(changed, repository_root=ROOT)
