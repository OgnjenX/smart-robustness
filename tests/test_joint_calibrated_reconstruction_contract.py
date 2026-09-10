import hashlib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[1]
RESULTS = ROOT / "docs/validation-results"


def _load(name: str) -> dict:
    return yaml.safe_load((RESULTS / name).read_text())


def _assert_local_hash_pairs(value: object) -> None:
    """Verify path/hash pairs without requiring ignored primary-source files."""

    if isinstance(value, dict):
        path = value.get("path")
        digest = value.get("sha256")
        if isinstance(path, str) and isinstance(digest, str):
            candidate = ROOT / path
            if candidate.is_file() and not path.startswith("tmp/"):
                assert hashlib.sha256(candidate.read_bytes()).hexdigest() == digest

        for key, digest in value.items():
            if not key.endswith("_sha256") or not isinstance(digest, str):
                continue
            path = value.get(key.removesuffix("_sha256"))
            if not isinstance(path, str):
                continue
            candidate = ROOT / path
            if candidate.is_file() and not path.startswith("tmp/"):
                assert hashlib.sha256(candidate.read_bytes()).hexdigest() == digest

        for child in value.values():
            _assert_local_hash_pairs(child)
    elif isinstance(value, list):
        for child in value:
            _assert_local_hash_pairs(child)


@pytest.mark.parametrize(
    "name",
    [
        "joint-calibration-t-match-screen-registration-770.yaml",
        "joint-calibration-t-mismatch-screen-registration-773.yaml",
        "joint-calibration-projection025-figure10-recovery-registration-779.yaml",
        "figure10-target-resolved-scale8-registration-783.yaml",
        "joint-calibration-projection038-figure10-registration-786.yaml",
        "joint-calibration-projection026-figure10-registration-789.yaml",
        "joint-calibration-first-order-confirmation-recovery-registration-795.yaml",
        "joint-calibration-learned-state-audit-recovery-registration-801.yaml",
        "joint-calibration-global-projection026-registration-804.yaml",
        "calibrated-figure14-holdout-registration-807.yaml",
        "calibrated-figure15-holdout-registration-810.yaml",
        "calibrated-figure16-holdout-registration-813.yaml",
        "figure15-analysis-sensitivity-recovery-registration-820.yaml",
    ],
)
def test_joint_calibration_registrations_retain_local_hash_identity(name: str) -> None:
    registration = _load(name)
    assert registration["status"].startswith("registered-before-execution")
    _assert_local_hash_pairs(registration)


@pytest.mark.parametrize(
    ("registration_name", "failure_name", "recovery_name"),
    [
        (
            "joint-calibration-projection025-figure10-registration-776.yaml",
            "joint-calibration-projection025-figure10-assessment-778.yaml",
            "joint-calibration-projection025-figure10-recovery-registration-779.yaml",
        ),
        (
            "joint-calibration-first-order-confirmation-registration-792.yaml",
            "joint-calibration-first-order-confirmation-assessment-794.yaml",
            "joint-calibration-first-order-confirmation-recovery-registration-795.yaml",
        ),
        (
            "joint-calibration-learned-state-audit-registration-798.yaml",
            "joint-calibration-learned-state-audit-assessment-800.yaml",
            "joint-calibration-learned-state-audit-recovery-registration-801.yaml",
        ),
    ],
)
def test_failed_first_attempts_are_preserved_and_recoveries_pin_current_runner(
    registration_name: str,
    failure_name: str,
    recovery_name: str,
) -> None:
    registration = _load(registration_name)
    failure = _load(failure_name)
    recovery = _load(recovery_name)

    script = ROOT / registration["script"]
    current_digest = hashlib.sha256(script.read_bytes()).hexdigest()
    assert current_digest != registration["script_sha256"]
    assert failure["status"].startswith(("implementation-failure", "setup-failure"))
    assert recovery["script"] == registration["script"]
    assert recovery["script_sha256"] == current_digest
    _assert_local_hash_pairs(recovery)


@pytest.mark.parametrize(
    "name",
    [
        "joint-calibration-t-match-screen-assessment-772.yaml",
        "joint-calibration-t-mismatch-screen-assessment-775.yaml",
        "joint-calibration-projection025-figure10-assessment-778.yaml",
        "joint-calibration-projection025-figure10-recovery-assessment-781.yaml",
        "figure10-target-resolved-scale8-assessment-785.yaml",
        "joint-calibration-projection038-figure10-assessment-788.yaml",
        "joint-calibration-projection026-figure10-assessment-791.yaml",
        "joint-calibration-first-order-confirmation-assessment-794.yaml",
        "joint-calibration-first-order-confirmation-recovery-assessment-797.yaml",
        "joint-calibration-learned-state-audit-assessment-800.yaml",
        "joint-calibration-learned-state-audit-recovery-assessment-803.yaml",
        "joint-calibration-global-projection026-assessment-806.yaml",
        "calibrated-figure14-holdout-assessment-809.yaml",
        "calibrated-figure15-holdout-assessment-812.yaml",
        "calibrated-figure16-holdout-assessment-815.yaml",
        "figure15-analysis-sensitivity-assessment-819.yaml",
        "figure15-analysis-sensitivity-recovery-assessment-822.yaml",
    ],
)
def test_joint_calibration_assessments_retain_result_hash_identity(name: str) -> None:
    _assert_local_hash_pairs(_load(name))


def test_joint_calibration_training_split_and_first_order_selection_are_fixed() -> None:
    methodology = _load("joint-calibrated-reconstruction-methodology-769.yaml")
    assert methodology["epistemic_boundary"]["label_if_successful"] == (
        "calibrated classic-SMART reconstruction"
    )
    assert "robustness under alternative neuron models" in methodology[
        "holdouts_not_open_during_calibration"
    ]
    assert "modern-anatomy variants" in methodology[
        "holdouts_not_open_during_calibration"
    ]

    match = _load("joint-calibration-t-match-screen-assessment-772.yaml")
    mismatch = _load("joint-calibration-t-mismatch-screen-assessment-775.yaml")
    assert match["survivors"] == [0.203125, 0.21875]
    assert [row["nonspecific_t_scale"] for row in mismatch["survivors"]] == [
        0.203125
    ]

    global_screen = _load("joint-calibration-global-projection026-assessment-806.yaml")
    assert global_screen["passing_scales"] == [0.5]
    assert global_screen["selected_projection026_scale"] == 0.5
    assert global_screen["first_order_endpoint_promoted"]
    endpoint = global_screen["promoted_endpoint"]
    assert endpoint == {
        "nonspecific_t_scale": 0.203125,
        "nonspecific_t_effective_density_mS_cm2": 50.78125,
        "projection025_scale": 8.0,
        "projection026_scale": 0.5,
        "projection036_spread_convention": "variance",
        "projection038_scale": 1.0,
        "projection008_scale": 0.75,
        "projection011_scale": 0.9375,
        "integration_method": "rk4",
    }
    assert not global_screen["original_smart_reproduced"]
    assert not global_screen["baseline_frozen"]


def test_downstream_only_pass_is_not_confused_with_global_retraining() -> None:
    downstream = _load("joint-calibration-projection026-figure10-assessment-791.yaml")
    confirmation = _load(
        "joint-calibration-first-order-confirmation-recovery-assessment-797.yaml"
    )
    learned_state = _load(
        "joint-calibration-learned-state-audit-recovery-assessment-803.yaml"
    )

    assert downstream["passing_scales"] == [0.75]
    assert downstream["parameter_selected"]
    assert not downstream["baseline_promoted"]
    assert confirmation["figure7"]["pass"]
    assert not confirmation["figure10"]["pass"]
    assert confirmation["comparison_with_screen_790"][
        "screen_training_scales_excluded_projection025_and_026"
    ]
    assert [row["exact_changed_values"] for row in learned_state["learned_weight_differences"]] == [
        25,
        13,
        65,
    ]
    assert not learned_state["parameter_selected"]


def test_calibrated_holdout_matrix_records_single_unresolved_figure15_gate() -> None:
    figure14 = _load("calibrated-figure14-holdout-assessment-809.yaml")
    figure15 = _load("calibrated-figure15-holdout-assessment-812.yaml")
    figure16 = _load("calibrated-figure16-holdout-assessment-815.yaml")

    assert figure14["figure14_holdout_pass"]
    assert figure14["match"]["dominant_frequency_hz"] == 55.0
    assert figure14["mismatch"]["dominant_frequency_hz"] == 10.0

    assert not figure15["figure15_holdout_pass"]
    assert figure15["result"]["pair"] == [39, 40]
    assert figure15["result"]["gamma_peak_hz"] == pytest.approx(
        53.026513256628306
    )
    assert not figure15["gates"]["peak_within_39_49_hz"]

    assert figure16["figure16_holdout_pass"]
    assert figure16["result"]["strongest_band_hz"] == [2.0, 4.0]
    assert figure16["result"]["lower_to_gamma_peak_ratio"] == pytest.approx(
        4.824415614511934
    )
    assert figure16["overall_validation_matrix"] == {
        "figure6_learning": "pass",
        "figure7_match_mismatch": "pass",
        "figure10_causal_reset": "pass",
        "figure14_match_gamma_mismatch_slow": "pass",
        "figure15_local_44hz_synchrony": "fail_53.0265hz_outside_39_49hz",
        "figure16_higher_order_lower_frequency_dominance": "pass",
    }
    assert not figure16["original_smart_reproduced"]
    assert not figure16["baseline_promoted"]
    assert not figure16["baseline_frozen"]


def test_figure15_source_audit_preserves_failure_and_identifiability_boundary() -> None:
    audit = _load("figure15-primary-source-identifiability-audit-816.yaml")
    _assert_local_hash_pairs(audit)

    assert audit["article_findings"]["figure_label"]["simulation_panel_label"] == (
        "F = 44 Hz"
    )
    assert not audit["article_findings"]["caption"][
        "identifies_cell_indices_or_coordinates"
    ]
    assert not audit["article_findings"]["caption"]["defines_frequency_estimator"]
    assert audit["archive_findings"]["figure15_pair_designation"] == "absent"
    assert audit["archive_findings"]["recorded_figure15_spike_arrays"] == "absent"
    assert not audit["identifiability_gates"]["exact_numeric_reproduction_test_identifiable"]
    assert audit["holdout_interpretation"]["registered_numeric_gate_result"] == "fail"
    assert audit["holdout_interpretation"][
        "source_identifiable_local_gamma_phenotype"
    ] == "reproduced"
    assert not audit["analysis_selected"]
    assert not audit["registered_holdout_reclassified_as_pass"]
    assert not audit["baseline_frozen"]


def test_figure15_sensitivity_failure_and_recovery_are_auditable() -> None:
    registration = _load("figure15-analysis-sensitivity-registration-817.yaml")
    failure = _load("figure15-analysis-sensitivity-818.yaml")
    assessment = _load("figure15-analysis-sensitivity-assessment-819.yaml")
    recovery = _load("figure15-analysis-sensitivity-recovery-registration-820.yaml")

    assert failure["status"] == "failed-after-simulation-before-yaml-emission"
    assert failure["temporary_output_bytes"] == 0
    assert not failure["scientific_values_emitted"]
    assert not assessment["scientific_result_available"]
    assert registration["script"] == recovery["script"]
    assert registration["script_sha256"] != recovery["script_sha256"]
    assert hashlib.sha256((ROOT / recovery["script"]).read_bytes()).hexdigest() == (
        recovery["script_sha256"]
    )
    assert recovery["recovery_scope"]["sole_change"].startswith(
        "cast numpy.isclose"
    )
    assert not recovery["recovery_scope"]["scientific_value_observed_before_recovery"]


def test_figure15_analysis_family_confirms_robust_high_peak_without_selection() -> None:
    result = _load("figure15-analysis-sensitivity-recovery-821.yaml")
    assessment = _load("figure15-analysis-sensitivity-recovery-assessment-822.yaml")

    assert all(result["deterministic_reproduction_gates"].values())
    assert result["deterministic_reproduction_pass"]
    assert result["fixed_pair"] == [39, 40]
    assert [len(result["raw_pair_spike_times_ms"][key]) for key in ("39", "40")] == [
        114,
        148,
    ]
    peaks = [row["gamma_peak_hz"] for row in result["analysis_family"]]
    assert peaks == pytest.approx(
        [
            53.026513256628306,
            53.026513256628306,
            52.631578947368425,
            52.631578947368425,
            53.0,
            55.0,
        ]
    )
    assert not result["selection_performed"]
    assert assessment["gates"]["every_method_in_published_20_70hz_gamma_band"]
    assert not assessment["gates"]["any_method_within_registered_39_49hz_gate"]
    assert assessment["gates"]["registered_result_is_analysis-family-robust"]
    assert not assessment["analysis_selected"]
    assert not assessment["registered_holdout_reclassified_as_pass"]
    assert not assessment["baseline_frozen"]
