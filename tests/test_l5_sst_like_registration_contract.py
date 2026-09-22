from __future__ import annotations

from pathlib import Path

import yaml


def _yaml(name: str) -> dict:
    return yaml.safe_load(Path(name).read_text())


def test_source_supplement_preserves_translation_limits() -> None:
    source = _yaml(
        "docs/validation-results/post2008-l5-sst-like-source-supplement-984.yaml"
    )
    assert source["network_execution_authorized"] is False
    assert source["translation_decision"]["selected_layer"] == "layer5"
    assert source["translation_decision"]["first_representation"] == (
        "collapsed-disynaptic-functional-route"
    )
    assert "explicit SST intrinsic dynamics" in source["translation_decision"]["not_claimed"]
    assert source["decision"]["vip_interaction_authorized"] is False


def test_sst_grid_is_complete_prospective_and_non_fitted() -> None:
    registration = _yaml(
        "docs/validation-results/"
        "post2008-l5-sst-like-functional-grid-registration-985.yaml"
    )
    region = registration["independent_parameter_region"]
    assert registration["network_execution_authorized"] is False
    assert region["resource_fractions"] == [0.125, 0.25, 0.5, 1.0]
    assert region["collapsed_delays_ms"] == [1.0, 3.0, 7.0]
    assert region["nonzero_grid_points"] == (
        len(region["resource_fractions"]) * len(region["collapsed_delays_ms"])
    )
    anchor = region["resource_anchor"]["value_nS"]
    assert region["realized_total_conductance_nS"] == [
        anchor * fraction for fraction in region["resource_fractions"]
    ]
    null = registration["staged_execution"]["stage1_exact_null"]
    assert null["resource_fraction"] == 0.0
    assert null["canonical_network_delay_ms"] == 3.0
    assert null["structurally_inert_delay_labels_ms"] == [1.0, 3.0, 7.0]
    assert null["network_repetitions"] == 2
    assert registration["staged_execution"]["stage3_behavioral_grid"]["report_all_points"] is True
    assert "fitting resource or delay to any SMART network outcome" in registration["forbidden"]


def test_nonzero_network_execution_requires_null_and_isolated_stages() -> None:
    registration = _yaml(
        "docs/validation-results/"
        "post2008-l5-sst-like-functional-grid-registration-985.yaml"
    )
    stages = registration["staged_execution"]
    assert stages["stage1_exact_null"]["promotion"].startswith("all zero-resource")
    assert stages["stage2_isolated_recruitment"]["promotion"].startswith("all engineering")
    assert stages["stage3_behavioral_grid"]["figure10_only_after_same-run-figure7-pass"] is True


def test_isolated_recruitment_design_fixes_all_pre_outcome_choices() -> None:
    registration = _yaml(
        "docs/validation-results/"
        "post2008-l5-sst-like-isolated-recruitment-registration-989.yaml"
    )
    assert registration["isolated_execution_authorized"] is False
    assert registration["nonzero_network_execution_authorized"] is False
    protocol = registration["fixed_protocol"]
    assert protocol["emission_times_ms"] == [50.0, 70.0, 90.0, 110.0, 130.0]
    assert protocol["expected_deliveries_per_run"] == 5
    grid = registration["registered_grid"]
    assert grid["delays_ms"] == [1.0, 3.0, 7.0]
    assert grid["dt_ms"] == [0.01, 0.005]
    assert grid["nonzero_points"] == 12
    assert grid["total_runs"] == 48
    assert registration["repeat_gate"]["required"] is True
    assert registration["numerical_convergence_gate"]["required"] is True
    assert registration["promotion"]["does_not_authorize"].startswith(
        "nonzero-network-execution"
    )


def test_timing_gate_addendum_changes_only_representation_identity() -> None:
    addendum = _yaml(
        "docs/validation-results/"
        "post2008-l5-sst-like-isolated-timing-gate-addendum-993.yaml"
    )
    assert addendum["corrected_execution_authorized"] is False
    assert addendum["scope"]["changed_field"].endswith("source_times_exact")
    definition = addendum["corrected_definition"]
    assert definition["name"] == "source_clock_ticks_exact"
    assert definition["raw_float_diagnostic"]["pass_threshold"] == "none"
    assert len(addendum["scope"]["unchanged_fields"]) == 9


def test_stage3_registration_fixes_complete_grid_and_classifications() -> None:
    registration = _yaml(
        "docs/validation-results/"
        "post2008-l5-sst-like-behavioral-grid-registration-997.yaml"
    )
    assert registration["network_execution_authorized"] is False
    grid = registration["fixed_grid"]
    assert grid["order"] == "delay-major-then-resource-fraction-ascending"
    assert grid["delays_ms"] == [1.0, 3.0, 7.0]
    assert grid["resource_fractions"] == [0.125, 0.25, 0.5, 1.0]
    assert grid["point_count"] == 12
    assert grid["repetitions_per_point"] == 2
    assert grid["execute_every_point"] is True
    assert grid["early_stop_on_behavioral_failure"] is False
    contract = registration["execution_contract"]
    assert contract["figure7"]["required_for_every_repetition"] is True
    assert contract["figure10"][
        "required_only_if_same-repetition-figure7-passes"
    ] is True
    assert contract["exact_repeat"]["required"] is True
    assert registration["prospective_region_classification"] == {
        "invariant_region": "all 12 points are exact_survival",
        "robust_region": (
            "all 12 points survive and at least one is robust_changed_survival"
        ),
        "mixed_region": (
            "at least one point survives and at least one point is failure"
        ),
        "no_survival_in_registered_region": "all 12 points are failure",
        "indeterminate": "any point is incomplete or has an engineering_stop",
    }


def test_stage4_registration_preserves_every_stage3_survivor_without_selection() -> None:
    registration = _yaml(
        "docs/validation-results/"
        "post2008-l5-sst-like-stage4-progression-registration-1002.yaml"
    )
    assert registration["network_execution_authorized"] is False
    eligible = registration["fixed_eligible_points"]
    assert eligible["count"] == len(eligible["points"]) == 7
    assert [point["point_id"] for point in eligible["points"]] == [
        "delay1p0-resource0p125",
        "delay1p0-resource0p25",
        "delay3p0-resource0p25",
        "delay3p0-resource0p5",
        "delay7p0-resource0p25",
        "delay7p0-resource0p5",
        "delay7p0-resource1p0",
    ]
    design = registration["execution_design"]
    assert design["substage_order"] == [
        "stage4a_learning_first_order",
        "stage4b_figure14",
        "stage4c_figure15",
        "stage4d_figure16",
    ]
    assert design["repetitions_per_point_per_substage"] == 2
    assert design["execute_all_seven_points_in_every_substage"] is True
    assert design["no_point_dropped_after_a_failed_substage"] is True
    assert design["failed_earlier_substage_cannot_be_compensated_by_later_pass"] is True
    figure15 = registration["stage4c_figure15"]
    assert figure15["source_identifiable_gate"]["peak_in_published_gamma_band_hz"] == [
        20.0,
        70.0,
    ]
    assert figure15["retained_noncompensatory_numeric_diagnostic"][
        "classic_baseline_status"
    ] == "fail"
