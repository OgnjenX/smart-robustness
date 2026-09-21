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
