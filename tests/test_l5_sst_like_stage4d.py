"""Synthetic-only tests; these do not execute the registered network holdout."""

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from smart_robustness.validation import l5_sst_like_stage4d as contract

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs/validation-results"


def registrations():
    return (
        yaml.safe_load(
            (DOCS / "post2008-l5-sst-like-stage4-progression-registration-1002.yaml").read_text()
        ),
        yaml.safe_load((DOCS / "calibrated-figure16-holdout-registration-813.yaml").read_text()),
    )


def outcomes():
    return [{"repetition": r, "gates": dict.fromkeys(contract.GATES, True)} for r in range(2)]


def earlier():
    return {
        "stage4a": "learning_first_order_survival",
        "stage4b": "figure14_survival",
        "stage4c": "figure15_survival",
    }


def test_registered_protocol_uses_one_geometry_per_area():
    reg, parent = registrations()
    protocol = contract.validate_protocol(reg, parent)
    assert protocol.recording_ms / protocol.recording_sample_ms == 1000
    assert reg["stage4d_figure16"]["geometry_seeds"] == [
        parent["figure16_protocol"]["geometry_seed_v1"],
        parent["figure16_protocol"]["geometry_seed_v2"],
    ]


@pytest.mark.parametrize(
    "key,value",
    [
        ("geometry_seeds", [17, 18]),
        ("recording_sample_ms", 2.0),
        ("input_learning_state", "route-free-weights"),
        ("intervention_scope", "v1-and-v2"),
        ("frequency_bands_hz", [[2, 20], [20, 100]]),
        ("backend", "brian2_numpy"),
    ],
)
def test_extension_protocol_drift_rejected(key, value):
    reg, parent = registrations()
    reg["stage4d_figure16"][key] = value
    with pytest.raises(ValueError):
        contract.validate_protocol(reg, parent)


def test_parent_reduction_and_inventory_cannot_change():
    for key, value in [("geometry_seed_v2", 18), ("higher_area_signal", "mean_v2_all")]:
        reg, parent = registrations()
        parent["figure16_protocol"][key] = value
        with pytest.raises(ValueError):
            contract.validate_protocol(reg, parent)
    reg, parent = registrations()
    reg["execution_design"]["no_point_dropped_after_a_failed_substage"] = False
    with pytest.raises(ValueError):
        contract.validate_protocol(reg, parent)


def test_joint_classification_never_rescues_earlier_failure():
    original = outcomes()
    assert contract.classify_point(original, earlier=earlier())["joint_classification"] == (
        "complete_progression_survival"
    )
    for stage, failed, expected in [
        ("stage4a", "failure", "learning_or_first_order_failure"),
        ("stage4b", "figure14_failure", "bounded_partial_survival"),
        ("stage4c", "figure15_failure", "bounded_partial_survival"),
    ]:
        old = {**earlier(), stage: failed}
        assert contract.classify_point(original, earlier=old)["joint_classification"] == expected
    assert original == outcomes()


def test_sample_finite_and_exact_repeat_failures_are_engineering_stops():
    for gate in contract.GATES[:2]:
        data = outcomes()
        for item in data:
            item["gates"][gate] = False
        assert (
            contract.classify_point(data, earlier=earlier())["classification"] == "engineering_stop"
        )
    data = outcomes()
    data[1]["raw_field"] = [1.0]
    assert contract.classify_point(data, earlier=earlier())["classification"] == "engineering_stop"


def test_spectral_failure_is_not_engineering_failure():
    data = outcomes()
    for item in data:
        item["gates"][contract.GATES[2]] = False
    classified = contract.classify_point(data, earlier=earlier())
    assert classified["classification"] == "figure16_failure"
    assert classified["joint_classification"] == "bounded_partial_survival"


def test_incomplete_repetitions_and_unassessed_earlier_inputs_rejected():
    with pytest.raises(ValueError):
        contract.classify_point(outcomes()[:1], earlier=earlier())
    data = deepcopy(outcomes())
    data.reverse()
    with pytest.raises(ValueError):
        contract.classify_point(data, earlier=earlier())
    with pytest.raises(ValueError):
        contract.classify_point(outcomes(), earlier={**earlier(), "stage4a": "engineering_stop"})


def test_full_catalog_builder_receives_only_registered_route(monkeypatch):
    calls = []
    monkeypatch.setattr(contract, "make_l5_sst_like_sector_builder", lambda **kw: calls.append(kw))
    contract.full_network_builder({"total_conductance_nS": 48.0, "delay_ms": 7.0})
    assert calls == [
        {
            "base_builder": contract.classic_sector.build_full_smart_network,
            "total_conductance_nS": 48.0,
            "delay_ms": 7.0,
        }
    ]
