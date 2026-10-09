"""Fail-closed model metadata checks and explicit missing-source boundaries."""

from __future__ import annotations

import pytest

from smart_robustness.validation.sst_vip_model_identity import reconcile, validate_batch


def model(model_id=10, specimen=1, template=None):
    return {
        "id": model_id,
        "specimen_id": specimen,
        "neuronal_model_template": template or {"id": 395310498, "name": "LIF"},
        "well_known_files": [],
    }


def payload(rows):
    return {"success": True, "total_rows": len(rows), "msg": rows}


def test_retains_no_model_specimens_and_sorts():
    rows = [{"id": 2, "neuronal_models": []}, {"id": 1, "neuronal_models": [model()]}]
    assert [r["id"] for r in validate_batch(payload(rows), [1, 2])] == [1, 2]


@pytest.mark.parametrize(
    "rows,requested",
    [
        ([{"id": 1, "neuronal_models": []}], [1, 2]),
        ([{"id": 2, "neuronal_models": []}], [1]),
        ([{"id": 1, "neuronal_models": []}] * 2, [1, 2]),
        ([{"id": 1}], [1]),
        ([{"id": 1, "neuronal_models": [model(specimen=2)]}], [1]),
        ([{"id": 1, "neuronal_models": [model(), model()]}], [1]),
    ],
)
def test_incomplete_wrong_missing_and_duplicate_metadata_reject(rows, requested):
    with pytest.raises((ValueError, TypeError)):
        validate_batch(payload(rows), requested)


def test_counts_reconcile_and_no_selection_authorized():
    rows = validate_batch(payload([{"id": 1, "neuronal_models": [model()]}]), [1])
    result = reconcile(
        rows, [{"specimen_id": 1, "model_counts": {"all_active": 0, "perisomatic": 0, "glif": 1}}]
    )
    assert result["count_discrepancies"] == []
    assert result["family_model_totals"]["glif"] == 1
    assert not result["candidate_selection_authorized"]


def test_unknown_templates_and_count_drift_retained():
    rows = [{"id": 1, "neuronal_models": [model(template={"id": 99, "name": "Unknown"})]}]
    result = reconcile(
        rows, [{"specimen_id": 1, "model_counts": {"all_active": 0, "perisomatic": 0, "glif": 1}}]
    )
    assert result["unknown_templates"][0]["model_id"] == 10
    assert result["count_discrepancies"][0]["resolved_count"] == 0


def test_missing_resource_metadata_is_reported():
    m = model()
    m["well_known_files"] = [{"id": 9}]
    result = reconcile(
        [{"id": 1, "neuronal_models": [m]}],
        [{"specimen_id": 1, "model_counts": {"all_active": 0, "perisomatic": 0, "glif": 1}}],
    )
    assert result["incomplete_resource_metadata"]
