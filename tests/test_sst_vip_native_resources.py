"""Physiology identity and missing-resource checks without model execution."""

from __future__ import annotations

import pytest

from smart_robustness.validation.sst_vip_native_resources import (
    reconcile_physiology,
    validate_physiology_batch,
)


def row():
    return {"id": 1, "ephys_result": None, "ephys_sweeps": [], "neuron_reconstructions": []}


def test_missing_resources_remain_explicit_and_no_sweep_selection():
    r = row()
    got = validate_physiology_batch({"success": True, "total_rows": 1, "msg": [r]}, [1])
    result = reconcile_physiology(
        got, [{"specimen_id": 1, "electrophysiology_file_id": None, "morphology_file_id": None}]
    )
    assert len(result["missing_resources"]) == 2
    assert result["sweep_records"] == 0
    assert not result["response_guided_sweep_selection"]


@pytest.mark.parametrize(
    "change",
    [
        {"id": 2},
        {"ephys_sweeps": None},
        {"neuron_reconstructions": None},
        {"ephys_result": []},
    ],
)
def test_identity_and_schema_changes_reject(change):
    with pytest.raises((ValueError, TypeError)):
        validate_physiology_batch({"success": True, "total_rows": 1, "msg": [row() | change]}, [1])


def test_partial_failed_duplicate_response_rejected():
    for payload in (
        {"success": False, "total_rows": 1, "msg": [row()]},
        {"success": True, "total_rows": 2, "msg": [row()]},
        {"success": True, "total_rows": 2, "msg": [row(), row()]},
    ):
        with pytest.raises(ValueError):
            validate_physiology_batch(payload, [1, 2])


def test_file_identity_drift_not_hidden_and_all_sweeps_kept():
    r = row()
    r["ephys_result"] = {
        "well_known_files": [{"id": 4, "well_known_file_type": {"name": "NWBDownload"}}]
    }
    r["ephys_sweeps"] = [{"sweep_number": 3}, {"sweep_number": 5, "passed": False}]
    result = reconcile_physiology(
        [r], [{"specimen_id": 1, "electrophysiology_file_id": 9, "morphology_file_id": None}]
    )
    assert result["identity_discrepancies"][0]["resolved_file_ids"] == [4]
    assert result["sweep_records"] == 2
    assert r["ephys_sweeps"][1]["passed"] is False
