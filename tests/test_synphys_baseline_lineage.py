"""Synthetic join cases retain missingness and raw QC without outcome selection."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/synphys_baseline_lineage.py"
spec = importlib.util.spec_from_file_location("baseline_lineage", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    context = {"pair_id": 7, "experiment_id": 10, "pre_cell_id": 20, "post_cell_id": 21}
    tables = {
        "cell": [{"id": 20, "experiment_id": 10, "electrode_id": 30},
                 {"id": 21, "experiment_id": 10, "electrode_id": 31}],
        "electrode": [{"id": 30, "experiment_id": 10}, {"id": 31, "experiment_id": 10}],
        "sync_rec": [{"id": 40, "experiment_id": 10}],
        "recording": [{"id": 50, "sync_rec_id": 40, "electrode_id": 30},
                      {"id": 51, "sync_rec_id": 40, "electrode_id": 31}],
        "pulse_response": [{"id": 1, "recording_id": 51, "stim_pulse_id": 60, "pair_id": 7,
                            "ex_qc_pass": 1, "in_qc_pass": 0}],
        "patch_clamp_recording": [{"id": 70, "recording_id": 51, "clamp_mode": "ic",
                                   "baseline_potential": -0.068, "qc_pass": 1}],
        "stim_pulse": [{"id": 60, "recording_id": 50, "cell_id": 20, "previous_pulse_dt": 9.0,
                        "qc_pass": 1, "n_spikes": 1, "first_spike_time": 0.001}],
    }
    return context, tables


def test_complete_lineage_is_not_waveform_or_biological_promotion():
    context, tables = fixture()
    result = module.join_original_pulse_metadata([1], context, tables)
    assert result["all_contributor_identities_complete"]
    assert result["records"][0]["baseline_value"] == -0.068
    assert not result["records"][0]["response_duration_confirmed"]
    assert not result["biological_eligibility_established"]


@pytest.mark.parametrize("table", module.COLUMNS)
def test_missing_parent_keeps_original_contributor(table):
    context, tables = fixture()
    tables[table] = []
    result = module.join_original_pulse_metadata([1], context, tables)
    assert result["original_contributor_count"] == 1
    assert not result["all_contributor_identities_complete"]
    assert result["records"][0]["original_pulse_response_id"] == 1


@pytest.mark.parametrize("table", module.COLUMNS)
def test_multiple_parents_are_not_first_row_selection(table):
    context, tables = fixture()
    tables[table].append(copy.deepcopy(tables[table][0]))
    result = module.join_original_pulse_metadata([1], context, tables)
    assert not result["all_contributor_identities_complete"]


@pytest.mark.parametrize("table,index,field,value", [
    ("pulse_response", 0, "pair_id", 99), ("cell", 1, "electrode_id", 99),
    ("cell", 0, "experiment_id", 99), ("electrode", 1, "experiment_id", 99),
    ("sync_rec", 0, "experiment_id", 99), ("stim_pulse", 0, "cell_id", 99),
    ("recording", 0, "sync_rec_id", 99),
])
def test_conflicting_identity_not_used_as_complete_source_state(table, index, field, value):
    context, tables = fixture()
    tables[table][index][field] = value
    result = module.join_original_pulse_metadata([1], context, tables)
    assert not result["all_contributor_identities_complete"]


@pytest.mark.parametrize("interval,passes", [(8.0, False), (8.0001, True), (float("inf"), True),
                                          (float("nan"), False), (None, False)])
def test_rest_interval_source_semantics(interval, passes):
    context, tables = fixture()
    tables["stim_pulse"][0]["previous_pulse_dt"] = interval
    result = module.join_original_pulse_metadata([1], context, tables)
    assert result["records"][0]["source_state_metadata_flags"]["previous_pulse_interval_gt_8s"] is passes


def test_raw_QC_and_multispike_values_do_not_reselect_contributors():
    context, tables = fixture()
    tables["stim_pulse"][0].update(qc_pass=0, n_spikes=2)
    tables["patch_clamp_recording"][0].update(qc_pass=0, baseline_potential=None)
    result = module.join_original_pulse_metadata([1], context, tables)
    assert result["original_contributor_count"] == 1
    assert result["all_contributor_identities_complete"]
    record = result["records"][0]
    assert record["baseline_value"] is None
    assert record["parents"]["stim_pulse"][0]["n_spikes"] == 2
    assert not record["biological_eligibility_established"]


def test_no_waveform_projection_and_no_duplicate_original_ids():
    context, tables = fixture()
    tables["pulse_response"][0]["data"] = b"not-authorized"
    with pytest.raises(ValueError, match="projection"):
        module.join_original_pulse_metadata([1], context, tables)
    context, tables = fixture()
    with pytest.raises(ValueError, match="unique"):
        module.join_original_pulse_metadata([1, 1], context, tables)


@pytest.mark.parametrize("table,index,field", [
    ("pulse_response", 0, "pair_id"), ("cell", 1, "electrode_id"),
    ("cell", 0, "experiment_id"), ("electrode", 1, "experiment_id"),
    ("sync_rec", 0, "experiment_id"), ("stim_pulse", 0, "cell_id"),
])
def test_numerically_equal_float_identity_is_not_certified(table, index, field):
    context, tables = fixture()
    tables[table][index][field] = float(tables[table][index][field])
    result = module.join_original_pulse_metadata([1], context, tables)
    assert not result["all_contributor_identities_complete"]
    assert any(type(row.get(field)) is float
               for parents in result["records"][0]["parents"].values() for row in parents)


@pytest.mark.parametrize("value", [True, False, 1.0, "1", None, 0, -1])
def test_identity_comparison_does_not_coerce(value):
    assert not module.same_identity(value, 1)


def test_repeated_recording_keeps_pulse_weights_and_missing_contributor():
    context, tables = fixture()
    second = copy.deepcopy(tables["pulse_response"][0])
    second["id"] = 2
    tables["pulse_response"].append(second)
    result = module.join_original_pulse_metadata([2, 1, 3], context, tables)
    assert result["original_contributor_count"] == 3
    assert [row["original_pulse_response_id"] for row in result["records"]] == [2, 1, 3]
    assert [row["baseline_value"] for row in result["records"]] == [-0.068, -0.068, None]
    assert [row["identity_complete"] for row in result["records"]] == [True, True, False]
    assert not result["all_contributor_identities_complete"]
