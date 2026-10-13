"""Synthetic identity joins retain all conflicts without cell qualification."""

from __future__ import annotations

import copy
import csv
import importlib.util
import io
import sys
import tarfile
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("identity_join", SCRIPTS / "join_patchseq_identity.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    assets = [{"asset_id": str(i), "path": f"unknown-{i}.nwb"} for i in range(4435)]
    assets[0]["path"] = "sub-10/sub-10_ses-1000_icephys.nwb"
    raw = dict.fromkeys(module.FIELDS, "")
    raw.update(cell_specimen_id="100", donor_id="10", ephys_session_id="1000", full_genotype="Vip-IRES-Cre", structure="VISp")
    rows = [{"source_row_number": 2, "raw_fields": raw,
             "canonical_ids": {field: module.canonical_id(raw[field]) for field in module.IDS}}]
    old = [{"specimen_id": i, "included": i <= 111} for i in range(1, 2334)]
    return assets, rows, old


def test_consistent_metadata_does_not_establish_class_layer_or_recording_identity():
    result = module.join_rows(*fixture())
    asset = result["assets"][0]
    assert asset["path_annotation_consistent"]
    assert not asset["internal_recording_identity_confirmed"]
    assert not asset["molecular_class_qualified"]
    assert not asset["measured_layer_qualified"]
    assert result["prior_specimen_ID_equalities"] == [100]
    assert result["prior_included_specimen_ID_equalities"] == [100]
    assert result["summary"]["assets_without_annotation_candidates"] == 4434


@pytest.mark.parametrize("value", ["", "nan", "001", "1.0", "+1", " 1", "١", None, True, 1])
def test_noncanonical_ids_not_coerced(value):
    assert module.canonical_id(value) is None


def test_positive_ascii_decimal_id():
    assert module.canonical_id("123456") == 123456


def test_multiple_candidates_retained_not_first_row_selected():
    assets, rows, old = fixture()
    second = copy.deepcopy(rows[0])
    second["source_row_number"] = 3
    rows.append(second)
    result = module.join_rows(assets, rows, old)
    assert len(result["assets"][0]["annotation_candidates"]) == 2
    assert not result["assets"][0]["path_annotation_consistent"]
    assert result["duplicate_source_session_IDs"] == [1000]
    assert result["duplicate_source_specimen_IDs"] == [100]


@pytest.mark.parametrize("fault", ["donor", "missing-cell", "unmatched-session", "path-shape", "duplicate-cell", "duplicate-catalog-session"])
def test_identity_conflicts_retained_without_promotion(fault):
    assets, rows, old = fixture()
    if fault == "donor":
        rows[0]["raw_fields"]["donor_id"] = "11"
    elif fault == "missing-cell":
        rows[0]["raw_fields"]["cell_specimen_id"] = ""
    elif fault == "unmatched-session":
        rows[0]["raw_fields"]["ephys_session_id"] = "999"
    elif fault == "path-shape":
        assets[0]["path"] = "sub-10/sub-11_ses-1000_icephys.nwb"
    elif fault == "duplicate-cell":
        second = copy.deepcopy(rows[0])
        second["source_row_number"] = 3
        second["raw_fields"]["ephys_session_id"] = "999"
        second["canonical_ids"]["ephys_session_id"] = 999
        rows.append(second)
    else:
        assets[1]["path"] = "sub-11/sub-11_ses-1000_icephys.nwb"
    rows[0]["canonical_ids"] = {field: module.canonical_id(rows[0]["raw_fields"][field]) for field in module.IDS}
    result = module.join_rows(assets, rows, old)
    assert not result["assets"][0]["path_annotation_consistent"]
    assert result["annotation_rows"] == rows
    assert not result["summary"]["biological_eligibility_established"]


def test_unmatched_source_rows_not_dropped():
    assets, rows, old = fixture()
    rows[0]["raw_fields"]["ephys_session_id"] = "999"
    rows[0]["canonical_ids"]["ephys_session_id"] = 999
    result = module.join_rows(assets, rows, old)
    assert result["unreferenced_source_row_numbers"] == [2]
    assert len(result["annotation_rows"]) == 1


def test_extra_fields_and_duplicate_source_numbers_reject():
    assets, rows, old = fixture()
    rows[0]["raw_fields"]["medical_conditions"] = "not-authorized"
    with pytest.raises(ValueError):
        module.join_rows(assets, rows, old)
    assets, rows, old = fixture()
    rows.append(copy.deepcopy(rows[0]))
    with pytest.raises(ValueError):
        module.join_rows(assets, rows, old)


def test_csv_projects_only_whitelist_without_numeric_response_fields(tmp_path):
    path = tmp_path / "source.tar"
    header = module.FIELDS + ["medical_conditions", "firing_rate"]
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(header)
    writer.writerow(["mouse", "100", "10", "1000", "sample", "VISp", "Vip", "id", "Vip alias", "secret", "99"])
    body = buffer.getvalue().encode()
    with tarfile.open(path, "w") as archive:
        info = tarfile.TarInfo("metadata.csv")
        info.size = len(body)
        archive.addfile(info, io.BytesIO(body))
    schema = {"annotation_row_values_interpreted": False, "CSV_member": "metadata.csv", "columns": header}
    rows = module.read_rows(path, schema)
    assert set(rows[0]["raw_fields"]) == set(module.FIELDS)
    assert "medical_conditions" not in rows[0]["raw_fields"]
    assert "firing_rate" not in rows[0]["raw_fields"]
    assert rows[0]["canonical_ids"]["ephys_session_id"] == 1000
