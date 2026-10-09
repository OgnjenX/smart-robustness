"""Exercise the sealed runner end-to-end on synthetic releases only."""

from __future__ import annotations

import hashlib
import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest
import yaml

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("sealed_reconciliation_runner", SCRIPTS / "run_synphys_release_reconciliation.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def save(path, value):
    path.write_text(yaml.safe_dump(value, sort_keys=False))


def source_fixture(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("results").mkdir()
    paths = {key: str(tmp_path / (key + (".sqlite" if key.endswith("database") else ".yaml")))
             for key in module.SOURCE_PATHS}
    monkeypatch.setattr(module, "SOURCE_PATHS", paths)
    eid = 10
    bucket = int(hashlib.sha256(f"synphys-voltage-target-v1:{eid}".encode()).hexdigest(), 16) % 3
    targets = [{"fit_id": 1000 + index, "synapse_id": index % 92 + 1, "pair_id": index % 92 + 1,
                "experiment_id": eid, "partition": "validation" if bucket == 0 else "fitting"}
               for index in range(98)]
    qc = {"pair": [{"id": value, "experiment_id": eid, "pre_cell_id": 20, "post_cell_id": 21}
                   for value in range(1, 93)],
          "synapse": [{"id": value, "pair_id": value} for value in range(1, 93)],
          "avg_response_fit": [{"id": row["fit_id"], "synapse_id": row["synapse_id"]} for row in targets]}
    identities = {"cell": [{"id": 20, "experiment_id": eid}, {"id": 21, "experiment_id": eid}],
                  "experiment": [{"id": eid, "slice_id": 30}], "slice": [{"id": 30}]}
    selection = module.build_selection(targets, qc, identities)
    for key in ("small_database", "medium_database"):
        with sqlite3.connect(paths[key]) as db:
            for table, columns in module.COLUMNS.items():
                fields = ','.join(f'"{column}" ' + ("INTEGER PRIMARY KEY" if column == "id"
                                                   else "BLOB" if column == "ic_pulse_ids"
                                                   else "INTEGER" if column.endswith("_id") else "TEXT")
                                  for column in columns)
                db.execute(f'CREATE TABLE "{table}" ({fields})')
                relationship, ids = selection[table]
                for index, identity in enumerate(ids):
                    row = {column: None for column in columns}
                    row["id"] = identity if relationship == "id" else index + 10000
                    row[relationship] = identity
                    if table == "resting_state_fit":
                        row["ic_pulse_ids"] = b"synthetic-pulse-ID-blob"
                    marks = ','.join("?" for _ in columns)
                    db.execute(f'INSERT INTO "{table}" VALUES ({marks})', [row[column] for column in columns])
    save(Path(paths["numeric_inventory"]), {"records": targets})
    save(Path(paths["qc_inventory"]), {"tables": qc})
    save(Path(paths["identity_inventory"]), {"tables": identities})
    size = Path(paths["medium_database"]).stat().st_size
    monkeypatch.setattr(module, "EXPECTED_MEDIUM_BYTES", size)
    medium_hash = module.stream_sha256(Path(paths["medium_database"]))
    acquisition = {"status": "medium-acquired-schema-only", "row_values_read": False,
                   "client_exit_code": 0, "bytes": size, "sha256": medium_hash,
                   "raw_path": paths["medium_database"], "schema": {"synthetic": "no-real-source"}}
    save(Path(paths["acquisition"]), acquisition)
    verification = {"status": "medium-object-and-schema-independently-verified",
                    "verifier_sha256": module.stream_sha256(SCRIPTS / "verify_synphys_medium_acquisition.py"),
                    "collector_imported": False, "data_rows_read": False,
                    "checks": {key: "passed" for key in
                               ("byte_count", "sha256", "sqlite_signature", "schema", "source_integrity")},
                    "acquisition_manifest": paths["acquisition"],
                    "acquisition_manifest_sha256": module.stream_sha256(Path(paths["acquisition"])),
                    "identity": {"bytes": size, "sha256": medium_hash, "sqlite_signature": True},
                    "schema": acquisition["schema"]}
    save(Path(paths["verification"]), verification)
    save(Path(paths["acquisition_assessment"]), {
        "status": "object-acquired-schema-verified-selected-reconciliation-pending",
        "result": paths["acquisition"],
        "result_sha256": module.stream_sha256(Path(paths["acquisition"])),
        "verification_result": paths["verification"],
        "verification_result_sha256": module.stream_sha256(Path(paths["verification"])),
        "data_rows_read": False})
    sources = {key: {"path": path, "sha256": module.stream_sha256(Path(path))} for key, path in paths.items()}
    monkeypatch.setattr(module, "KNOWN_HASHES", {key: source["sha256"] for key, source in sources.items()})
    reg = {"selected_reconciliation_authorized": True, **{key: False for key in module.DENIED},
           "sources": sources, "allowed_columns": module.COLUMNS,
           "selected_counts": {"average_fit_ids": 98, "synapse_ids": 92},
           "bounds": {"rows_per_parent": 1000, "rows_per_table": 10000, "blob_bytes": 1048576,
                      "payload_bytes_per_release": 33554432},
           "runner_sha256": module.stream_sha256(SCRIPTS / "run_synphys_release_reconciliation.py"),
           "helper_sha256": module.stream_sha256(SCRIPTS / "synphys_release_reconciliation.py"),
           "verifier_sha256": module.stream_sha256(SCRIPTS / "verify_synphys_medium_acquisition.py"),
           "output_directory": "results/synphys-selected-reconciliation-synthetic"}
    registration = tmp_path / "registration.yaml"
    save(registration, reg)
    return registration, reg, acquisition, verification


def test_full_synthetic_run_and_duplicate_output_preservation(tmp_path, monkeypatch):
    registration, reg, _, _ = source_fixture(tmp_path, monkeypatch)
    report = module.run(registration)
    assert report["status"] == "exact-selected-equivalence-independent-assessment-pending"
    assert len(report["original_target_lineage"]) == 98
    assert len(report["selection"]["synapse"][1]) == 92
    assert report["baseline_joins_performed"] is False
    assert len(report["small_tables"]["resting_state_fit"]) == 92
    output = Path(reg["output_directory"]) / "manifest.yaml"
    previous = output.read_bytes()
    with pytest.raises(FileExistsError):
        module.run(registration)
    assert output.read_bytes() == previous


@pytest.mark.parametrize("key", module.DENIED)
def test_scope_expansion_rejected_before_output(tmp_path, monkeypatch, key):
    registration, reg, _, _ = source_fixture(tmp_path, monkeypatch)
    reg[key] = True
    save(registration, reg)
    with pytest.raises(ValueError, match="scope"):
        module.run(registration)
    assert not Path(reg["output_directory"]).exists()


def test_input_hash_change_retained_without_projections(tmp_path, monkeypatch):
    registration, reg, _, _ = source_fixture(tmp_path, monkeypatch)
    with Path(reg["sources"]["numeric_inventory"]["path"]).open("a") as handle:
        handle.write("# changed after seal\n")
    report = module.run(registration)
    assert report["status"] == "engineering-failure-retained"
    assert "input content changed" in report["error"]
    assert "small_tables" not in report


@pytest.mark.parametrize("field,value", [("status", "not-verified"), ("collector_imported", True),
                                        ("data_rows_read", True), ("checks", {})])
def test_independent_gate_never_inferred_from_finished_download(tmp_path, monkeypatch, field, value):
    _, reg, acquisition, verification = source_fixture(tmp_path, monkeypatch)
    verification[field] = value
    with pytest.raises(ValueError, match="independent"):
        module.validate_acquisition(reg, acquisition, verification)


def test_changed_summary_retained_even_with_resealed_synthetic_source(tmp_path, monkeypatch):
    registration, reg, acquisition, verification = source_fixture(tmp_path, monkeypatch)
    path = Path(module.SOURCE_PATHS["medium_database"])
    with sqlite3.connect(path) as db:
        db.execute("UPDATE synapse SET latency='different' WHERE id=1")
    digest = module.stream_sha256(path)
    acquisition["sha256"] = digest
    save(Path(module.SOURCE_PATHS["acquisition"]), acquisition)
    verification["identity"]["sha256"] = digest
    verification["acquisition_manifest_sha256"] = module.stream_sha256(Path(module.SOURCE_PATHS["acquisition"]))
    save(Path(module.SOURCE_PATHS["verification"]), verification)
    assessment_path = Path(module.SOURCE_PATHS["acquisition_assessment"])
    assessment = yaml.safe_load(assessment_path.read_text())
    assessment["result_sha256"] = module.stream_sha256(Path(module.SOURCE_PATHS["acquisition"]))
    assessment["verification_result_sha256"] = module.stream_sha256(Path(module.SOURCE_PATHS["verification"]))
    save(assessment_path, assessment)
    for key, source in reg["sources"].items():
        source["sha256"] = module.stream_sha256(Path(source["path"]))
        module.KNOWN_HASHES[key] = source["sha256"]
    save(registration, reg)
    report = module.run(registration)
    assert report["status"] == "selected-release-differences-retained"
    assert report["comparisons"]["synapse"]["differences"] == [
        {"id": 1, "kind": "changed-value", "column": "latency", "small": None, "medium": "different"}]


@pytest.mark.parametrize("field,value", [("status", "unassessed"), ("data_rows_read", True),
                                        ("result_sha256", "changed"), ("verification_result_sha256", "changed")])
def test_acquisition_assessment_gate_required(tmp_path, monkeypatch, field, value):
    _, reg, _, _ = source_fixture(tmp_path, monkeypatch)
    assessment = yaml.safe_load(Path(module.SOURCE_PATHS["acquisition_assessment"]).read_text())
    assessment[field] = value
    with pytest.raises(ValueError, match="assessment"):
        module.validate_assessment(reg, assessment)


def test_input_mutation_during_projection_cannot_promote_equality(tmp_path, monkeypatch):
    registration, _, _, _ = source_fixture(tmp_path, monkeypatch)
    original = module.reconcile_selected

    def mutate_after_projection(small, medium, selection):
        result = original(small, medium, selection)
        with medium.open("ab") as handle:
            handle.write(b"mutation-after-projection")
        return result

    monkeypatch.setattr(module, "reconcile_selected", mutate_after_projection)
    report = module.run(registration)
    assert report["status"] == "engineering-failure-retained"
    assert report["equal"] is True
    assert "changed during reconciliation" in report["error"]
    assert report["baseline_joins_performed"] is False
