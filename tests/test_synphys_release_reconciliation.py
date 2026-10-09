"""Test release identity gates using synthetic data only, not source outcomes."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import sqlite3
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/synphys_release_reconciliation.py"
spec = importlib.util.spec_from_file_location("release_reconciliation", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("value", [None, 1, "sst", b"numeric-id-blob", 0.001, float("inf")])
def test_exact_source_values(value):
    assert module.same_value(value, value)


@pytest.mark.parametrize("left,right", [(None, float("nan")), (1, 1.0), (True, 1),
                                       (0.0, -0.0), (0.001, 0.0010000000001), ("sst", "Sst")])
def test_no_coercion_imputation_or_numeric_tolerance(left, right):
    assert not module.same_value(left, right)


def test_nan_pair_only_and_unsupported_types():
    assert module.same_value(float("nan"), float("nan"))
    with pytest.raises(ValueError, match="unsupported"):
        module.same_value([], [])


def test_order_independent_exact_projection_and_all_differences():
    small = [{"id": 1, "v": None}, {"id": 2, "v": 3.0}]
    assert module.compare_rows(small, small[::-1], ["id", "v"])["equal"]
    medium = [{"id": 2, "v": 4.0}, {"id": 3, "v": None}]
    result = module.compare_rows(small, medium, ["id", "v"])
    assert not result["equal"]
    assert len(result["differences"]) == 3
    assert [row["id"] for row in result["differences"]] == [1, 2, 3]


@pytest.mark.parametrize("rows", [[{"id": 1, "v": 0}] * 2, [{"id": None, "v": 0}],
                                 [{"id": True, "v": 0}], [{"id": 1, "v": 0, "data": b"x"}]])
def test_duplicate_or_invalid_projection_rejected(rows):
    with pytest.raises(ValueError):
        module.compare_rows(rows, rows, ["id", "v"])


def test_primary_key_authorizer_denies_unlisted_reads_and_writes():
    with sqlite3.connect(":memory:") as db:
        db.execute("CREATE TABLE cell (id INTEGER PRIMARY KEY, cre_type TEXT, data BLOB)")
        db.execute("INSERT INTO cell VALUES (1, 'sst', X'01')")
        db.execute("CREATE TABLE other (id INTEGER PRIMARY KEY)")
        db.commit()
        columns = {"cell": ["id", "cre_type"]}
        db.set_authorizer(module.make_authorizer(columns))
        columns["cell"].append("data")
        assert db.execute("SELECT id,cre_type FROM cell WHERE id=?", (1,)).fetchall() == [(1, "sst")]
        for query in ["SELECT data FROM cell", "SELECT id FROM other", "SELECT count(*) FROM cell",
                      "UPDATE cell SET cre_type='vip'", "DELETE FROM cell"]:
            with pytest.raises(sqlite3.DatabaseError):
                db.execute(query)


def test_chunked_hash(tmp_path):
    path = tmp_path / "synthetic.sqlite"
    raw = b"bounded-memory-source" * 7
    path.write_bytes(raw)
    assert module.stream_sha256(path, chunk_bytes=3) == hashlib.sha256(raw).hexdigest()
    for size in [0, -1, True, 0.5]:
        with pytest.raises(ValueError):
            module.stream_sha256(path, chunk_bytes=size)


def inventories():
    eid = 10
    bucket = int(hashlib.sha256(f"synphys-voltage-target-v1:{eid}".encode()).hexdigest(), 16) % 3
    targets = [{"fit_id": 4, "synapse_id": 3, "pair_id": 2, "experiment_id": eid,
                "partition": "validation" if bucket == 0 else "fitting"}]
    qc = {"pair": [{"id": 2, "experiment_id": eid, "pre_cell_id": 20, "post_cell_id": 21}],
          "synapse": [{"id": 3, "pair_id": 2}],
          "avg_response_fit": [{"id": 4, "synapse_id": 3}]}
    identity = {"cell": [{"id": 20, "experiment_id": eid}, {"id": 21, "experiment_id": eid}],
                "experiment": [{"id": eid, "slice_id": 30}], "slice": [{"id": 30}]}
    return targets, qc, identity


def test_selection_uses_original_parents_and_partition():
    selected = module.build_selection(*inventories())
    assert selected["cell"] == ("id", [20, 21])
    assert selected["resting_state_fit"] == ("synapse_id", [3])
    assert selected["cortical_cell_location"] == ("cell_id", [20, 21])
    assert selected["avg_response_fit"] == ("id", [4])


@pytest.mark.parametrize("mutation", ["duplicate-fit", "partition", "pair", "synapse", "cell", "slice"])
def test_source_lineage_changes_rejected(mutation):
    targets, qc, identity = copy.deepcopy(inventories())
    if mutation == "duplicate-fit":
        targets += targets
    elif mutation == "partition":
        targets[0]["partition"] = "changed"
    elif mutation == "pair":
        qc["pair"][0]["experiment_id"] = 11
    elif mutation == "synapse":
        qc["synapse"][0]["pair_id"] = 99
    elif mutation == "cell":
        identity["cell"][0]["experiment_id"] = 11
    else:
        identity["slice"] = []
    with pytest.raises(ValueError):
        module.build_selection(targets, qc, identity)


def synthetic_database(path, selection):
    with sqlite3.connect(path) as db:
        for table, columns in module.COLUMNS.items():
            definition = ','.join(f'"{name}" ' + ("INTEGER PRIMARY KEY" if name == "id"
                                                    else "BLOB" if name == "ic_pulse_ids"
                                                    else "INTEGER" if name.endswith("_id")
                                                    else "TEXT") for name in columns)
            db.execute(f'CREATE TABLE "{table}" ({definition})')
            key, ids = selection[table]
            for index, identity in enumerate(ids):
                row = {name: None for name in columns}
                row["id"] = identity if key == "id" else index + 100
                row[key] = identity
                if table == "resting_state_fit":
                    row["ic_pulse_ids"] = b"synthetic-blob-no-decoding"
                marks = ','.join("?" for _ in columns)
                db.execute(f'INSERT INTO "{table}" VALUES ({marks})', [row[name] for name in columns])


def test_two_synthetic_releases_and_retained_missing_rows(tmp_path):
    selection = module.build_selection(*inventories())
    small, medium = tmp_path / "small.sqlite", tmp_path / "medium.sqlite"
    for path in (small, medium):
        synthetic_database(path, selection)
    result = module.reconcile_selected(small, medium, selection)
    assert result["equal"]
    assert result["small_tables"]["resting_state_fit"][0]["ic_pulse_ids"] == b"synthetic-blob-no-decoding"
    with sqlite3.connect(medium) as db:
        db.execute("DELETE FROM cortical_cell_location WHERE cell_id=20")
        db.execute("UPDATE synapse SET latency='changed'")
    result = module.reconcile_selected(small, medium, selection)
    assert not result["equal"]
    assert not result["comparisons"]["synapse"]["equal"]
    assert not result["comparisons"]["cortical_cell_location"]["equal"]


@pytest.mark.parametrize("mutation", ["table", "key", "ids", "relationship"])
def test_projection_scope_rejected_before_open(tmp_path, mutation):
    selection = module.build_selection(*inventories())
    if mutation == "table":
        selection["recording"] = ("id", [1])
    elif mutation == "key":
        selection["cell"] = ("data", [1])
    elif mutation == "ids":
        selection["cell"] = ("id", [21, 20])
    else:
        selection["cell"] = ("experiment_id", [10])
    with pytest.raises(ValueError):
        module.project_selected(tmp_path / "must-not-be-opened.sqlite", selection)


def test_one_to_many_row_bound_is_failure_not_truncation(tmp_path):
    selection = module.build_selection(*inventories())
    path = tmp_path / "source.sqlite"
    synthetic_database(path, selection)
    with sqlite3.connect(path) as db:
        db.executemany("INSERT INTO conductance (id,synapse_id) VALUES (?,3)",
                       [(value,) for value in range(1000, 2000)])
    with pytest.raises(ValueError, match="row bound"):
        module.project_selected(path, selection)


def test_integer_bool_collision_not_hidden_by_deduplication():
    with pytest.raises(ValueError):
        module.positive_ids([1, True])
    targets, qc, identity = inventories()
    targets[0]["experiment_id"] = 10.0
    with pytest.raises(ValueError):
        module.build_selection(targets, qc, identity)


def test_oversized_blob_is_failure_not_truncation(tmp_path):
    selection = module.build_selection(*inventories())
    path = tmp_path / "source.sqlite"
    synthetic_database(path, selection)
    with sqlite3.connect(path) as db:
        db.execute("UPDATE resting_state_fit SET ic_pulse_ids=?", (b"x" * 1048577,))
    with pytest.raises(ValueError, match="blob bound"):
        module.project_selected(path, selection)
