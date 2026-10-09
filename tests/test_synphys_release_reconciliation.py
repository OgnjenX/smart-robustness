"""Test release identity gates using synthetic data only, not source outcomes."""

from __future__ import annotations

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
