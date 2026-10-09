"""Schema reads must not expose biological rows."""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("small_schema", SCRIPTS / "acquire_synphys_small_schema.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_schema_only(tmp_path):
    path = tmp_path / "fixture.sqlite"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE synapse (id INTEGER PRIMARY KEY, amplitude REAL)")
        db.execute("INSERT INTO synapse VALUES (1, 12345)")
    report = module.inspect_schema(path)
    assert report["data_rows_read"] is False
    assert report["tables"][0]["name"] == "synapse"
    assert "12345" not in str(report)


@pytest.mark.parametrize("query", ["SELECT * FROM synapse", "SELECT count(*) FROM synapse", "DROP TABLE synapse"])
def test_authorizer_denies_data_and_mutation(query):
    with sqlite3.connect(":memory:") as db:
        db.execute("CREATE TABLE synapse (amplitude REAL)")
        db.set_authorizer(module.schema_authorizer)
        with pytest.raises(sqlite3.DatabaseError):
            db.execute(query).fetchall()
