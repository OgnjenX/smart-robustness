"""Outcome barriers and missing metadata must survive identity inventories."""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("identity_inventory", SCRIPTS / "inventory_synphys_identity.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    return {
        "slice": [{"id": 1, "ext_id": "s", "species": "mouse"}],
        "experiment": [{"id": 2, "ext_id": "e", "slice_id": 1,
                        "project_name": "project", "target_region": "intended"}],
        "cell": [{"id": 3, "ext_id": "c", "experiment_id": 2, "cre_type": "Sst",
                  "cell_class_nonsynaptic": "in", "target_layer": "5"}],
        "cortical_cell_location": [],
    }


def test_placeholder_never_replaces_measured_layer():
    result = module.reconstruct(fixture())
    assert result["no_measured_location"] == 1
    assert result["joint_labels"][0]["labels"]["measured_layers"] == []
    assert result["joint_labels"][0]["labels"]["placeholder_layer"] == "5"


def test_missing_and_multiple_joins_retained():
    tables = fixture()
    tables["experiment"] = []
    tables["cortical_cell_location"] = [
        {"id": 1, "cell_id": 3, "cortical_layer": "4"},
        {"id": 2, "cell_id": 3, "cortical_layer": "5"},
        {"id": 3, "cell_id": 99, "cortical_layer": None},
    ]
    result = module.reconstruct(tables)
    assert result["missing_experiments"] == result["missing_slices"] == 1
    assert result["multiple_locations"] == 1
    assert len(result["cells"][0]["locations"]) == 2
    assert len(result["orphan_locations"]) == 1


def test_duplicate_primary_id_rejected():
    tables = fixture()
    tables["cell"].append(tables["cell"][0].copy())
    with pytest.raises(ValueError, match="duplicate"):
        module.reconstruct(tables)


@pytest.mark.parametrize("query", ["SELECT cell_class FROM cell", "SELECT amplitude FROM synapse",
                                  "UPDATE cell SET cre_type='Vip'", "SELECT count(*) FROM synapse"])
def test_outcome_columns_and_mutations_denied(query):
    with sqlite3.connect(":memory:") as db:
        db.execute("CREATE TABLE cell (id INTEGER, cre_type TEXT, cell_class TEXT)")
        db.execute("CREATE TABLE synapse (amplitude REAL)")
        db.set_authorizer(module.authorizer)
        with pytest.raises(sqlite3.DatabaseError):
            db.execute(query).fetchall()


def test_metadata_projection_allowed():
    with sqlite3.connect(":memory:") as db:
        db.execute("CREATE TABLE cell (id INTEGER, cre_type TEXT, cell_class TEXT)")
        db.execute("INSERT INTO cell VALUES (1, 'Vip', 'in')")
        db.commit()
        db.set_authorizer(module.authorizer)
        assert db.execute("SELECT id, cre_type FROM cell").fetchall() == [(1, "Vip")]
