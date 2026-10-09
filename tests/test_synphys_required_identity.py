"""An empty matching projection must never establish selected-ID coverage."""

from __future__ import annotations

import copy
import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/synphys_required_identity.py"
sys.path.insert(0, str(SCRIPT.parent))
spec = importlib.util.spec_from_file_location("required_identity", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    selection = {table: ("id", [1, 2]) for table in module.REQUIRED_TABLES}
    selection.update({table: (key, [1, 2]) for table, key in module.CHILD_KEYS.items()})
    tables = {table: [{"id": 1}, {"id": 2}] for table in module.REQUIRED_TABLES}
    tables.update({table: [] for table in module.CHILD_KEYS})
    return tables, selection


def test_complete_primary_ids_and_missing_optional_children():
    tables, selection = fixture()
    result = module.paired_required_id_coverage(tables, tables, selection)
    assert result["complete"]
    assert len(result["small"]["tables"]) == 6
    # Lack of a baseline/location/rested child remains separate source evidence;
    # do not relabel it as a missing primary identity or invent a child record.
    assert tables["conductance"] == []


@pytest.mark.parametrize("table", module.REQUIRED_TABLES)
def test_identically_missing_primary_ids_are_not_coverage(table):
    tables, selection = fixture()
    tables[table] = []
    result = module.paired_required_id_coverage(tables, copy.deepcopy(tables), selection)
    assert not result["complete"]
    assert result["small"]["tables"][table]["missing_ids"] == [1, 2]
    assert result["medium"]["tables"][table]["missing_ids"] == [1, 2]


def test_equal_counts_with_different_ids_do_not_pass():
    tables, selection = fixture()
    medium = copy.deepcopy(tables)
    medium["synapse"] = [{"id": 1}, {"id": 3}]
    result = module.paired_required_id_coverage(tables, medium, selection)
    assert not result["complete"]
    assert result["medium"]["tables"]["synapse"]["missing_ids"] == [2]
    assert result["medium"]["tables"]["synapse"]["unexpected_ids"] == [3]


@pytest.mark.parametrize("ids", [[1, 1], [1, True], [None], [0], [-1], [1.0]])
def test_invalid_or_duplicate_projected_ids_rejected(ids):
    tables, selection = fixture()
    tables["pair"] = [{"id": identity} for identity in ids]
    with pytest.raises(ValueError):
        module.required_id_coverage(tables, selection)


@pytest.mark.parametrize("mutation", ["empty-selection", "changed-key", "extra-table", "missing-table", "child-key"])
def test_no_scope_or_required_selection_relaxation(mutation):
    tables, selection = fixture()
    if mutation == "empty-selection":
        selection["cell"] = ("id", [])
    elif mutation == "changed-key":
        selection["cell"] = ("experiment_id", [1, 2])
    elif mutation == "extra-table":
        tables["recording"] = []
    elif mutation == "missing-table":
        del tables["synapse"]
    else:
        selection["conductance"] = ("id", [1, 2])
    with pytest.raises(ValueError):
        module.required_id_coverage(tables, selection)


def make_database(path):
    from synphys_release_reconciliation import COLUMNS

    with sqlite3.connect(path) as db:
        for table, columns in COLUMNS.items():
            fields = ','.join(f'"{column}" ' + ("INTEGER PRIMARY KEY" if column == "id" else "BLOB")
                              for column in columns)
            db.execute(f'CREATE TABLE "{table}" ({fields})')
            if table in module.REQUIRED_TABLES:
                db.executemany(f'INSERT INTO "{table}" (id) VALUES (?)', [(1,), (2,)])


def test_guarded_actual_projection_preserves_optional_missing_children(tmp_path):
    _, selection = fixture()
    small, medium = tmp_path / "small.sqlite", tmp_path / "medium.sqlite"
    for path in (small, medium):
        make_database(path)
    result = module.reconcile_with_required_ids(small, medium, selection)
    assert result["equal"]
    assert result["projected_values_equal"]
    assert result["required_primary_id_coverage"]["complete"]
    assert result["small_tables"]["conductance"] == []


@pytest.mark.parametrize("table", module.REQUIRED_TABLES)
def test_guarded_projection_rejects_same_missing_id_in_both_databases(tmp_path, table):
    _, selection = fixture()
    small, medium = tmp_path / "small.sqlite", tmp_path / "medium.sqlite"
    for path in (small, medium):
        make_database(path)
        with sqlite3.connect(path) as db:
            db.execute(f'DELETE FROM "{table}" WHERE id=2')
    result = module.reconcile_with_required_ids(small, medium, selection)
    assert result["projected_values_equal"] is True
    assert result["equal"] is False
    for side in ("small", "medium"):
        assert result["required_primary_id_coverage"][side]["tables"][table]["missing_ids"] == [2]


def test_value_mismatch_still_fails_with_complete_identities(tmp_path):
    _, selection = fixture()
    small, medium = tmp_path / "small.sqlite", tmp_path / "medium.sqlite"
    for path in (small, medium):
        make_database(path)
    with sqlite3.connect(medium) as db:
        db.execute("UPDATE cell SET cre_type='changed' WHERE id=1")
    result = module.reconcile_with_required_ids(small, medium, selection)
    assert result["required_primary_id_coverage"]["complete"] is True
    assert result["projected_values_equal"] is False
    assert result["equal"] is False
