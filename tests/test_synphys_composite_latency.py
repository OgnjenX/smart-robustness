"""Distinct latency provenance must leave source failures untouched."""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("composite_latency", SCRIPTS / "audit_synphys_composite_latency.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    return {"resting_state_fit": [{"ic_amp": 0.001, "ic_latency": None, "ic_rise_time": 0.002,
                                   "ic_decay_tau": 0.02, "ic_nrmse": 0.1}],
            "conductance": [{"avg_baseline_potential": -0.067}],
            "lineage": [{"excitatory": True}]}


def test_original_missing_latency_never_repaired():
    record = fixture()
    assert module.coverage(record, [{"latency": 0.001}])
    assert record["resting_state_fit"][0]["ic_latency"] is None


@pytest.mark.parametrize("rows", [[], [{"latency": None}], [{"latency": float("nan")}], [{"latency": -0.1}], [{"latency": 0.001}] * 2])
def test_unresolved_shared_latency_rejected(rows):
    assert not module.coverage(fixture(), rows)


def test_no_missing_baseline_substitution():
    record = fixture()
    record["conductance"] = []
    assert not module.coverage(record, [{"latency": 0.001}])


@pytest.mark.parametrize("query", ["SELECT psp_amplitude FROM synapse", "DELETE FROM synapse"])
def test_column_scope_and_mutation_barrier(query):
    with sqlite3.connect(":memory:") as db:
        db.execute("CREATE TABLE synapse (id INTEGER, latency REAL, psp_amplitude REAL)")
        db.set_authorizer(module.authorizer)
        with pytest.raises(sqlite3.DatabaseError):
            db.execute(query).fetchall()
