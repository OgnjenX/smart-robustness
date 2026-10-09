"""Numeric targets retain failures and global experiment-level holdouts."""

from __future__ import annotations

import hashlib
import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("numeric_targets", SCRIPTS / "extract_synphys_numeric_targets.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_partition_exact_global_rule():
    for eid in range(40):
        v = int(hashlib.sha256(f"synphys-voltage-target-v1:{eid}".encode()).hexdigest(), 16) % 3
        assert module.partition(eid) == ("validation" if v == 0 else "fitting")


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -1, 0])
def test_positive_waveform_times_required(value):
    row = {"fit_amp": 0.001, "fit_xoffset": 0.001, "fit_rise_time": value,
           "fit_decay_tau": 0.01, "nrmse": 0.2}
    assert module.quality(row, True)["fit_rise_time"] is False
    assert row["fit_rise_time"] is value


def test_inhibitory_sign_not_repaired():
    row = {"fit_amp": 0.001, "fit_xoffset": 0.001, "fit_rise_time": 0.001,
           "fit_decay_tau": 0.01, "nrmse": 0.2}
    assert module.quality(row, False)["fit_amp"] is False
    assert row["fit_amp"] == 0.001


def test_minimum_experiment_units_not_fit_counts():
    records = [{"stratum": {"route": "test"}, "partition": arm, "pair_id": i,
                "experiment_id": 1 if arm == "fitting" else 2, "quality_flags": {"valid": True}}
               for arm in ["fitting", "validation"] for i in range(20)]
    assert module.summarize(records)[0]["minimum_units_met"] is False


@pytest.mark.parametrize("query", ["SELECT effective_conductance FROM conductance", "SELECT avg_data FROM avg_response_fit", "DELETE FROM avg_response_fit"])
def test_out_of_scope_reads_rejected(query):
    with sqlite3.connect(":memory:") as db:
        db.execute("CREATE TABLE conductance (effective_conductance REAL)")
        db.execute("CREATE TABLE avg_response_fit (avg_data BLOB)")
        db.set_authorizer(module.authorizer)
        with pytest.raises(sqlite3.DatabaseError):
            db.execute(query).fetchall()
