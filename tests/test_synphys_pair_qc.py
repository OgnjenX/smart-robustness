"""QC availability is not numerical fidelity or outcome-based identity."""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("pair_qc", SCRIPTS / "inventory_synphys_pair_qc.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def cell(cid, kind, layer="5"):
    return {"cell_id": cid, "experiment": {"id": 1, "target_region": "VisP",
            "project_name": "mouse V1 coarse matrix"}, "slice": {"species": "mouse"},
            "locations": [{"cortical_layer": layer}],
            "cell": {"cre_type": kind, "cell_class_nonsynaptic": "ex" if kind == "E" else "in"}}


def fixture():
    return {
        "pair": [{"id": 1, "experiment_id": 1, "pre_cell_id": 1, "post_cell_id": 2,
                  "n_ex_test_spikes": 0, "n_in_test_spikes": 10}],
        "synapse": [{"id": 7, "pair_id": 1}],
        "avg_response_fit": [{"id": 8, "synapse_id": 7, "poly_synapse_id": None,
                              "clamp_mode": "ic", "holding": -55, "manual_qc_pass": 1,
                              "n_averaged_responses": 10}],
        "dynamics": [],
    }, [cell(1, "vip", "2/3"), cell(2, "sst")]


def test_descriptive_support_and_holding_retained():
    tables, cells = fixture()
    result = module.reconstruct(tables, cells)
    assert result["pairs"][0]["descriptive_support"] is True
    assert result["pairs"][0]["average_fits"][0]["holding"] == -55
    assert result["strata"][0]["labels"]["route"] == "VIP-to-SST5"
    assert result["strata"][0]["labels"]["pre_layer"] == "2/3"


@pytest.mark.parametrize("change", ["mixed", "missing_layer", "experiment", "probe", "qc", "multiple"])
def test_no_promotion_for_unresolved_or_insufficient_support(change):
    tables, cells = fixture()
    if change == "mixed":
        cells[0]["cell"]["cre_type"] = "vip,sim1"
    elif change == "missing_layer":
        cells[0]["locations"] = []
    elif change == "experiment":
        tables["pair"][0]["experiment_id"] = 99
    elif change == "probe":
        tables["pair"][0]["n_in_test_spikes"] = None
    elif change == "qc":
        tables["avg_response_fit"][0]["manual_qc_pass"] = None
    else:
        tables["synapse"].append({"id": 9, "pair_id": 1})
    assert module.reconstruct(tables, cells)["pairs"][0]["descriptive_support"] is False


@pytest.mark.parametrize("query", ["SELECT psp_amplitude FROM synapse", "SELECT has_synapse FROM pair",
                                  "SELECT fit_amp FROM avg_response_fit", "DELETE FROM pair"])
def test_response_and_mutation_barriers(query):
    with sqlite3.connect(":memory:") as db:
        db.execute("CREATE TABLE pair (has_synapse BOOLEAN)")
        db.execute("CREATE TABLE synapse (psp_amplitude REAL)")
        db.execute("CREATE TABLE avg_response_fit (fit_amp REAL)")
        db.set_authorizer(module.authorizer)
        with pytest.raises(sqlite3.DatabaseError):
            db.execute(query).fetchall()
