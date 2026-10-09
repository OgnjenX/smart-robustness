"""Read sealed QC/availability columns; response numbers remain blocked."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path

import yaml
from collect_synphys_source_metadata import digest, save

REGISTRATION = Path("docs/validation-results/post2008-synphys-pair-QC-registration-1107.yaml")
COLUMNS = {
    "pair": ["id", "experiment_id", "pre_cell_id", "post_cell_id", "n_ex_test_spikes", "n_in_test_spikes"],
    "synapse": ["id", "pair_id"],
    "avg_response_fit": ["id", "synapse_id", "poly_synapse_id", "clamp_mode", "holding", "manual_qc_pass", "n_averaged_responses"],
    "dynamics": ["id", "pair_id", "qc_pass", "n_source_events"],
}


def authorizer(action, table, column, database, trigger):
    if action == sqlite3.SQLITE_READ:
        return sqlite3.SQLITE_OK if column in COLUMNS.get(table, []) else sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK if action == sqlite3.SQLITE_SELECT else sqlite3.SQLITE_DENY


def identity(cell, experiment_id):
    if cell is None or cell["experiment"] is None or cell["slice"] is None:
        return None
    ex = cell["experiment"]
    if (ex["id"] != experiment_id or cell["slice"]["species"] != "mouse"
        or ex["target_region"] != "VisP"
        or ex["project_name"] not in {"mouse V1 coarse matrix", "mouse V1 pre-production"}):
        return None
    loc = cell["locations"]
    if len(loc) != 1 or loc[0]["cortical_layer"] is None:
        return None
    c = cell["cell"]
    if c["cell_class_nonsynaptic"] == "ex":
        kind = "E"
    elif c["cell_class_nonsynaptic"] == "in" and c["cre_type"] in {"sst", "vip", "pvalb"}:
        kind = c["cre_type"]
    else:
        return None
    return {"kind": kind, "layer": loc[0]["cortical_layer"], "project": ex["project_name"]}


def route(pre, post):
    if pre is None or post is None:
        return None
    if post["kind"] == "sst" and post["layer"] == "5":
        return {"E": "E-to-SST5", "vip": "VIP-to-SST5"}.get(pre["kind"])
    if post["kind"] == "E" and post["layer"] == "5":
        if pre["kind"] == "sst" and pre["layer"] == "5":
            return "SST5-to-E5"
        if pre["kind"] == "pvalb":
            return "PV-to-E5"
    return None


def reconstruct(tables, cells):
    for rows in tables.values():
        if len({v["id"] for v in rows}) != len(rows):
            raise ValueError("duplicate primary ID")
    by_cell = {v["cell_id"]: v for v in cells}
    if len(by_cell) != len(cells):
        raise ValueError("duplicate identity cell ID")
    syn, fits, dynamics = {}, {}, {}
    for name, key, target in [("synapse", "pair_id", syn), ("avg_response_fit", "synapse_id", fits),
                              ("dynamics", "pair_id", dynamics)]:
        for row in tables[name]:
            target.setdefault(row[key], []).append(row)
    records, strata = [], {}
    for pair in tables["pair"]:
        pre = identity(by_cell.get(pair["pre_cell_id"]), pair["experiment_id"])
        post = identity(by_cell.get(pair["post_cell_id"]), pair["experiment_id"])
        label = route(pre, post)
        summaries = syn.get(pair["id"], [])
        averages = [f for s in summaries for f in fits.get(s["id"], [])]
        relevant = None if pre is None else pair["n_ex_test_spikes" if pre["kind"] == "E" else "n_in_test_spikes"]
        positive_probe = relevant is not None and relevant > 0
        passed = [v for v in averages if v["manual_qc_pass"] == 1 and v["clamp_mode"] == "ic"
                  and v["n_averaged_responses"] is not None and v["n_averaged_responses"] > 0]
        support = label is not None and positive_probe and len(summaries) == 1 and bool(passed)
        records.append({"pair_id": pair["id"], "pre_identity": pre, "post_identity": post,
                        "route": label, "synapses": summaries, "average_fits": averages,
                        "dynamics": dynamics.get(pair["id"], []), "descriptive_support": support})
        if label is not None:
            key = json.dumps({"route": label, "project": pre["project"],
                              "pre_layer": pre["layer"], "post_layer": post["layer"]}, sort_keys=True)
            counts = strata.setdefault(key, Counter())
            counts["pairs"] += 1
            counts["positive_relevant_probe"] += positive_probe
            counts["single_synapse_record"] += len(summaries) == 1
            counts["manual_QC_IC_fit"] += bool(passed)
            counts["descriptive_support"] += support
    pair_ids = {v["id"] for v in tables["pair"]}
    syn_ids = {v["id"] for v in tables["synapse"]}
    return {"pairs": records,
            "strata": [{"labels": json.loads(k), "counts": dict(v)} for k, v in sorted(strata.items())],
            "orphan_synapses": [v for v in tables["synapse"] if v["pair_id"] not in pair_ids],
            "orphan_dynamics": [v for v in tables["dynamics"] if v["pair_id"] not in pair_ids],
            "unlinked_average_fits": [v for v in tables["avg_response_fit"] if v["synapse_id"] not in syn_ids]}


def main():
    reg = yaml.safe_load(REGISTRATION.read_text())
    if reg["allowed_columns"] != COLUMNS or reg["QC_and_summary_availability_reads_authorized"] is not True:
        raise ValueError("QC projection changed")
    for key in ("numeric_response_values_reads_authorized", "trace_reads_authorized",
                "parameter_fitting_authorized", "network_execution_authorized"):
        if reg[key] is not False:
            raise ValueError("response boundary changed")
    for key in ("database", "identity_inventory"):
        if digest(reg[key]) != reg[key + "_sha256"]:
            raise ValueError("source lineage changed")
    root = Path(reg["output_directory"])
    root.mkdir()
    manifest = {"registration_sha256": digest(REGISTRATION), "collector_sha256": digest(__file__),
                "numeric_response_values_read": False}
    try:
        cells = yaml.safe_load(Path(reg["identity_inventory"]).read_text())["reconstruction"]["cells"]
        tables = {}
        with sqlite3.connect(Path(reg["database"]).resolve().as_uri()+"?mode=ro&immutable=1", uri=True) as db:
            db.execute("PRAGMA query_only=ON")
            db.set_authorizer(authorizer)
            for name, cols in COLUMNS.items():
                query = 'SELECT '+','.join(f'"{c}"' for c in cols)+f' FROM "{name}" ORDER BY id'
                tables[name] = [dict(zip(cols, row, strict=True)) for row in db.execute(query)]
        manifest.update(tables=tables, reconstruction=reconstruct(tables, cells), status="QC-inventory-complete")
    except (sqlite3.Error, ValueError) as exc:
        manifest.update(status="failure-retained", error=str(exc))
    save(root / "manifest.yaml", manifest)
    print(manifest["status"])


if __name__ == "__main__":
    main()
