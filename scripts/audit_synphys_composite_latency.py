"""Audit a distinct shared-latency composite, never repair missing source data."""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path

import yaml
from collect_synphys_source_metadata import digest, save

REGISTRATION = Path("docs/validation-results/post2008-synphys-composite-latency-registration-1118.yaml")


def finite(v):
    return isinstance(v, (int, float)) and math.isfinite(v)


def coverage(record, latency_rows):
    rs, bs = record["resting_state_fit"], record["conductance"]
    if len(rs) != 1 or len(bs) != 1 or len(latency_rows) != 1:
        return False
    r, b, latency = rs[0], bs[0]["avg_baseline_potential"], latency_rows[0]["latency"]
    fields = ["ic_amp", "ic_rise_time", "ic_decay_tau", "ic_nrmse"]
    return (all(finite(r[k]) for k in fields) and finite(b) and -0.1 <= b <= 0
            and finite(latency) and latency >= 0 and r["ic_rise_time"] > 0
            and r["ic_decay_tau"] > 0 and r["ic_nrmse"] >= 0
            and (r["ic_amp"] > 0 if record["lineage"][0]["excitatory"] else r["ic_amp"] < 0))


def summary(records):
    groups = {}
    for rec in records:
        first = rec["source_record"]["lineage"][0]
        labels = {k: v for k, v in first["stratum"].items() if k != "holding"}
        key = json.dumps(labels, sort_keys=True)
        g = groups.setdefault(key, {"fitting": set(), "validation": set(), "pairs": set()})
        if rec["composite_coverage"]:
            g[first["partition"]].add(first["experiment_id"])
            g["pairs"].add(first["pair_id"])
    return [{"stratum": json.loads(k), "pairs": len(g["pairs"]),
             "fitting_experiments": len(g["fitting"]), "validation_experiments": len(g["validation"]),
             "minimum_units_met": len(g["fitting"]) >= 5 and len(g["validation"]) >= 3}
            for k, g in sorted(groups.items())]


def authorizer(action, table, column, database, trigger):
    if action == sqlite3.SQLITE_READ:
        return sqlite3.SQLITE_OK if table == "synapse" and column in {"id", "latency"} else sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK if action == sqlite3.SQLITE_SELECT else sqlite3.SQLITE_DENY


def main():
    reg = yaml.safe_load(REGISTRATION.read_text())
    if reg["allowed_columns"] != {"synapse": ["id", "latency"]} or reg["new_numeric_reads_authorized"] is not True:
        raise ValueError("latency scope changed")
    for k in ("parameter_fitting_authorized", "cell_or_network_execution_authorized", "trace_reads_authorized"):
        if reg[k] is not False:
            raise ValueError("execution scope changed")
    if reg["minimum_units"] != {"fitting_experiments": 5, "validation_experiments": 3}:
        raise ValueError("unit gate changed")
    for k in ("database", "resting_inventory"):
        if digest(reg[k]) != reg[k + "_sha256"]:
            raise ValueError("source lineage changed")
    source = yaml.safe_load(Path(reg["resting_inventory"]).read_text())
    root = Path(reg["output_directory"])
    root.mkdir()
    manifest = {"registration_sha256": digest(REGISTRATION), "collector_sha256": digest(__file__)}
    try:
        records = []
        with sqlite3.connect(Path(reg["database"]).resolve().as_uri()+"?mode=ro&immutable=1", uri=True) as db:
            db.execute("PRAGMA query_only=ON")
            db.set_authorizer(authorizer)
            for r in source["records"]:
                rows = [{"id": v[0], "latency": v[1]} for v in db.execute("SELECT id,latency FROM synapse WHERE id=?", (r["synapse_id"],))]
                interval = None
                if len(rows) == 1 and finite(rows[0]["latency"]):
                    value = rows[0]["latency"]
                    interval = [value - 0.0001, value + 0.0001]
                records.append({"source_record": r, "shared_latency_rows": rows,
                                "source_fit_search_interval_seconds": interval,
                                "composite_coverage": coverage(r, rows)})
        manifest.update(records=records, summary=summary(records), status="composite-coverage-audited-no-fitting")
    except (sqlite3.Error, ValueError) as exc:
        manifest.update(status="failure-retained", error=str(exc))
    save(root / "manifest.yaml", manifest)
    print(manifest["status"])


if __name__ == "__main__":
    main()
