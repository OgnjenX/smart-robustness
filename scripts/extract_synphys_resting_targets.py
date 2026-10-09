"""Retain rested responses and measured baselines without state substitution."""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path

import yaml
from collect_synphys_source_metadata import digest, save

REGISTRATION = Path("docs/validation-results/post2008-synphys-resting-target-registration-1114.yaml")
COLUMNS = {
    "resting_state_fit": ["id", "synapse_id", "ic_amp", "ic_latency", "ic_rise_time", "ic_decay_tau", "ic_nrmse"],
    "conductance": ["id", "synapse_id", "avg_baseline_potential"],
}


def authorizer(action, table, column, database, trigger):
    if action == sqlite3.SQLITE_READ:
        return sqlite3.SQLITE_OK if column in COLUMNS.get(table, []) else sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK if action == sqlite3.SQLITE_SELECT else sqlite3.SQLITE_DENY


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def coverage(rested, baselines, excitatory):
    if len(rested) != 1 or len(baselines) != 1:
        return False
    r = rested[0]
    values = [r[k] for k in COLUMNS["resting_state_fit"][2:]]
    b = baselines[0]["avg_baseline_potential"]
    return (all(finite(v) for v in values) and finite(b) and -0.1 <= b <= 0
            and (r["ic_amp"] > 0 if excitatory else r["ic_amp"] < 0)
            and r["ic_latency"] >= 0 and r["ic_rise_time"] > 0
            and r["ic_decay_tau"] > 0 and r["ic_nrmse"] >= 0)


def main():
    reg = yaml.safe_load(REGISTRATION.read_text())
    if reg["allowed_columns"] != COLUMNS or reg["numeric_extraction_authorized"] is not True:
        raise ValueError("projection changed")
    for key in ("effective_conductance_numeric_reads_authorized", "parameter_fitting_authorized",
                "cell_or_network_execution_authorized", "trace_reads_authorized"):
        if reg[key] is not False:
            raise ValueError("source-state scope changed")
    for key in ("database", "target_inventory"):
        if digest(reg[key]) != reg[key + "_sha256"]:
            raise ValueError("lineage changed")
    targets = yaml.safe_load(Path(reg["target_inventory"]).read_text())["records"]
    groups = {}
    for r in targets:
        groups.setdefault(r["synapse_id"], []).append(r)
    root = Path(reg["output_directory"])
    root.mkdir()
    manifest = {"registration_sha256": digest(REGISTRATION), "collector_sha256": digest(__file__)}
    try:
        records = []
        with sqlite3.connect(Path(reg["database"]).resolve().as_uri()+"?mode=ro&immutable=1", uri=True) as db:
            db.execute("PRAGMA query_only=ON")
            db.set_authorizer(authorizer)
            for sid, lineage in sorted(groups.items()):
                if len({(r["pair_id"], r["experiment_id"], r["partition"], r["excitatory"]) for r in lineage}) != 1:
                    raise ValueError("synapse lineage conflicting")
                record = {"synapse_id": sid, "lineage": lineage}
                for table, cols in COLUMNS.items():
                    sql = 'SELECT '+','.join(f'"{c}"' for c in cols)+f' FROM "{table}" WHERE synapse_id=? ORDER BY id'
                    record[table] = [dict(zip(cols, row, strict=True)) for row in db.execute(sql, (sid,))]
                record["descriptive_coverage"] = coverage(record["resting_state_fit"], record["conductance"], lineage[0]["excitatory"])
                records.append(record)
        manifest.update(records=records, status="rested-state-coverage-extracted-no-fitting")
    except (ValueError, sqlite3.Error) as exc:
        manifest.update(status="failure-retained", error=str(exc))
    save(root / "manifest.yaml", manifest)
    print(manifest["status"])


if __name__ == "__main__":
    main()
