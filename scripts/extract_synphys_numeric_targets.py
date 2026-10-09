"""Freeze voltage-response targets and global experiment holdout assignments."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path

import yaml
from collect_synphys_source_metadata import digest, save

REGISTRATION = Path("docs/validation-results/post2008-synphys-numeric-target-registration-1110.yaml")
COLUMNS = ["id", "synapse_id", "fit_amp", "fit_xoffset", "fit_rise_time", "fit_decay_tau", "nrmse"]


def partition(experiment_id):
    key = f"synphys-voltage-target-v1:{experiment_id}".encode()
    bucket = int(hashlib.sha256(key).hexdigest(), 16) % 3
    return "validation" if bucket == 0 else "fitting"


def quality(row, excitatory):
    flags = {}
    for field in COLUMNS[2:]:
        v = row[field]
        finite = isinstance(v, (int, float)) and math.isfinite(v)
        if field == "fit_amp":
            flags[field] = finite and (v > 0 if excitatory else v < 0)
        elif field in {"fit_rise_time", "fit_decay_tau"}:
            flags[field] = finite and v > 0
        else:
            flags[field] = finite and v >= 0
    return flags


def authorizer(action, table, column, database, trigger):
    if action == sqlite3.SQLITE_READ:
        return sqlite3.SQLITE_OK if table == "avg_response_fit" and column in COLUMNS else sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK if action == sqlite3.SQLITE_SELECT else sqlite3.SQLITE_DENY


def selected(qc):
    pairs = {v["id"]: v for v in qc["tables"]["pair"]}
    records = []
    seen = set()
    for pair in qc["reconstruction"]["pairs"]:
        if not pair["descriptive_support"]:
            continue
        eid = pairs[pair["pair_id"]]["experiment_id"]
        for fit in pair["average_fits"]:
            if fit["clamp_mode"] != "ic" or fit["manual_qc_pass"] != 1 or not fit["n_averaged_responses"]:
                continue
            if fit["n_averaged_responses"] < 0:
                continue
            if fit["id"] in seen:
                raise ValueError("duplicate selected fit")
            seen.add(fit["id"])
            a, b = pair["pre_identity"], pair["post_identity"]
            records.append({"fit_id": fit["id"], "pair_id": pair["pair_id"],
                            "experiment_id": eid, "synapse_id": fit["synapse_id"],
                            "partition": partition(eid), "excitatory": a["kind"] == "E",
                            "stratum": {"project": a["project"], "route": pair["route"],
                                        "pre_layer": a["layer"], "post_layer": b["layer"],
                                        "holding": fit["holding"]}})
    return records


def summarize(records):
    groups = defaultdict(list)
    for r in records:
        groups[json.dumps(r["stratum"], sort_keys=True)].append(r)
    summary = []
    for key, rows in sorted(groups.items()):
        counts = {}
        for arm in ("fitting", "validation"):
            arm_rows = [r for r in rows if r["partition"] == arm]
            valid = [r for r in arm_rows if all(r["quality_flags"].values())]
            counts[arm] = {"fits": len(arm_rows), "pairs": len({r["pair_id"] for r in arm_rows}),
                           "experiments": len({r["experiment_id"] for r in arm_rows}),
                           "quality_valid_fits": len(valid),
                           "quality_valid_experiments": len({r["experiment_id"] for r in valid})}
        summary.append({"stratum": json.loads(key), "counts": counts,
                        "minimum_units_met": counts["fitting"]["quality_valid_experiments"] >= 5
                        and counts["validation"]["quality_valid_experiments"] >= 3})
    return summary


def main():
    reg = yaml.safe_load(REGISTRATION.read_text())
    if reg["allowed_columns"] != {"avg_response_fit": COLUMNS} or reg["numeric_extraction_authorized"] is not True:
        raise ValueError("numeric projection changed")
    if any(reg[k] is not False for k in ("parameter_fitting_authorized", "cell_or_network_execution_authorized", "trace_reads_authorized")):
        raise ValueError("extraction-only scope changed")
    if (reg["minimum_units"]["fitting_experiments"], reg["minimum_units"]["validation_experiments"]) != (5, 3):
        raise ValueError("minimum units changed")
    if (reg["partition"]["unit"], reg["partition"]["validation_bucket"],
        reg["partition"]["fitting_buckets"], reg["partition"]["experiment_leakage_allowed"]) != ("experiment_id", 0, [1, 2], False):
        raise ValueError("partition contract changed")
    for key in ("database", "QC_inventory"):
        if digest(reg[key]) != reg[key + "_sha256"]:
            raise ValueError("source lineage changed")
    root = Path(reg["output_directory"])
    root.mkdir()
    manifest = {"registration_sha256": digest(REGISTRATION), "collector_sha256": digest(__file__)}
    try:
        qc = yaml.load(Path(reg["QC_inventory"]).read_text(), Loader=yaml.CSafeLoader)
        records = selected(qc)
        with sqlite3.connect(Path(reg["database"]).resolve().as_uri()+"?mode=ro&immutable=1", uri=True) as db:
            db.execute("PRAGMA query_only=ON")
            db.set_authorizer(authorizer)
            query = 'SELECT '+','.join(f'"{c}"' for c in COLUMNS)+' FROM avg_response_fit WHERE id=?'
            for r in records:
                values = db.execute(query, (r["fit_id"],)).fetchall()
                if len(values) != 1:
                    raise ValueError("numeric fit identity unresolved")
                row = dict(zip(COLUMNS, values[0], strict=True))
                if row["synapse_id"] != r["synapse_id"]:
                    raise ValueError("numeric fit join changed")
                r.update(values=row, quality_flags=quality(row, r["excitatory"]))
        manifest.update(records=records, summary=summarize(records), status="targets-extracted-no-fitting")
    except (sqlite3.Error, ValueError) as exc:
        manifest.update(status="failure-retained", error=str(exc))
    save(root / "manifest.yaml", manifest)
    print(manifest["status"])


if __name__ == "__main__":
    main()
