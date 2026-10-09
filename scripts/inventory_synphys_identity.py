"""Inventory nonsynaptic metadata with column-level outcome read barriers."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path

import yaml
from collect_synphys_source_metadata import digest, save

REGISTRATION = Path(
    "docs/validation-results/post2008-synphys-identity-inventory-registration-1104.yaml"
)
COLUMNS = {
    "slice": ["id", "ext_id", "species"],
    "experiment": ["id", "ext_id", "slice_id", "project_name", "target_region"],
    "cell": ["id", "ext_id", "experiment_id", "cre_type", "cell_class_nonsynaptic", "target_layer"],
    "cortical_cell_location": ["id", "cell_id", "cortical_layer"],
}


def authorizer(action, table, column, database, trigger):
    if action == sqlite3.SQLITE_READ:
        return sqlite3.SQLITE_OK if column in COLUMNS.get(table, []) else sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_SELECT:
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def reconstruct(tables):
    indexed = {}
    for name, rows in tables.items():
        indexed[name] = {r["id"]: r for r in rows}
        if len(indexed[name]) != len(rows):
            raise ValueError("duplicate metadata primary ID")
    locations = {}
    for row in tables["cortical_cell_location"]:
        locations.setdefault(row["cell_id"], []).append(row)
    cells, joint = [], Counter()
    for cell in tables["cell"]:
        ex = indexed["experiment"].get(cell["experiment_id"])
        sl = None if ex is None else indexed["slice"].get(ex["slice_id"])
        loc = locations.get(cell["id"], [])
        identity = {
            "cell_id": cell["id"], "experiment": ex, "slice": sl,
            "cell": cell, "locations": loc,
            "missing_experiment": ex is None, "missing_slice": sl is None,
            "location_count": len(loc),
        }
        cells.append(identity)
        label = {
            "species": None if sl is None else sl["species"],
            "project": None if ex is None else ex["project_name"],
            "intended_region": None if ex is None else ex["target_region"],
            "cre_type": cell["cre_type"], "nonsynaptic_class": cell["cell_class_nonsynaptic"],
            "measured_layers": [v["cortical_layer"] for v in loc],
            "placeholder_layer": cell["target_layer"],
        }
        joint[json.dumps(label, sort_keys=True)] += 1
    return {
        "cells": cells,
        "joint_labels": [{"labels": json.loads(k), "cells": v} for k, v in sorted(joint.items())],
        "row_counts": {k: len(v) for k, v in tables.items()},
        "missing_experiments": sum(c["missing_experiment"] for c in cells),
        "missing_slices": sum(c["missing_slice"] for c in cells),
        "no_measured_location": sum(c["location_count"] == 0 for c in cells),
        "multiple_locations": sum(c["location_count"] > 1 for c in cells),
        "orphan_locations": [r for r in tables["cortical_cell_location"] if r["cell_id"] not in indexed["cell"]],
    }


def inventory(path):
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True) as db:
        db.execute("PRAGMA query_only=ON")
        db.set_authorizer(authorizer)
        tables = {}
        for name, cols in COLUMNS.items():
            query = 'SELECT ' + ','.join(f'"{c}"' for c in cols) + f' FROM "{name}" ORDER BY id'
            tables[name] = [dict(zip(cols, row, strict=True)) for row in db.execute(query)]
    return {"tables": tables, "reconstruction": reconstruct(tables)}


def main():
    reg = yaml.safe_load(REGISTRATION.read_text())
    if reg["allowed_columns"] != COLUMNS or reg["metadata_values_authorized"] is not True:
        raise ValueError("metadata projection changed")
    for key in ("synaptic_or_intrinsic_outcome_reads_authorized", "QC_outcome_reads_authorized",
                "parameter_fitting_authorized", "network_execution_authorized"):
        if reg[key] is not False:
            raise ValueError("metadata-only boundary changed")
    for key in ("database", "parent_manifest"):
        if digest(reg[key]) != reg[key + "_sha256"]:
            raise ValueError("source lineage changed")
    root = Path(reg["output_directory"])
    root.mkdir()
    result = {"registration_sha256": digest(REGISTRATION), "collector_sha256": digest(__file__),
              "database_sha256": reg["database_sha256"], "outcome_values_read": False}
    try:
        result.update(inventory(Path(reg["database"])))
        result["status"] = "metadata-inventory-complete"
    except (sqlite3.Error, ValueError) as exc:
        result.update(status="failure-retained", error=str(exc))
    save(root / "manifest.yaml", result)
    print(result["status"])


if __name__ == "__main__":
    main()
