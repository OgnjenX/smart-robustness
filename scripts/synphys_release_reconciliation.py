"""Synthetic-tested reconciliation helpers; no database execution entry point.

Real source queries require a separate sealed registration and independently
verified medium acquisition. These helpers do not authorize baseline joins.
"""

from __future__ import annotations

import hashlib
import math
import sqlite3
import struct
from pathlib import Path

COLUMNS = {
    "slice": ["id", "ext_id", "species"],
    "experiment": ["id", "ext_id", "slice_id", "project_name", "target_region"],
    "cell": ["id", "ext_id", "experiment_id", "cre_type", "cell_class_nonsynaptic", "target_layer"],
    "cortical_cell_location": ["id", "cell_id", "cortical_layer"],
    "pair": ["id", "experiment_id", "pre_cell_id", "post_cell_id", "n_ex_test_spikes", "n_in_test_spikes"],
    "synapse": ["id", "pair_id", "latency"],
    "avg_response_fit": ["id", "synapse_id", "poly_synapse_id", "clamp_mode", "holding",
                         "manual_qc_pass", "n_averaged_responses", "fit_amp", "fit_xoffset",
                         "fit_rise_time", "fit_decay_tau", "nrmse"],
    "resting_state_fit": ["id", "synapse_id", "ic_amp", "ic_latency", "ic_rise_time",
                          "ic_decay_tau", "ic_nrmse", "ic_pulse_ids"],
    "conductance": ["id", "synapse_id", "avg_baseline_potential"],
}


def stream_sha256(path: Path, chunk_bytes: int = 1048576) -> str:
    """Hash large releases without allocating the complete database."""
    if not isinstance(chunk_bytes, int) or isinstance(chunk_bytes, bool) or chunk_bytes <= 0:
        raise ValueError("positive integer chunk size required")
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            checksum.update(chunk)
    return checksum.hexdigest()


def same_value(left, right):
    """Exact typed values; NaN matches NaN, never a missing value.

    Retain signed zero and reject bool/int coercion. No numeric tolerance,
    rounding, unit conversion, string normalization or null imputation.
    """
    if type(left) is not type(right):
        return False
    if isinstance(left, float):
        if math.isnan(left) or math.isnan(right):
            return math.isnan(left) and math.isnan(right)
        return struct.pack("!d", left) == struct.pack("!d", right)
    if left is None or isinstance(left, (int, str, bytes)):
        return left == right
    raise ValueError("unsupported source value type")


def index_rows(rows, columns):
    if not columns or len(set(columns)) != len(columns) or "id" not in columns:
        raise ValueError("unique columns including primary ID required")
    indexed = {}
    for row in rows:
        if set(row) != set(columns):
            raise ValueError("row projection differs from registered columns")
        identity = row["id"]
        if type(identity) is not int or identity <= 0 or identity in indexed:
            raise ValueError("invalid or duplicate primary ID")
        indexed[identity] = row
    return indexed


def compare_rows(small_rows, medium_rows, columns):
    """Retain all mismatches, independent of row order; never select a best fit."""
    small = index_rows(small_rows, columns)
    medium = index_rows(medium_rows, columns)
    differences = []
    for identity in sorted(small.keys() | medium.keys()):
        if identity not in small or identity not in medium:
            differences.append({"id": identity, "kind": "missing-row",
                                "missing_from": "small" if identity not in small else "medium"})
            continue
        for column in columns:
            if not same_value(small[identity][column], medium[identity][column]):
                differences.append({"id": identity, "kind": "changed-value", "column": column,
                                    "small": small[identity][column],
                                    "medium": medium[identity][column]})
    return {"equal": not differences, "small_rows": len(small), "medium_rows": len(medium),
            "differences": differences}


def make_authorizer(columns):
    """Freeze exact projection; deny writes, traces and all but bounded substr.

    SQLite emits an empty-column READ for INTEGER PRIMARY KEY projections.
    Permit that callback only for the explicitly registered tables; it does
    not grant access to any named column outside the projection.
    """
    allowed = {table: frozenset(projection) for table, projection in columns.items()}
    if not allowed or any("id" not in projection for projection in allowed.values()):
        raise ValueError("every registered table requires primary ID")

    def authorize(action, table, column, database, trigger):
        if action == sqlite3.SQLITE_READ:
            permitted = table in allowed and (column == "" or column in allowed[table])
            return sqlite3.SQLITE_OK if permitted else sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_FUNCTION:
            return sqlite3.SQLITE_OK if column == "substr" else sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK if action == sqlite3.SQLITE_SELECT else sqlite3.SQLITE_DENY

    return authorize


def positive_ids(values):
    values = list(values)
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("invalid selected ID")
    return sorted(set(values))


def build_selection(targets, qc_tables, identity_tables):
    """Use only sealed small-release inventories, never medium values.

    Preserve experiment assignments and require all source lineage relations
    to agree. Duplicate target IDs are errors; repeated parent IDs are normal.
    """
    if not targets:
        raise ValueError("empty selected target inventory")
    for name in ("fit_id", "synapse_id", "pair_id", "experiment_id"):
        positive_ids(row[name] for row in targets)
    fit_ids = positive_ids(row["fit_id"] for row in targets)
    if len(fit_ids) != len(targets):
        raise ValueError("duplicate selected fit ID")
    def source_index(tables, name):
        rows = tables[name]
        if not rows:
            raise ValueError("empty required source inventory")
        return index_rows(rows, list(rows[0]))

    pairs = source_index(qc_tables, "pair")
    synapses = source_index(qc_tables, "synapse")
    fits = source_index(qc_tables, "avg_response_fit")
    cells = source_index(identity_tables, "cell")
    experiments = source_index(identity_tables, "experiment")
    slices = source_index(identity_tables, "slice")
    cell_ids, slice_ids = set(), set()
    for row in targets:
        pair = pairs.get(row["pair_id"])
        synapse = synapses.get(row["synapse_id"])
        fit = fits.get(row["fit_id"])
        if (pair is None or synapse is None or fit is None
            or synapse["pair_id"] != pair["id"] or fit["synapse_id"] != synapse["id"]
            or pair["experiment_id"] != row["experiment_id"]):
            raise ValueError("selected summary lineage differs")
        eid = row["experiment_id"]
        bucket = int(hashlib.sha256(f"synphys-voltage-target-v1:{eid}".encode()).hexdigest(), 16) % 3
        if row["partition"] != ("validation" if bucket == 0 else "fitting"):
            raise ValueError("original experiment partition changed")
        ex = experiments.get(eid)
        if ex is None or ex["slice_id"] not in slices:
            raise ValueError("missing selected experiment or slice")
        slice_ids.add(ex["slice_id"])
        for key in ("pre_cell_id", "post_cell_id"):
            cell = cells.get(pair[key])
            if cell is None or cell["experiment_id"] != eid:
                raise ValueError("selected cell lineage differs")
            cell_ids.add(cell["id"])
    synapse_ids = positive_ids(row["synapse_id"] for row in targets)
    return {
        "slice": ("id", positive_ids(slice_ids)),
        "experiment": ("id", positive_ids(row["experiment_id"] for row in targets)),
        "cell": ("id", positive_ids(cell_ids)),
        "cortical_cell_location": ("cell_id", positive_ids(cell_ids)),
        "pair": ("id", positive_ids(row["pair_id"] for row in targets)),
        "synapse": ("id", synapse_ids),
        "avg_response_fit": ("id", fit_ids),
        "resting_state_fit": ("synapse_id", synapse_ids),
        "conductance": ("synapse_id", synapse_ids),
    }


def project_selected(path, selection):
    """Bound selected projections under an exact-column read barrier.

    This callable is tested only on synthetic databases until separately
    registered. It reads no pulse-response or recording tables or waveforms.
    """
    if set(selection) != set(COLUMNS):
        raise ValueError("selected table scope differs")
    for table, (key, ids) in selection.items():
        if key not in COLUMNS[table] or ids != positive_ids(ids) or len(ids) > 10000:
            raise ValueError("selected key or ID scope differs")
        expected_key = {"cortical_cell_location": "cell_id", "resting_state_fit": "synapse_id",
                        "conductance": "synapse_id"}.get(table, "id")
        if key != expected_key:
            raise ValueError("unregistered relationship key")
    tables, payload_bytes = {}, 0
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True) as db:
        db.execute("PRAGMA query_only=ON")
        db.set_authorizer(make_authorizer(COLUMNS))
        for table, (key, ids) in selection.items():
            columns = COLUMNS[table]
            expressions = ["substr(ic_pulse_ids,1,1048577)" if column == "ic_pulse_ids"
                           else f'"{column}"' for column in columns]
            query = ('SELECT ' + ','.join(expressions)
                     + f' FROM "{table}" WHERE "{key}"=? ORDER BY id LIMIT 1001')
            rows = []
            for identity in ids:
                for count, values in enumerate(db.execute(query, (identity,)), start=1):
                    if count > 1000 or len(rows) >= 10000:
                        raise ValueError("selected row bound exceeded")
                    if any(isinstance(value, bytes) and len(value) > 1048576 for value in values):
                        raise ValueError("selected blob bound exceeded")
                    payload_bytes += sum(len(value) if isinstance(value, bytes)
                                         else len(value.encode()) if isinstance(value, str) else 8
                                         for value in values)
                    if payload_bytes > 33554432:
                        raise ValueError("selected aggregate payload bound exceeded")
                    rows.append(dict(zip(columns, values, strict=True)))
            index_rows(rows, columns)
            tables[table] = rows
    return tables


def reconcile_selected(small_path, medium_path, selection):
    """Return full selected projections and every exact difference.

    No baseline gate is promoted by this function; a separate independent
    check of the archived evidence is required even if every value matches.
    """
    small = project_selected(small_path, selection)
    medium = project_selected(medium_path, selection)
    comparisons = {table: compare_rows(small[table], medium[table], columns)
                   for table, columns in COLUMNS.items()}
    return {"small_tables": small, "medium_tables": medium, "comparisons": comparisons,
            "equal": all(result["equal"] for result in comparisons.values())}
