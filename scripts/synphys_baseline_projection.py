"""Bounded metadata-only lineage projection; real use needs a separate seal.

No command-line entry point, download, waveform access or fit execution.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from synphys_baseline_lineage import COLUMNS, positive

MAX_IDS = 10000
MAX_ROWS_PER_TABLE = 10000
MAX_ROWS_PER_PARENT = 1000
MAX_PAYLOAD_BYTES = 32 * 1024 * 1024
MAX_STRING_BYTES = 4096


def metadata_authorizer(action, table, column, database, trigger):
    if action == sqlite3.SQLITE_READ:
        permitted = (database == "main" and trigger is None and table in COLUMNS
                     and (column == "" or column in COLUMNS[table]))
        return sqlite3.SQLITE_OK if permitted else sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK if action == sqlite3.SQLITE_SELECT else sqlite3.SQLITE_DENY


def project_original_pulse_metadata(path, pulse_ids, context):
    """Fetch only parents reached from fixed original pulse and selected cell IDs.

    Missing parents remain absent for the lineage checker to retain and mark.
    Multiple patch-clamp parents are retained, never first-row selected.
    Integer links are followed without bool/float coercion. No data-driven
    contributor selection or extra historical QC filtering occurs.
    """
    pulse_ids = list(pulse_ids)
    if (not pulse_ids or len(pulse_ids) > MAX_IDS
        or any(not positive(value) for value in pulse_ids)
        or len(set(pulse_ids)) != len(pulse_ids)):
        raise ValueError("bounded positive unique original pulse IDs required")
    if (set(context) != {"pair_id", "experiment_id", "pre_cell_id", "post_cell_id"}
        or not all(positive(value) for value in context.values())):
        raise ValueError("fixed selected pair context required")
    payload_bytes = 0
    tables = {table: [] for table in COLUMNS}

    def read(db, table, identities, key="id"):
        nonlocal payload_bytes
        identities = sorted(set(identities))
        if len(identities) > MAX_IDS or any(not positive(identity) for identity in identities):
            raise ValueError("parent ID bound or type differs")
        if key != ("recording_id" if table == "patch_clamp_recording" else "id"):
            raise ValueError("parent selection key differs")
        columns = COLUMNS[table]
        # Bound variable-length data inside SQLite before Python allocation.
        expressions = [
            (f'CASE WHEN typeof("{column}") IN (\'text\',\'null\') '
             f'THEN substr("{column}",1,{MAX_STRING_BYTES + 1}) ELSE 0 END')
            if column == "clamp_mode" else
            (f'CASE WHEN typeof("{column}") IN (\'integer\',\'real\',\'null\') '
             f'THEN "{column}" ELSE \'invalid-numeric-scalar\' END')
            for column in columns
        ]
        query = ("SELECT " + ",".join(expressions)
                 + f' FROM "{table}" WHERE "{key}"=? ORDER BY id LIMIT {MAX_ROWS_PER_PARENT + 1}')
        seen = set()
        for identity in identities:
            for count, values in enumerate(db.execute(query, (identity,)), start=1):
                if count > MAX_ROWS_PER_PARENT or len(tables[table]) >= MAX_ROWS_PER_TABLE:
                    raise ValueError("metadata row bound exceeded")
                row = dict(zip(columns, values, strict=True))
                if not positive(row["id"]) or row["id"] in seen:
                    raise ValueError("invalid or duplicate projected primary ID")
                seen.add(row["id"])
                for column, value in row.items():
                    if column == "clamp_mode":
                        if value is not None and type(value) is not str:
                            raise ValueError("clamp mode scalar type differs")
                        size = len(value.encode("utf-8")) if value is not None else 1
                        if size > MAX_STRING_BYTES:
                            raise ValueError("metadata string bound exceeded")
                    else:
                        if value is not None and type(value) not in (int, float):
                            raise ValueError("metadata numeric scalar type differs")
                        size = 8 if value is not None else 1
                    payload_bytes += size
                    if payload_bytes > MAX_PAYLOAD_BYTES:
                        raise ValueError("metadata payload bound exceeded")
                tables[table].append(row)

    def links(table, column):
        return {row[column] for row in tables[table] if positive(row[column])}

    with sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro&immutable=1", uri=True) as db:
        db.execute("PRAGMA query_only=ON")
        # typeof rejects nonnumeric payloads before allocation; substr bounds text.
        def authorize(action, table, column, database, trigger):
            if action == sqlite3.SQLITE_FUNCTION and column in ("substr", "typeof"):
                return sqlite3.SQLITE_OK
            return metadata_authorizer(action, table, column, database, trigger)

        db.set_authorizer(authorize)
        read(db, "cell", {context["pre_cell_id"], context["post_cell_id"]})
        read(db, "pulse_response", pulse_ids)
        read(db, "stim_pulse", links("pulse_response", "stim_pulse_id"))
        read(db, "recording", links("pulse_response", "recording_id") | links("stim_pulse", "recording_id"))
        post_ids = links("pulse_response", "recording_id")
        read(db, "patch_clamp_recording", post_ids, key="recording_id")
        read(db, "electrode", links("recording", "electrode_id"))
        read(db, "sync_rec", links("recording", "sync_rec_id"))
    return {"tables": tables, "projected_scalar_payload_bytes": payload_bytes,
            "original_pulse_response_ids": pulse_ids, "waveform_values_read": False,
            "biological_eligibility_established": False}
