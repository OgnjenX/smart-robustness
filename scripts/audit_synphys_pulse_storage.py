"""Decode bounded numeric pulse IDs, never pickle or physiological waveforms."""

from __future__ import annotations

import hashlib
import io
import sqlite3
from pathlib import Path

import numpy as np
import yaml
from collect_synphys_source_metadata import digest, save

REGISTRATION = Path("docs/validation-results/post2008-synphys-pulse-storage-registration-1121.yaml")
COLUMNS = {"resting_state_fit": ["id", "synapse_id", "ic_pulse_ids"],
           "pulse_response": ["id"], "recording": ["id"],
           "patch_clamp_recording": ["id"], "stim_pulse": ["id"]}


def decode_ids(raw):
    if not isinstance(raw, bytes) or not raw or len(raw) > 1048576:
        raise ValueError("missing or oversized ID blob")
    buf = io.BytesIO(raw)
    version = np.lib.format.read_magic(buf)
    if version == (1, 0):
        shape, order, dtype = np.lib.format.read_array_header_1_0(buf, max_header_size=10000)
    elif version == (2, 0):
        shape, order, dtype = np.lib.format.read_array_header_2_0(buf, max_header_size=10000)
    else:
        raise ValueError("unregistered NPY version")
    if dtype.kind not in "iu" or dtype.hasobject or not (len(shape) == 1 or (len(shape) == 2 and shape[0] == 1)):
        raise ValueError("unregistered ID dtype or shape")
    count = shape[-1]
    if count <= 0 or count > 100000 or count * dtype.itemsize != len(raw) - buf.tell():
        raise ValueError("ID extent or byte length invalid")
    values = np.load(io.BytesIO(raw), allow_pickle=False, max_header_size=10000).reshape(-1)
    ids = [int(v) for v in values]
    if any(v <= 0 for v in ids) or len(set(ids)) != len(ids):
        raise ValueError("nonpositive or duplicate pulse IDs")
    return {"ids": ids, "source_shape": list(shape), "dtype": str(dtype), "fortran_order": order}


def authorizer(action, table, column, database, trigger):
    if action == sqlite3.SQLITE_READ:
        return sqlite3.SQLITE_OK if column in COLUMNS.get(table, []) else sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_FUNCTION:
        return sqlite3.SQLITE_OK if column == "count" else sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK if action == sqlite3.SQLITE_SELECT else sqlite3.SQLITE_DENY


def main():
    reg = yaml.safe_load(REGISTRATION.read_text())
    if reg["allowed_columns"] != COLUMNS or reg["metadata_reads_authorized"] is not True:
        raise ValueError("metadata scope changed")
    for key in ("waveform_reads_authorized", "database_download_authorized", "parameter_fitting_authorized", "cell_or_network_execution_authorized"):
        if reg[key] is not False:
            raise ValueError("execution scope changed")
    if reg["decoder"] != {"maximum_blob_bytes": 1048576, "maximum_header_bytes": 10000, "maximum_ids": 100000,
                          "allowed_dtype_kinds": ["i", "u"], "allowed_shapes": "one-dimensional-or-singleton-leading-two-dimensional",
                          "numpy_versions": [1, 2], "pickle_allowed": False,
                          "object_arrays_allowed": False, "positive_unique_IDs_required": True}:
        raise ValueError("decoder limits changed")
    for key in ("database", "resting_inventory"):
        if digest(reg[key]) != reg[key + "_sha256"]:
            raise ValueError("source lineage changed")
    source = yaml.safe_load(Path(reg["resting_inventory"]).read_text())
    root = Path(reg["output_directory"])
    root.mkdir()
    records, counts = [], {}
    with sqlite3.connect(Path(reg["database"]).resolve().as_uri()+"?mode=ro&immutable=1", uri=True) as db:
        db.execute("PRAGMA query_only=ON")
        db.set_authorizer(authorizer)
        for name in COLUMNS:
            if name != "resting_state_fit":
                counts[name] = db.execute(f'SELECT count(id) FROM "{name}"').fetchone()[0]
        for parent in source["records"]:
            sid = parent["synapse_id"]
            rows = db.execute("SELECT id,synapse_id,ic_pulse_ids FROM resting_state_fit WHERE synapse_id=? ORDER BY id", (sid,)).fetchall()
            for rid, actual_sid, raw in rows:
                record = {"resting_id": rid, "synapse_id": actual_sid}
                if isinstance(raw, bytes):
                    path = root / f"resting-{rid}-pulse-ids.npy"
                    with path.open("xb") as handle:
                        handle.write(raw)
                    record.update(raw_path=str(path), raw_bytes=len(raw), raw_sha256=hashlib.sha256(raw).hexdigest())
                try:
                    record.update(decode_ids(raw), status="decoded-numeric-IDs")
                except (ValueError, OSError, EOFError) as exc:
                    record.update(status="decode-failure-retained", error=str(exc))
                records.append(record)
    save(root / "manifest.yaml", {"registration_sha256": digest(REGISTRATION), "collector_sha256": digest(__file__),
                                  "records": records, "metadata_table_row_counts": counts,
                                  "waveform_values_read": False, "status": "storage-audit-complete"})
    print("storage-audit-complete")


if __name__ == "__main__":
    main()
