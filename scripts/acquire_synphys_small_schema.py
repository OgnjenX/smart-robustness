"""Acquire one sealed object and inspect schema, never response rows."""

from __future__ import annotations

import hashlib
import shutil
import sqlite3
import subprocess
from pathlib import Path

import yaml
from collect_synphys_source_metadata import digest, save, tls_context

REGISTRATION = Path(
    "docs/validation-results/post2008-synphys-small-acquisition-registration-1101.yaml"
)


def schema_authorizer(action, first, second, database, trigger):
    if action == sqlite3.SQLITE_READ:
        return sqlite3.SQLITE_OK if first in {"sqlite_master", "sqlite_schema"} else sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_SELECT:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_PRAGMA and first == "table_info":
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def inspect_schema(path):
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True) as db:
        db.execute("PRAGMA query_only=ON")
        db.set_authorizer(schema_authorizer)
        tables = []
        for name, sql in db.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall():
            quoted = name.replace('"', '""')
            columns = db.execute(f'PRAGMA table_info("{quoted}")').fetchall()
            tables.append({"name": name, "definition": sql, "columns": [list(c) for c in columns]})
        return {"tables": tables, "data_rows_read": False}


def main():
    reg = yaml.safe_load(REGISTRATION.read_text())
    if (reg["database_download_authorized"], reg["schema_only_reads_authorized"],
        reg["numerical_or_response_row_reads_authorized"], reg["parameter_fitting_authorized"],
        reg["network_execution_authorized"]) != (True, True, False, False, False):
        raise ValueError("acquisition scope changed")
    if reg["url"] != "https://allen-synphys.s3-us-west-2.amazonaws.com/synphys_r2.1_small.sqlite":
        raise ValueError("unregistered object")
    if (reg["expected_bytes"], reg["maximum_seconds"], reg["maximum_requests"]) != (176771072, 180, 1):
        raise ValueError("bounds changed")
    tls_context(reg)
    root = Path(reg["output_directory"])
    if shutil.disk_usage(root.parent).free < reg["minimum_free_bytes"]:
        raise ValueError("storage prerequisite failed")
    root.mkdir()
    path = root / "synphys_r2.1_small.sqlite.partial"
    manifest = {"registration_sha256": digest(REGISTRATION), "collector_sha256": digest(__file__),
                "numerical_or_response_row_reads": False, "status": "failure-retained"}
    args = ["/usr/bin/curl", "--silent", "--show-error", "--fail", "--proto", "=https",
            "--max-time", "180", "--max-filesize", str(reg["expected_bytes"]),
            "--cacert", reg["certificate_authority_bundle"], "--header",
            "If-Match: " + reg["expected_etag"], reg["url"]]
    try:
        count = 0
        hasher = hashlib.sha256()
        with path.open("xb") as out, subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        ) as proc:
            while count <= reg["expected_bytes"]:
                chunk = proc.stdout.read(min(1048576, reg["expected_bytes"] + 1 - count))
                if not chunk:
                    break
                out.write(chunk)
                hasher.update(chunk)
                count += len(chunk)
            if count > reg["expected_bytes"]:
                proc.kill()
            code = proc.wait(timeout=185)
        manifest.update(bytes=count, sha256=hasher.hexdigest(), client_exit_code=code, raw_path=str(path))
        with path.open("rb") as handle:
            signature = handle.read(16)
        if code or count != reg["expected_bytes"] or signature != b"SQLite format 3\x00":
            raise ValueError("transport, exact length or SQLite signature failed")
        final = root / "synphys_r2.1_small.sqlite"
        path.rename(final)
        manifest["raw_path"] = str(final)
        manifest["schema"] = inspect_schema(final)
        manifest["status"] = "acquired-schema-only"
    except (OSError, ValueError, sqlite3.Error, subprocess.TimeoutExpired) as exc:
        manifest["error"] = str(exc)
    save(root / "manifest.yaml", manifest)
    print(manifest["status"])


if __name__ == "__main__":
    main()
