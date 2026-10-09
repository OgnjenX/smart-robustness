"""Independent streamed object/schema verification; never read data rows.

No acquisition helpers are imported. Run only after terminal acquisition;
missing output is not permission to rerun the download.
"""

from __future__ import annotations

import argparse
import hashlib
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import yaml

REGISTRATION = Path("docs/validation-results/post2008-synphys-medium-acquisition-registration-1130.yaml")
MANIFEST = Path("results/synphys-medium-schema-1131/manifest.yaml")
OUTPUT = Path("results/synphys-medium-verification-1132")


def object_identity(path):
    checksum, count = hashlib.sha256(), 0
    with path.open("rb") as handle:
        signature = handle.read(16)
        handle.seek(0)
        for block in iter(lambda: handle.read(1048576), b""):
            checksum.update(block)
            count += len(block)
    return {"bytes": count, "sha256": checksum.hexdigest(),
            "sqlite_signature": signature == b"SQLite format 3\x00"}


def checksum(path):
    return object_identity(Path(path))["sha256"]


def schema_guard(action, first, second, database, trigger):
    if action == sqlite3.SQLITE_READ and first in {"sqlite_master", "sqlite_schema"}:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_SELECT:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_PRAGMA and first == "table_info":
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def reconstruct_schema(path):
    tables = []
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True) as db:
        db.execute("PRAGMA query_only=ON")
        db.set_authorizer(schema_guard)
        for name, definition in db.execute(
            "SELECT name,sql FROM sqlite_master WHERE type='table' ORDER BY name"
        ):
            quoted = name.replace('"', '""')
            fields = db.execute(f'PRAGMA table_info("{quoted}")').fetchall()
            tables.append({"name": name, "definition": definition,
                           "columns": [list(field) for field in fields]})
    return {"tables": tables, "data_rows_read": False}


def verify_object(path, manifest, expected_bytes):
    if manifest.get("status") != "medium-acquired-schema-only" or manifest.get("client_exit_code") != 0:
        raise ValueError("terminal successful acquisition required")
    before = path.stat()
    if before.st_size != expected_bytes or manifest["bytes"] != expected_bytes:
        raise ValueError("exact object length failed")
    identity = object_identity(path)
    if (identity["bytes"] != expected_bytes or identity["sha256"] != manifest["sha256"]
        or not identity["sqlite_signature"]):
        raise ValueError("independent object identity failed")
    schema = reconstruct_schema(path)
    if schema != manifest["schema"]:
        raise ValueError("independent complete schema comparison failed")
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError("object changed during verification")
    return {"identity": identity, "schema": schema}


def verify_sources(registration, manifest):
    reg = yaml.safe_load(registration.read_text())
    permissions = ("database_download_authorized", "schema_only_reads_authorized",
                   "row_reads_authorized", "parameter_fitting_authorized",
                   "cell_or_network_execution_authorized", "full_release_fallback_authorized")
    if (reg["expected_bytes"] != 11125997568
        or reg["output_directory"] != "results/synphys-medium-schema-1131"
        or any(reg[key] is not expected for key, expected in
               zip(permissions, (True, True, False, False, False, False), strict=True))):
        raise ValueError("registered acquisition scope changed")
    dependencies = {
        "registration_sha256": registration,
        "collector_sha256": Path(__file__).with_name("acquire_synphys_medium_schema.py"),
        "schema_inspector_sha256": Path(__file__).with_name("acquire_synphys_small_schema.py"),
    }
    for field, path in dependencies.items():
        if checksum(path) != manifest[field]:
            raise ValueError("acquisition dependency changed: " + field)
    for field in ("header_manifest", "certificate_authority_bundle"):
        hash_key = "certificate_authority_sha256" if field == "certificate_authority_bundle" else field + "_sha256"
        if checksum(reg[field]) != reg[hash_key]:
            raise ValueError("acquisition source lineage changed: " + field)
    expected_path = Path(reg["output_directory"]) / "synphys_r2.1_medium.sqlite"
    if Path(manifest["raw_path"]) != expected_path or manifest["row_values_read"] is not False:
        raise ValueError("acquisition path or data-read boundary differs")
    return reg, expected_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    # A missing manifest means no terminal evidence; never create an output
    # claiming failure or completion simply because observation is premature.
    if not MANIFEST.is_file():
        raise ValueError("terminal acquisition manifest absent; do not restart download")
    OUTPUT.mkdir()
    report = {"verifier_sha256": checksum(__file__), "acquisition_manifest": str(MANIFEST),
              "acquisition_manifest_sha256": checksum(MANIFEST),
              "started_at_utc": datetime.now(UTC).isoformat(),
              "collector_imported": False, "data_rows_read": False,
              "status": "verification-failure-retained"}
    try:
        manifest = yaml.safe_load(MANIFEST.read_text())
        reg, path = verify_sources(REGISTRATION, manifest)
        report.update(verify_object(path, manifest, reg["expected_bytes"]))
        report.update(status="medium-object-and-schema-independently-verified",
                      checks={name: "passed" for name in
                              ("byte_count", "sha256", "sqlite_signature", "schema", "source_integrity")})
    except (ValueError, OSError, KeyError, sqlite3.Error) as exc:
        report["error"] = str(exc)
    report["finished_at_utc"] = datetime.now(UTC).isoformat()
    with (OUTPUT / "manifest.yaml").open("x") as handle:
        yaml.safe_dump(report, handle, sort_keys=False)
    print(report["status"])
    raise SystemExit(0 if report["status"] == "medium-object-and-schema-independently-verified" else 1)


if __name__ == "__main__":
    main()
