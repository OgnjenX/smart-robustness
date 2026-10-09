"""Stream the sealed medium object with bounded memory; inspect schema only."""

from __future__ import annotations

import hashlib
import shutil
import sqlite3
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import yaml
from acquire_synphys_small_schema import inspect_schema
from collect_synphys_source_metadata import digest, save, tls_context

REGISTRATION = Path("docs/validation-results/post2008-synphys-medium-acquisition-registration-1130.yaml")


def stream_object(stream, target, expected_bytes, chunk_bytes=1048576):
    """Return hash and size even for a short or overlong response."""
    count, hasher = 0, hashlib.sha256()
    while count <= expected_bytes:
        chunk = stream.read(min(chunk_bytes, expected_bytes + 1 - count))
        if not chunk:
            break
        target.write(chunk)
        hasher.update(chunk)
        count += len(chunk)
    return count, hasher.hexdigest()


def main():
    reg = yaml.safe_load(REGISTRATION.read_text())
    if (reg["database_download_authorized"], reg["schema_only_reads_authorized"], reg["row_reads_authorized"],
        reg["parameter_fitting_authorized"], reg["cell_or_network_execution_authorized"],
        reg["full_release_fallback_authorized"]) != (True, True, False, False, False, False):
        raise ValueError("acquisition scope changed")
    if (reg["url"], reg["expected_bytes"], reg["expected_etag"], reg["maximum_seconds"],
        reg["maximum_requests"], reg["chunk_bytes"], reg["minimum_free_bytes"]) != (
        "https://allen-synphys.s3-us-west-2.amazonaws.com/synphys_r2.1_medium.sqlite",
        11125997568, '"d954cbad0d7c7b0002bf3a2879e40e90-1327"', 3600, 1, 1048576, 22251995136):
        raise ValueError("object identity or bounds changed")
    if digest(reg["header_manifest"]) != reg["header_manifest_sha256"]:
        raise ValueError("header lineage changed")
    tls_context(reg)
    root = Path(reg["output_directory"])
    if shutil.disk_usage(root.parent).free < reg["minimum_free_bytes"]:
        raise ValueError("storage prerequisite failed")
    root.mkdir()
    path = root / "synphys_r2.1_medium.sqlite.partial"
    dependency = Path(__file__).with_name("acquire_synphys_small_schema.py")
    manifest = {"registration_sha256": digest(REGISTRATION), "collector_sha256": digest(__file__),
                "schema_inspector_sha256": digest(dependency), "row_values_read": False,
                "started_at_utc": datetime.now(UTC).isoformat(), "status": "failure-retained",
                "raw_path": str(path)}
    save(root / "started.yaml", manifest)
    args = ["/usr/bin/curl", "--silent", "--show-error", "--fail", "--proto", "=https",
            "--max-time", "3600", "--max-filesize", str(reg["expected_bytes"]),
            "--cacert", reg["certificate_authority_bundle"], "--header",
            "If-Match: " + reg["expected_etag"], reg["url"]]
    try:
        with (
            path.open("xb") as out,
            (root / "transport-stderr.txt").open("xb") as err,
            subprocess.Popen(args, stdout=subprocess.PIPE, stderr=err) as proc,
        ):
            try:
                count, checksum = stream_object(proc.stdout, out, reg["expected_bytes"])
                if count > reg["expected_bytes"]:
                    proc.kill()
                code = proc.wait(timeout=3605)
            except BaseException:
                proc.kill()
                proc.wait()
                raise
        manifest.update(bytes=count, sha256=checksum, client_exit_code=code)
        with path.open("rb") as handle:
            signature = handle.read(16)
        if code or count != reg["expected_bytes"] or signature != b"SQLite format 3\x00":
            raise ValueError("transport, exact length or SQLite signature failed")
        final = root / "synphys_r2.1_medium.sqlite"
        path.rename(final)
        manifest["raw_path"] = str(final)
        manifest["schema"] = inspect_schema(final)
        manifest["status"] = "medium-acquired-schema-only"
    except (OSError, ValueError, sqlite3.Error, subprocess.TimeoutExpired) as exc:
        manifest["error"] = str(exc)
        # Preserve a bounded-memory identity of partial bytes after local I/O failure.
        partial = Path(manifest["raw_path"])
        if partial.exists():
            with partial.open("rb") as handle:
                hasher = hashlib.sha256()
                for block in iter(lambda: handle.read(1048576), b""):
                    hasher.update(block)
            manifest.update(bytes=partial.stat().st_size, sha256=hasher.hexdigest())
    manifest["finished_at_utc"] = datetime.now(UTC).isoformat()
    save(root / "manifest.yaml", manifest)
    print(manifest["status"])


if __name__ == "__main__":
    main()
