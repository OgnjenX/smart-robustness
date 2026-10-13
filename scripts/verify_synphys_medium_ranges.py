"""Independently verify sealed range assembly and schema; never read data rows."""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import yaml
from verify_synphys_medium_acquisition import reconstruct_schema

REGISTRATION = Path("docs/validation-results/post2008-synphys-medium-range-continuation-registration-1145.yaml")
SEAL = Path("docs/validation-results/post2008-synphys-medium-range-continuation-seal-1147.yaml")
MANIFEST = Path("results/synphys-medium-range-continuation-1146/manifest.yaml")
OUTPUT = Path("results/synphys-medium-range-verification-1150")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def snapshot(path):
    value = path.stat()
    return value.st_size, value.st_mtime_ns, value.st_ino


def check_range_header(path, first, last, total, etag):
    """Independent terminal response parsing, with no downloader imports."""
    with path.open("rb") as handle:
        content = handle.read(65537)
    if len(content) > 65536:
        raise ValueError("header length exceeds verifier bound")
    terminal = None
    for section in content.decode("ascii").replace("\r\n", "\n").split("\n\n"):
        if section.startswith("HTTP/"):
            terminal = section.splitlines()
    if not terminal or terminal[0].split()[1:2] != ["206"]:
        raise ValueError("terminal range response is not 206")
    fields = {}
    for line in terminal[1:]:
        if not line or line[0].isspace() or ":" not in line:
            raise ValueError("invalid header grammar")
        key, value = line.split(":", 1)
        fields.setdefault(key.strip().lower(), []).append(value.strip())
    for key, required in {"etag": etag, "content-length": str(last - first + 1),
                          "content-range": f"bytes {first}-{last}/{total}"}.items():
        if fields.get(key) != [required]:
            raise ValueError("response identity or boundary mismatch")
    if fields.get("content-encoding", ["identity"]) != ["identity"]:
        raise ValueError("response encoding changed")
    return {key: values[0] for key, values in fields.items() if len(values) == 1}


def verify_assembly(path, prefix, prefix_bytes, prefix_hash, manifest, total, extent, maximum, etag):
    """Compare each original source byte to its position in the assembled object."""
    if (manifest.get("status") != "medium-range-acquired-schema-only"
        or manifest.get("bytes") != total or manifest.get("row_values_read") is not False
        or manifest.get("original_prefix_modified") is not False):
        raise ValueError("terminal successful metadata-only acquisition required")
    accepted = manifest["accepted_ranges"]
    required_count = (total - prefix_bytes + extent - 1) // extent
    if not 0 < required_count <= maximum or len(accepted) != required_count:
        raise ValueError("registered complete range count differs")
    if path.stat().st_size != total:
        raise ValueError("assembled object length differs")
    evidence_paths = [path, prefix]
    for index in range(len(accepted)):
        evidence_paths.extend(path.parent / f"range-{index:02d}{suffix}"
                              for suffix in (".partial", "-headers.txt", "-stderr.txt"))
    before = {item: snapshot(item) for item in evidence_paths}
    sources = [(prefix, prefix_bytes, prefix_hash)]
    cursor = prefix_bytes
    for index, record in enumerate(accepted):
        last = min(cursor + extent, total) - 1
        if (any(type(record[key]) is not int for key in ("index", "first", "last", "bytes", "client_exit_code"))
            or record["index"] != index or record["first"] != cursor or record["last"] != last
            or record["bytes"] != last - cursor + 1 or record["client_exit_code"] != 0):
            raise ValueError("range ledger has gap, overlap, changed count or unsuccessful exit")
        chunk = path.parent / f"range-{index:02d}.partial"
        header = path.parent / f"range-{index:02d}-headers.txt"
        stderr = path.parent / f"range-{index:02d}-stderr.txt"
        if (Path(record["raw_path"]) != chunk or Path(record["header_path"]) != header
            or Path(record["stderr_path"]) != stderr):
            raise ValueError("exclusive range evidence paths differ")
        if check_range_header(header, cursor, last, total, etag) != record["headers"]:
            raise ValueError("archived header projection differs")
        if not stderr.is_file():
            raise ValueError("range stderr evidence absent")
        sources.append((chunk, last - cursor + 1, record["sha256"]))
        cursor = last + 1
    if cursor != total:
        raise ValueError("final range boundary differs")
    whole = hashlib.sha256()
    component_hashes = []
    with path.open("rb") as assembled:
        if assembled.read(16) != b"SQLite format 3\x00":
            raise ValueError("SQLite signature differs")
        assembled.seek(0)
        for source_path, size, expected_hash in sources:
            if source_path.stat().st_size != size:
                raise ValueError("component length differs")
            component = hashlib.sha256()
            with source_path.open("rb") as original:
                remaining = size
                while remaining:
                    block = original.read(min(remaining, 1048576))
                    if not block or assembled.read(len(block)) != block:
                        raise ValueError("assembled bytes differ from retained component")
                    whole.update(block)
                    component.update(block)
                    remaining -= len(block)
                if original.read(1):
                    raise ValueError("component grew during verification")
            checksum = component.hexdigest()
            if checksum != expected_hash:
                raise ValueError("independent component SHA256 differs")
            component_hashes.append(checksum)
        if assembled.read(1):
            raise ValueError("assembled object grew during verification")
    if whole.hexdigest() != manifest["sha256"]:
        raise ValueError("independent complete object SHA256 differs")
    if any(snapshot(item) != state for item, state in before.items()):
        raise ValueError("source changed during independent verification")
    return {"bytes": total, "sha256": whole.hexdigest(), "component_sha256": component_hashes,
            "accepted_ranges_verified": len(accepted), "original_prefix_and_every_range_byte_equal": True}


def main():
    if not MANIFEST.is_file():
        raise ValueError("terminal manifest absent; do not rerun acquisition")
    OUTPUT.mkdir()
    report = {"status": "range-verification-failure-retained", "verifier_sha256": digest(__file__),
              "schema_verifier_sha256": digest(Path(__file__).with_name("verify_synphys_medium_acquisition.py")),
              "acquisition_manifest_sha256": digest(MANIFEST), "collector_imported": False,
              "data_rows_read": False, "started_at_utc": datetime.now(UTC).isoformat()}
    try:
        reg = yaml.safe_load(REGISTRATION.read_text())
        seal = yaml.safe_load(SEAL.read_text())
        manifest = yaml.safe_load(MANIFEST.read_text())
        for key, value in {"expected_bytes": 11125997568, "prefix_bytes": 3573501533,
                           "url": "https://allen-synphys.s3-us-west-2.amazonaws.com/synphys_r2.1_medium.sqlite",
                           "maximum_total_seconds": 18000, "maximum_seconds_per_request": 1200,
                           "range_bytes": 536870912, "maximum_requests": 15,
                           "expected_etag": '"d954cbad0d7c7b0002bf3a2879e40e90-1327"',
                           "output_directory": "results/synphys-medium-range-continuation-1146",
                           "database_download_authorized": True, "schema_only_reads_authorized": True,
                           "row_reads_authorized": False, "parameter_fitting_authorized": False,
                           "cell_or_network_execution_authorized": False, "full_release_fallback_authorized": False}.items():
            if type(reg.get(key)) is not type(value) or reg[key] != value:
                raise ValueError("fixed verifier identity or scope differs")
        if (type(manifest.get("elapsed_seconds")) not in (int, float)
            or not 0 <= manifest["elapsed_seconds"] <= reg["maximum_total_seconds"]
            or manifest.get("copied_prefix_sha256") != reg["prefix_sha256"]):
            raise ValueError("registered time bound or copied prefix identity differs")
        if seal.get("execution_authorized") is not True or digest(SEAL) != manifest["seal_sha256"]:
            raise ValueError("original execution seal differs")
        for key, source in {"registration_sha256": REGISTRATION,
                            "collector_sha256": Path(__file__).with_name("acquire_synphys_medium_ranges.py"),
                            "schema_inspector_sha256": Path(__file__).with_name("acquire_synphys_small_schema.py")}.items():
            if digest(source) != seal[key] or seal[key] != manifest[key]:
                raise ValueError("sealed source dependency differs")
        for key in ("parent_manifest", "parent_assessment", "certificate_authority_bundle"):
            field = "certificate_authority_sha256" if key == "certificate_authority_bundle" else key + "_sha256"
            if digest(reg[key]) != reg[field]:
                raise ValueError("registered parent or certificate source differs")
        path = Path(reg["output_directory"]) / "synphys_r2.1_medium.sqlite"
        if Path(manifest["raw_path"]) != path:
            raise ValueError("assembled evidence path differs")
        source_paths = [REGISTRATION, SEAL, MANIFEST, Path(reg["parent_manifest"]),
                        Path(reg["parent_assessment"]), Path(reg["certificate_authority_bundle"])]
        before = {item: snapshot(item) for item in source_paths}
        report["identity"] = verify_assembly(path, Path(reg["prefix_path"]), reg["prefix_bytes"], reg["prefix_sha256"],
                                             manifest, reg["expected_bytes"], reg["range_bytes"],
                                             reg["maximum_requests"], reg["expected_etag"])
        state = snapshot(path)
        schema = reconstruct_schema(path)
        if schema != manifest["schema"] or snapshot(path) != state:
            raise ValueError("independent schema or object stability differs")
        if any(snapshot(item) != initial for item, initial in before.items()):
            raise ValueError("registration or source evidence changed")
        report.update(schema=schema, status="medium-range-object-and-schema-independently-verified")
    except (OSError, ValueError, KeyError, sqlite3.Error) as exc:
        report["error"] = str(exc)
    report["finished_at_utc"] = datetime.now(UTC).isoformat()
    with (OUTPUT / "manifest.yaml").open("x") as handle:
        yaml.safe_dump(report, handle, sort_keys=False)
    print(report["status"])
    raise SystemExit(0 if report["status"] == "medium-range-object-and-schema-independently-verified" else 1)


if __name__ == "__main__":
    main()
