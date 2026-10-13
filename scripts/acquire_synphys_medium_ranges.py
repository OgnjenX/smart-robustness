"""Sealed sequential byte-range continuation; no retries or physiological reads."""

from __future__ import annotations

import hashlib
import shutil
import sqlite3
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml
from acquire_synphys_small_schema import inspect_schema

REGISTRATION = Path("docs/validation-results/post2008-synphys-medium-range-continuation-registration-1145.yaml")
SEAL = Path("docs/validation-results/post2008-synphys-medium-range-continuation-seal-1147.yaml")


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def write_receipt(path, data):
    # Fixed output remains inspectable at each completed range boundary.
    with path.open("w") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def ranges(start, total, extent):
    if any(type(value) is not int or value <= 0 for value in (start, total, extent)) or start >= total:
        raise ValueError("positive bounded range geometry required")
    for first in range(start, total, extent):
        yield first, min(first + extent, total) - 1


def validate_headers(raw, first, last, total, etag):
    """Use the last HTTP response block; CONNECT or interim headers are not it."""
    if len(raw) > 65536:
        raise ValueError("response header byte bound exceeded")
    blocks = raw.decode("ascii", errors="strict").replace("\r\n", "\n").split("\n\n")
    responses = [block for block in blocks if block.startswith("HTTP/")]
    if not responses:
        raise ValueError("HTTP response headers absent")
    lines = responses[-1].splitlines()
    fields = lines[0].split()
    if len(fields) < 2 or fields[1] != "206":
        raise ValueError("exact 206 range response required")
    headers = {}
    for line in lines[1:]:
        if ":" not in line or line[0].isspace():
            raise ValueError("malformed response header")
        name, value = line.split(":", 1)
        name, value = name.lower().strip(), value.strip()
        headers.setdefault(name, []).append(value)
    required = {"content-range": f"bytes {first}-{last}/{total}",
                "content-length": str(last - first + 1), "etag": etag}
    for key, expected in required.items():
        if headers.get(key) != [expected]:
            raise ValueError("range identity header differs: " + key)
    if headers.get("content-encoding", ["identity"]) != ["identity"]:
        raise ValueError("identity encoding required")
    return {key: values[0] for key, values in headers.items() if len(values) == 1}


def stream_body(source, destination, expected, chunk_bytes=1048576):
    if type(expected) is not int or expected <= 0 or type(chunk_bytes) is not int or chunk_bytes <= 0:
        raise ValueError("positive body bounds required")
    count, digest = 0, hashlib.sha256()
    while count <= expected:
        block = source.read(min(chunk_bytes, expected + 1 - count))
        if not block:
            break
        destination.write(block)
        count += len(block)
        digest.update(block)
    return count, digest.hexdigest()


def copy_prefix(source, target, expected_bytes, expected_hash):
    before = source.stat()
    if before.st_size != expected_bytes:
        raise ValueError("original prefix length changed")
    with source.open("rb") as src, target.open("xb") as dst:
        if src.read(16) != b"SQLite format 3\x00":
            raise ValueError("original prefix SQLite signature absent")
        src.seek(0)
        count, digest = stream_body(src, dst, expected_bytes)
    after = source.stat()
    if ((before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino)
        or count != expected_bytes or digest != expected_hash):
        raise ValueError("original prefix identity changed")
    return digest


def validate_registration(reg):
    fixed = {
        "url": "https://allen-synphys.s3-us-west-2.amazonaws.com/synphys_r2.1_medium.sqlite",
        "expected_bytes": 11125997568,
        "expected_etag": '"d954cbad0d7c7b0002bf3a2879e40e90-1327"',
        "prefix_path": "results/synphys-medium-schema-1131/synphys_r2.1_medium.sqlite.partial",
        "prefix_bytes": 3573501533,
        "prefix_sha256": "f2f24c87df8539e2d05e0011ca8ab276cf0f29d6dbf74e7ba72d8b62c77cd001",
        "parent_manifest": "results/synphys-medium-schema-1131/manifest.yaml",
        "parent_manifest_sha256": "63dde91bb1a8eb5d54f18b7226cc2fcc16502318590d9a91fc4a3939f4e42906",
        "parent_assessment": "docs/validation-results/post2008-synphys-medium-acquisition-failure-assessment-1144.yaml",
        "parent_assessment_sha256": "c52fd3831ec479efe6564d3033e2da75fdef90f3ce2786630ede541b43c86494",
        "certificate_authority_bundle": "/etc/ssl/cert.pem",
        "certificate_authority_sha256": "9dae8d76e55cb08991f2b672d58999ea15560d910759c16b544f843bdffbb994",
        "output_directory": "results/synphys-medium-range-continuation-1146",
        "range_bytes": 536870912, "stream_chunk_bytes": 1048576, "maximum_requests": 15,
        "maximum_seconds_per_request": 1200, "maximum_total_seconds": 18000,
        "minimum_free_bytes": 22251995136,
        "database_download_authorized": True, "schema_only_reads_authorized": True,
        "row_reads_authorized": False, "parameter_fitting_authorized": False,
        "cell_or_network_execution_authorized": False, "full_release_fallback_authorized": False,
    }
    if any(type(reg.get(key)) is not type(value) or reg[key] != value for key, value in fixed.items()):
        raise ValueError("fixed acquisition identity, bounds or scope changed")
    if len(list(ranges(reg["prefix_bytes"], reg["expected_bytes"], reg["range_bytes"]))) > reg["maximum_requests"]:
        raise ValueError("request count exceeds registration")


def main():
    reg = yaml.safe_load(REGISTRATION.read_text())
    validate_registration(reg)
    seal = yaml.safe_load(SEAL.read_text())
    inspector = Path(__file__).with_name("acquire_synphys_small_schema.py")
    dependencies = {"registration_sha256": REGISTRATION, "collector_sha256": Path(__file__),
                    "schema_inspector_sha256": inspector}
    if seal.get("execution_authorized") is not True:
        raise ValueError("separate execution seal absent")
    for key, path in dependencies.items():
        if checksum(path) != seal[key]:
            raise ValueError("sealed dependency changed: " + key)
    for key in ("parent_manifest", "parent_assessment", "certificate_authority_bundle"):
        field = "certificate_authority_sha256" if key == "certificate_authority_bundle" else key + "_sha256"
        if checksum(reg[key]) != reg[field]:
            raise ValueError("registered parent or CA changed")
    parent = yaml.safe_load(Path(reg["parent_manifest"]).read_text())
    if (parent.get("status") != "failure-retained" or parent.get("client_exit_code") != 28
        or parent.get("bytes") != reg["prefix_bytes"] or parent.get("sha256") != reg["prefix_sha256"]
        or parent.get("raw_path") != reg["prefix_path"] or parent.get("row_values_read") is not False):
        raise ValueError("terminal original acquisition evidence differs")
    root = Path(reg["output_directory"])
    if shutil.disk_usage(root.parent).free < reg["minimum_free_bytes"]:
        raise ValueError("registered storage prerequisite failed")
    root.mkdir()  # An existing directory is never replayed or overwritten.
    started = time.monotonic()
    report = {key: checksum(path) for key, path in dependencies.items()}
    report.update(seal_sha256=checksum(SEAL), status="range-acquisition-failure-retained",
                  started_at_utc=datetime.now(UTC).isoformat(), accepted_ranges=[],
                  row_values_read=False, original_prefix_modified=False)
    target = root / "synphys_r2.1_medium.sqlite.partial"
    report["raw_path"] = str(target)
    write_receipt(root / "started.yaml", report)
    try:
        report["copied_prefix_sha256"] = copy_prefix(Path(reg["prefix_path"]), target,
                                                     reg["prefix_bytes"], reg["prefix_sha256"])
        for index, (first, last) in enumerate(ranges(reg["prefix_bytes"], reg["expected_bytes"], reg["range_bytes"])):
            remaining = reg["maximum_total_seconds"] - (time.monotonic() - started)
            if remaining <= 0:
                raise ValueError("registered total time limit reached")
            timeout = min(reg["maximum_seconds_per_request"], remaining)
            extent = last - first + 1
            chunk = root / f"range-{index:02d}.partial"
            headers, stderr = root / f"range-{index:02d}-headers.txt", root / f"range-{index:02d}-stderr.txt"
            report["attempted_range"] = {"index": index, "first": first, "last": last,
                                          "raw_path": str(chunk)}
            write_receipt(root / "checkpoint.yaml", report)
            args = ["/usr/bin/curl", "--silent", "--show-error", "--fail", "--proto", "=https",
                    "--max-time", str(timeout), "--max-filesize", str(extent),
                    "--cacert", reg["certificate_authority_bundle"], "--dump-header", str(headers),
                    "--header", "If-Match: " + reg["expected_etag"], "--header", f"Range: bytes={first}-{last}",
                    "--header", "Accept-Encoding: identity", reg["url"]]
            with (
                chunk.open("xb") as output,
                stderr.open("xb") as error,
                subprocess.Popen(args, stdout=subprocess.PIPE, stderr=error) as proc,
            ):
                try:
                    count, digest = stream_body(proc.stdout, output, extent, reg["stream_chunk_bytes"])
                    if count > extent:
                        proc.kill()
                    code = proc.wait(timeout=timeout + 5)
                except BaseException:
                    proc.kill()
                    proc.wait()
                    raise
            report["attempted_range"].update(bytes=count, sha256=digest, client_exit_code=code)
            write_receipt(root / "checkpoint.yaml", report)
            if code != 0 or count != extent:
                raise ValueError("range client exit or exact body length failed")
            with headers.open("rb") as handle:
                verified = validate_headers(handle.read(65537), first, last,
                                            reg["expected_bytes"], reg["expected_etag"])
            if target.stat().st_size != first:
                raise ValueError("assembled prefix boundary differs")
            with target.open("ab") as output, chunk.open("rb") as source:
                appended, appended_hash = stream_body(source, output, extent)
            if appended != extent or appended_hash != digest or target.stat().st_size != last + 1:
                raise ValueError("retained range changed during append")
            report["accepted_ranges"].append(dict(report["attempted_range"], headers=verified,
                                                  header_path=str(headers), stderr_path=str(stderr)))
            write_receipt(root / "checkpoint.yaml", report)
        if time.monotonic() - started > reg["maximum_total_seconds"]:
            raise ValueError("registered total time limit reached")
        if target.stat().st_size != reg["expected_bytes"]:
            raise ValueError("assembled object exact length failed")
        with target.open("rb") as handle:
            if handle.read(16) != b"SQLite format 3\x00":
                raise ValueError("assembled object signature failed")
        final = root / "synphys_r2.1_medium.sqlite"
        target.rename(final)
        report["raw_path"] = str(final)
        report.update(bytes=final.stat().st_size, sha256=checksum(final), schema=inspect_schema(final))
        if (Path(reg["prefix_path"]).stat().st_size != reg["prefix_bytes"]
            or checksum(reg["prefix_path"]) != reg["prefix_sha256"]):
            raise ValueError("original prefix changed during continuation")
        if time.monotonic() - started > reg["maximum_total_seconds"]:
            raise ValueError("registered total time limit reached during final verification")
        report["status"] = "medium-range-acquired-schema-only"
    except (OSError, ValueError, KeyError, sqlite3.Error, subprocess.TimeoutExpired) as exc:
        report["error"] = str(exc)
    report.update(finished_at_utc=datetime.now(UTC).isoformat(), elapsed_seconds=time.monotonic() - started)
    assembled = Path(report["raw_path"])
    if assembled.exists() and report["status"] != "medium-range-acquired-schema-only":
        report.update(bytes=assembled.stat().st_size, sha256=checksum(assembled))
    attempted = report.get("attempted_range")
    if attempted and Path(attempted["raw_path"]).exists():
        attempted.update(bytes=Path(attempted["raw_path"]).stat().st_size,
                         sha256=checksum(attempted["raw_path"]))
    write_receipt(root / "manifest.yaml", report)
    print(report["status"])
    raise SystemExit(0 if report["status"] == "medium-range-acquired-schema-only" else 1)


if __name__ == "__main__":
    main()
