"""Synthetic range-transport contracts; no network or scientific database reads."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import sys
from itertools import pairwise
from pathlib import Path

import pytest
import yaml

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("medium_ranges", SCRIPTS / "acquire_synphys_medium_ranges.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def headers():
    return b'HTTP/1.1 206 Partial Content\r\nContent-Range: bytes 10-19/100\r\nContent-Length: 10\r\nETag: "sealed"\r\n\r\n'


def test_exact_response_identity_and_proxy_interim_blocks():
    response = module.validate_headers(b"HTTP/1.1 200 Connection established\r\n\r\n" + headers(), 10, 19, 100, '"sealed"')
    assert response["content-range"] == "bytes 10-19/100"
    assert response["etag"] == '"sealed"'


@pytest.mark.parametrize("old,new", [
    (b"206", b"200"), (b"10-19/100", b"0-9/100"), (b"10-19/100", b"10-19/101"),
    (b"Content-Length: 10", b"Content-Length: 9"), (b'"sealed"', b'"changed"'),
    (b"ETag: ", b"Missing: "),
])
def test_changed_range_or_identity_rejected(old, new):
    with pytest.raises(ValueError):
        module.validate_headers(headers().replace(old, new), 10, 19, 100, '"sealed"')


@pytest.mark.parametrize("extra", [b"ETag: \"sealed\"", b"Content-Length: 10", b"Content-Range: bytes 10-19/100",
                                  b"Content-Encoding: gzip", b" folded-header", b"malformed"])
def test_duplicate_identity_encoding_and_malformed_headers_rejected(extra):
    raw = headers().replace(b"\r\n\r\n", b"\r\n" + extra + b"\r\n\r\n")
    with pytest.raises(ValueError):
        module.validate_headers(raw, 10, 19, 100, '"sealed"')


def test_last_response_block_must_match_not_earlier_valid_block():
    with pytest.raises(ValueError):
        module.validate_headers(headers() + b"HTTP/1.1 302 Found\r\n\r\n", 10, 19, 100, '"sealed"')


def test_header_size_bound():
    with pytest.raises(ValueError):
        module.validate_headers(b"x" * 65537, 10, 19, 100, '"sealed"')


def test_range_plan_covers_remaining_bytes_exactly_once():
    plan = list(module.ranges(3573501533, 11125997568, 536870912))
    assert len(plan) == 15
    assert plan[0][0] == 3573501533
    assert plan[-1][1] == 11125997567
    assert all(a[1] + 1 == b[0] for a, b in pairwise(plan))
    assert sum(last - first + 1 for first, last in plan) == 11125997568 - 3573501533


@pytest.mark.parametrize("arguments", [(0, 100, 10), (100, 100, 10), (10, 100, 0), (True, 100, 10)])
def test_invalid_range_geometry(arguments):
    with pytest.raises(ValueError):
        list(module.ranges(*arguments))


@pytest.mark.parametrize("length", [0, 9, 10, 11, 100])
def test_bounded_body_and_retained_partial_hash(length):
    output = io.BytesIO()
    count, digest = module.stream_body(io.BytesIO(b"x" * length), output, 10, 3)
    assert count == min(length, 11)
    assert digest == hashlib.sha256(output.getvalue()).hexdigest()


def test_stream_requires_bounded_reads():
    class Guarded(io.BytesIO):
        def read(self, size=-1):
            assert 0 < size <= 4
            return super().read(size)

    assert module.stream_body(Guarded(b"abcdefghij"), io.BytesIO(), 10, 4)[0] == 10


def test_original_prefix_copied_without_modification_and_exclusive_output(tmp_path):
    raw = b"SQLite format 3\x00" + b"recorded-prefix"
    source, target = tmp_path / "original", tmp_path / "new"
    source.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    before = source.stat()
    assert module.copy_prefix(source, target, len(raw), digest) == digest
    assert target.read_bytes() == raw == source.read_bytes()
    assert before.st_mtime_ns == source.stat().st_mtime_ns
    with pytest.raises(FileExistsError):
        module.copy_prefix(source, target, len(raw), digest)


@pytest.mark.parametrize("fault", ["length", "hash", "signature"])
def test_changed_prefix_never_accepted(tmp_path, fault):
    raw = b"SQLite format 3\x00" + b"recorded-prefix"
    source, target = tmp_path / "original", tmp_path / "new"
    source.write_bytes(b"not SQLite" if fault == "signature" else raw)
    with pytest.raises(ValueError):
        module.copy_prefix(source, target, source.stat().st_size + (fault == "length"),
                           "wrong" if fault == "hash" else hashlib.sha256(raw).hexdigest())
    assert source.read_bytes() == (b"not SQLite" if fault == "signature" else raw)


@pytest.mark.parametrize("key,value", [("row_reads_authorized", True), ("maximum_requests", 16),
                                       ("expected_bytes", 11125997567), ("prefix_bytes", 0),
                                       ("database_download_authorized", 1), ("maximum_total_seconds", 36000)])
def test_registration_cannot_expand_or_coerce_scope(key, value):
    reg = yaml.safe_load(module.REGISTRATION.read_text())
    module.validate_registration(reg)
    reg[key] = value
    with pytest.raises(ValueError):
        module.validate_registration(reg)


def test_missing_seal_denies_execution_before_any_request(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "SEAL", tmp_path / "no-seal")
    with pytest.raises(FileNotFoundError):
        module.main()


def simulated_execution(tmp_path, monkeypatch, fault=None):
    """Exercise orchestration on tiny synthetic objects, bypassing fixed geometry only."""
    from types import SimpleNamespace

    prefix = b"SQLite format 3\x00"
    source = tmp_path / "original.partial"
    source.write_bytes(prefix)
    parent = tmp_path / "parent.yaml"
    assessment, ca = tmp_path / "assessment", tmp_path / "ca"
    assessment.write_text("synthetic parent assessment")
    ca.write_text("synthetic CA fixture")
    root = tmp_path / "continuation"
    reg = yaml.safe_load(module.REGISTRATION.read_text())
    reg.update(prefix_path=str(source), prefix_bytes=len(prefix), prefix_sha256=module.checksum(source),
               parent_manifest=str(parent), parent_assessment=str(assessment), certificate_authority_bundle=str(ca),
               output_directory=str(root), range_bytes=2, expected_bytes=len(prefix) + 4)
    parent.write_text(yaml.safe_dump({"status": "failure-retained", "client_exit_code": 28,
                                    "bytes": len(prefix), "sha256": reg["prefix_sha256"],
                                    "raw_path": str(source), "row_values_read": False}))
    for key in ("parent_manifest", "parent_assessment"):
        reg[key + "_sha256"] = module.checksum(reg[key])
    reg["certificate_authority_sha256"] = module.checksum(ca)
    registration = tmp_path / "registration.yaml"
    registration.write_text(yaml.safe_dump(reg))
    seal = tmp_path / "seal.yaml"
    seal.write_text(yaml.safe_dump({"execution_authorized": True, "registration_sha256": module.checksum(registration),
                                    "collector_sha256": module.checksum(module.__file__),
                                    "schema_inspector_sha256": module.checksum(SCRIPTS / "acquire_synphys_small_schema.py")}))
    monkeypatch.setattr(module, "REGISTRATION", registration)
    monkeypatch.setattr(module, "SEAL", seal)
    monkeypatch.setattr(module, "validate_registration", lambda _: None)
    monkeypatch.setattr(module.shutil, "disk_usage", lambda _: SimpleNamespace(free=10**12))
    monkeypatch.setattr(module, "inspect_schema", lambda _: {"tables": [], "data_rows_read": False})
    calls = []

    class Client:
        def __init__(self, args, **kwargs):
            calls.append(args)
            assert "--cacert" in args and "--location" not in args and "--insecure" not in args
            range_arg = next(value for value in args if value.startswith("Range: bytes="))
            first, last = map(int, range_arg.split("=")[1].split("-"))
            body = b"ab" if first == 16 else b"cd"
            if fault == "overlong":
                body += b"excess"
            self.stdout = io.BytesIO(body)
            self.code = 28 if fault == "timeout" else 0
            status = "200 OK" if fault == "ignored-range" else "206 Partial Content"
            etag = '"wrong"' if fault == "etag" else reg["expected_etag"]
            response = (f"HTTP/1.1 {status}\r\nContent-Range: bytes {first}-{last}/{reg['expected_bytes']}\r\n"
                        f"Content-Length: {last-first+1}\r\nETag: {etag}\r\n\r\n")
            Path(args[args.index("--dump-header") + 1]).write_bytes(response.encode("ascii"))

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stdout.close()

        def wait(self, **kwargs):
            return self.code

        def kill(self):
            self.code = -9

    monkeypatch.setattr(module.subprocess, "Popen", Client)
    return source, root, calls


def test_complete_sequential_execution_preserves_prefix_and_rejects_duplicate(tmp_path, monkeypatch):
    source, root, calls = simulated_execution(tmp_path, monkeypatch)
    original = source.read_bytes()
    with pytest.raises(SystemExit) as exit_status:
        module.main()
    assert exit_status.value.code == 0
    receipt = yaml.safe_load((root / "manifest.yaml").read_text())
    assert receipt["status"] == "medium-range-acquired-schema-only"
    assert len(receipt["accepted_ranges"]) == len(calls) == 2
    assert (root / "synphys_r2.1_medium.sqlite").read_bytes() == original + b"abcd"
    assert receipt["sha256"] == module.checksum(root / "synphys_r2.1_medium.sqlite")
    assert source.read_bytes() == original
    with pytest.raises(FileExistsError):
        module.main()
    assert len(calls) == 2


@pytest.mark.parametrize("fault", ["timeout", "ignored-range", "etag", "overlong"])
def test_failed_range_retained_separately_without_append_or_retry(tmp_path, monkeypatch, fault):
    source, root, calls = simulated_execution(tmp_path, monkeypatch, fault)
    original = source.read_bytes()
    with pytest.raises(SystemExit) as exit_status:
        module.main()
    assert exit_status.value.code == 1
    receipt = yaml.safe_load((root / "manifest.yaml").read_text())
    assert receipt["accepted_ranges"] == []
    assert len(calls) == 1
    assert (root / "synphys_r2.1_medium.sqlite.partial").read_bytes() == original
    assert receipt["attempted_range"]["sha256"] == module.checksum(root / "range-00.partial")
    assert source.read_bytes() == original
