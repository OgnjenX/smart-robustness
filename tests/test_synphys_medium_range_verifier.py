"""Independent range verifier exercises byte corruption and ledger failures."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("range_verifier", SCRIPTS / "verify_synphys_medium_ranges.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def source(tmp_path):
    prefix_bytes = b"SQLite format 3\x00"
    prefix = tmp_path / "original.partial"
    prefix.write_bytes(prefix_bytes)
    path = tmp_path / "synphys_r2.1_medium.sqlite"
    path.write_bytes(prefix_bytes + b"abcd")
    accepted = []
    for index, body in enumerate((b"ab", b"cd")):
        first, last = 16 + 2 * index, 17 + 2 * index
        chunk = tmp_path / f"range-{index:02d}.partial"
        header = tmp_path / f"range-{index:02d}-headers.txt"
        stderr = tmp_path / f"range-{index:02d}-stderr.txt"
        chunk.write_bytes(body)
        header.write_bytes((f'HTTP/1.1 206 Partial Content\r\nContent-Range: bytes {first}-{last}/20\r\n'
                            'Content-Length: 2\r\nETag: "fixed"\r\n\r\n').encode())
        stderr.write_bytes(b"")
        accepted.append({"index": index, "first": first, "last": last, "bytes": 2,
                         "client_exit_code": 0, "raw_path": str(chunk), "header_path": str(header),
                         "stderr_path": str(stderr), "sha256": hashlib.sha256(body).hexdigest(),
                         "headers": {"content-range": f"bytes {first}-{last}/20", "content-length": "2", "etag": '"fixed"'}})
    manifest = {"status": "medium-range-acquired-schema-only", "bytes": 20, "row_values_read": False,
                "original_prefix_modified": False, "accepted_ranges": accepted,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return path, prefix, hashlib.sha256(prefix_bytes).hexdigest(), manifest


def verify(values):
    path, prefix, digest, manifest = values
    return module.verify_assembly(path, prefix, 16, digest, manifest, 20, 2, 2, '"fixed"')


def test_independent_complete_byte_and_hash_verification(tmp_path):
    result = verify(source(tmp_path))
    assert result["accepted_ranges_verified"] == 2
    assert result["original_prefix_and_every_range_byte_equal"]
    assert len(result["component_sha256"]) == 3


@pytest.mark.parametrize("target", ["assembled", "prefix", "range"])
def test_component_and_assembly_corruption_rejected_even_with_resealed_whole_hash(tmp_path, target):
    values = source(tmp_path)
    path, prefix, _, manifest = values
    altered = {"assembled": path, "prefix": prefix, "range": tmp_path / "range-00.partial"}[target]
    raw = altered.read_bytes()
    altered.write_bytes(raw[:-1] + b"X")
    manifest["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        verify(values)


@pytest.mark.parametrize("field,value", [("index", 1), ("first", 15), ("last", 18),
                                       ("bytes", 1), ("client_exit_code", 28), ("index", False),
                                       ("sha256", "wrong"), ("raw_path", "elsewhere")])
def test_gap_overlap_exit_identity_and_typed_ledger_failures(tmp_path, field, value):
    values = source(tmp_path)
    values[-1]["accepted_ranges"][0][field] = value
    with pytest.raises(ValueError):
        verify(values)


@pytest.mark.parametrize("field,value", [("status", "range-acquisition-failure-retained"),
                                       ("bytes", 19), ("row_values_read", True),
                                       ("original_prefix_modified", True), ("accepted_ranges", [])])
def test_missing_terminal_success_and_wrong_scope_rejected(tmp_path, field, value):
    values = source(tmp_path)
    values[-1][field] = value
    with pytest.raises(ValueError):
        verify(values)


@pytest.mark.parametrize("old,new", [(b"206", b"200"), (b"/20", b"/21"),
                                   (b'"fixed"', b'"other"'), (b"Content-Length: 2", b"Content-Length: 3")])
def test_independent_http_identity_checks(tmp_path, old, new):
    values = source(tmp_path)
    header = tmp_path / "range-00-headers.txt"
    header.write_bytes(header.read_bytes().replace(old, new))
    with pytest.raises(ValueError):
        verify(values)


def test_unverified_header_projection_not_trusted(tmp_path):
    values = source(tmp_path)
    values[-1]["accepted_ranges"][0]["headers"]["etag"] = '"other"'
    with pytest.raises(ValueError):
        verify(values)


def test_source_mutation_during_verification_rejected(tmp_path, monkeypatch):
    values = source(tmp_path)
    original = module.check_range_header

    def changing_header(path, *args):
        parsed = original(path, *args)
        path.write_bytes(path.read_bytes() + b"\n")
        return parsed

    monkeypatch.setattr(module, "check_range_header", changing_header)
    with pytest.raises(ValueError, match="source changed"):
        verify(values)


def test_missing_terminal_evidence_creates_no_verifier_output(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "MANIFEST", tmp_path / "absent")
    monkeypatch.setattr(module, "OUTPUT", tmp_path / "output")
    with pytest.raises(ValueError, match="terminal manifest absent"):
        module.main()
    assert not module.OUTPUT.exists()
