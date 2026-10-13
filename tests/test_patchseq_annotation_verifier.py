"""Synthetic raw-block TAR reconstruction, independent of acquisition code."""

from __future__ import annotations

import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify_patchseq_annotation_schema.py"
spec = importlib.util.spec_from_file_location("annotation_verifier", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def source(path, name=None, header=b"specimen_id,genotype\n123456,Vip\n", kind=tarfile.REGTYPE):
    with tarfile.open(path, "w", format=tarfile.USTAR_FORMAT) as archive:
        info = tarfile.TarInfo(name or module.NAME)
        info.type = kind
        info.size = len(header) if kind == tarfile.REGTYPE else 0
        if kind in (tarfile.LNKTYPE, tarfile.SYMTYPE):
            info.linkname = "outside"
        archive.addfile(info, io.BytesIO(header) if kind == tarfile.REGTYPE else None)


def test_raw_block_reconstruction_returns_no_annotation_rows(tmp_path):
    path = tmp_path / "source.tar"
    source(path)
    rebuilt = module.reconstruct(path)
    assert rebuilt["columns"] == ["specimen_id", "genotype"]
    assert rebuilt["header_bytes"] == len(b"specimen_id,genotype\n")
    assert rebuilt["annotation_row_values_interpreted"] is False
    assert rebuilt["archive_extracted_to_disk"] is False
    assert "Vip" not in json.dumps(rebuilt)
    assert not (tmp_path / module.NAME).exists()


@pytest.mark.parametrize("name,kind", [("../escape", tarfile.REGTYPE), ("/absolute", tarfile.REGTYPE),
                                     ("a\\escape", tarfile.REGTYPE), (module.NAME, tarfile.SYMTYPE),
                                     (module.NAME, tarfile.LNKTYPE), (module.NAME, tarfile.FIFOTYPE)])
def test_unsafe_paths_and_special_members_rejected(tmp_path, name, kind):
    path = tmp_path / "source.tar"
    source(path, name=name, kind=kind)
    with pytest.raises(ValueError):
        module.reconstruct(path)


@pytest.mark.parametrize("fault", ["checksum", "truncated", "terminator", "trailing"])
def test_raw_archive_integrity_controls(tmp_path, fault):
    path = tmp_path / "source.tar"
    source(path)
    raw = bytearray(path.read_bytes())
    if fault == "checksum":
        raw[10] ^= 1
    elif fault == "truncated":
        raw = raw[:512]
    elif fault == "terminator":
        raw[1536] = 1
    else:
        raw[-1] = 1
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        module.reconstruct(path)


@pytest.mark.parametrize("header", [b"", b",genotype\n", b"id,id\n", b"x"*65537,
                                    (",".join(f"c{i}" for i in range(257)) + "\n").encode()])
def test_header_bounds_and_ambiguity(tmp_path, header):
    path = tmp_path / "source.tar"
    source(path, header=header)
    with pytest.raises(ValueError):
        module.reconstruct(path)


@pytest.mark.parametrize("raw", [b"-1", b"8", b"\x80\x00", b"not-number"])
def test_extended_or_invalid_size_encoding_rejected(raw):
    with pytest.raises(ValueError):
        module.octal(raw)


def test_octal_and_zero_size():
    assert module.octal(b"0000010\x00") == 8
    assert module.octal(b"\x00 "*3) == 0


def test_transport_headers_independently_reconstructed(tmp_path):
    path = tmp_path / "headers"
    raw = b'HTTP/1.1 200 OK\r\nContent-Length: 1075200\r\nETag: "106800-5a9f4c80e2eec"\r\n\r\n'
    path.write_bytes(raw)
    assert module.check_headers(path)["content-length"] == "1075200"
    path.write_bytes(raw.replace(b"200", b"302"))
    with pytest.raises(ValueError):
        module.check_headers(path)


def test_missing_terminal_receipt_does_not_create_output(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "MANIFEST", tmp_path / "absent")
    monkeypatch.setattr(module, "OUTPUT", tmp_path / "output")
    with pytest.raises(ValueError, match="terminal archive receipt absent"):
        module.main()
    assert not module.OUTPUT.exists()
