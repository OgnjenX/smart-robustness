"""Synthetic TAR safety and first-line-only schema inspection contracts."""

from __future__ import annotations

import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest
import yaml

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/acquire_patchseq_annotation_schema.py"
spec = importlib.util.spec_from_file_location("annotation_schema", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def archive(path, members, mode="w"):
    with tarfile.open(path, mode) as target:
        for name, body, kind in members:
            info = tarfile.TarInfo(name)
            info.type = kind
            info.size = len(body) if kind == tarfile.REGTYPE else 0
            if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
                info.linkname = "outside-target"
            target.addfile(info, io.BytesIO(body) if info.isfile() else None)


def test_only_header_and_member_metadata_are_returned(tmp_path):
    path = tmp_path / "metadata.tar"
    archive(path, [(module.CSV_NAME, b"specimen_id,class_label,layer\n123456,Vip,5\n", tarfile.REGTYPE)])
    result = module.inspect_archive(path)
    assert result["columns"] == ["specimen_id", "class_label", "layer"]
    assert result["CSV_member"] == module.CSV_NAME
    assert result["annotation_row_values_interpreted"] is False
    assert result["archive_extracted_to_disk"] is False
    assert "Vip" not in json.dumps(result)
    assert not (tmp_path / module.CSV_NAME).exists()


def test_declared_same_named_directory_layout(tmp_path):
    path = tmp_path / "metadata.tar"
    directory = "20200625_patchseq_metadata_mouse"
    archive(path, [(directory + "/", b"", tarfile.DIRTYPE),
                   (directory + "/" + module.CSV_NAME, b"specimen_id\n42\n", tarfile.REGTYPE)])
    assert module.inspect_archive(path)["columns"] == ["specimen_id"]


@pytest.mark.parametrize("name,kind", [("../escape", tarfile.REGTYPE), ("/absolute", tarfile.REGTYPE),
                                     ("a\\escape", tarfile.REGTYPE), ("a//b", tarfile.REGTYPE),
                                     ("a/./b", tarfile.REGTYPE), ("symlink", tarfile.SYMTYPE),
                                     ("hardlink", tarfile.LNKTYPE), ("device", tarfile.CHRTYPE),
                                     ("fifo", tarfile.FIFOTYPE)])
def test_unsafe_archive_member_rejected_without_extraction(tmp_path, name, kind):
    path = tmp_path / "metadata.tar"
    archive(path, [(module.CSV_NAME, b"specimen_id\n42\n", tarfile.REGTYPE), (name, b"x", kind)])
    with pytest.raises(ValueError):
        module.inspect_archive(path)


@pytest.mark.parametrize("header", [b"", b",layer\n", b"id,id\n", b"x"*65537 + b"\n",
                                    (",".join(f"c{i}" for i in range(257)) + "\n").encode()])
def test_header_size_column_and_missingness_bounds(header, tmp_path):
    path = tmp_path / "metadata.tar"
    archive(path, [(module.CSV_NAME, header, tarfile.REGTYPE)])
    with pytest.raises(ValueError):
        module.inspect_archive(path)


def test_duplicate_declared_csv_candidates_not_first_row_selection(tmp_path):
    path = tmp_path / "metadata.tar"
    archive(path, [(name, b"specimen_id\n42\n", tarfile.REGTYPE) for name in sorted(module.CSV_MEMBERS)])
    with pytest.raises(ValueError, match="exactly one"):
        module.inspect_archive(path)


def test_member_count_and_declared_payload_bound(tmp_path):
    path = tmp_path / "metadata.tar"
    archive(path, [(module.CSV_NAME, b"id\n42\n", tarfile.REGTYPE)] + [(f"extra-{i}", b"", tarfile.REGTYPE) for i in range(20)])
    with pytest.raises(ValueError):
        module.inspect_archive(path)
    archive(path, [(module.CSV_NAME, b"id\n" + b"x"*2097152, tarfile.REGTYPE)])
    with pytest.raises(ValueError, match="member-byte"):
        module.inspect_archive(path)


def test_compression_not_silently_enabled(tmp_path):
    path = tmp_path / "metadata.tar"
    archive(path, [(module.CSV_NAME, b"id\n42\n", tarfile.REGTYPE)], mode="w:gz")
    with pytest.raises(tarfile.TarError):
        module.inspect_archive(path)


def response(path):
    path.write_bytes(b'HTTP/1.1 200 OK\r\nContent-Length: 1075200\r\nETag: "106800-5a9f4c80e2eec"\r\n\r\n')
    return yaml.safe_load(module.REGISTRATION.read_text())


def test_exact_transport_response_identity(tmp_path):
    path = tmp_path / "headers"
    reg = response(path)
    assert module.response_identity(path, reg)["content-length"] == "1075200"


@pytest.mark.parametrize("old,new", [(b"200", b"302"), (b"1075200", b"1075201"),
                                   (b"106800-5a9f4c80e2eec", b"different")])
def test_changed_transport_identity_rejects(old, new, tmp_path):
    path = tmp_path / "headers"
    reg = response(path)
    path.write_bytes(path.read_bytes().replace(old, new))
    with pytest.raises(ValueError):
        module.response_identity(path, reg)


@pytest.mark.parametrize("extra", [b"Content-Length: 1075200", b"Content-Encoding: gzip"])
def test_duplicate_header_or_encoded_object_rejects(extra, tmp_path):
    path = tmp_path / "headers"
    reg = response(path)
    path.write_bytes(path.read_bytes().replace(b"\r\n\r\n", b"\r\n" + extra + b"\r\n\r\n"))
    with pytest.raises(ValueError):
        module.response_identity(path, reg)


@pytest.mark.parametrize("key,value", [("annotation_row_interpretation_authorized", True),
                                       ("maximum_requests", 2), ("expected_bytes", 1075201),
                                       ("archive_acquisition_authorized", 1)])
def test_fixed_scope_cannot_expand_or_coerce(key, value):
    reg = yaml.safe_load(module.REGISTRATION.read_text())
    module.validate_registration(reg)
    reg[key] = value
    with pytest.raises(ValueError):
        module.validate_registration(reg)
