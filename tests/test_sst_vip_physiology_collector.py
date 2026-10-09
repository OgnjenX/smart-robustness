"""Collector never replaces saved failures or reads changed server resources."""

from __future__ import annotations

import importlib.util
import io
import sys
from email.message import Message
from pathlib import Path

import pytest


@pytest.fixture
def collector(monkeypatch):
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    name = "physiology_body_collector_test"
    spec = importlib.util.spec_from_file_location(
        name, Path("scripts/collect_sst_vip_physiology.py")
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def inputs():
    head = {
        "source": {
            "file_id": 3,
            "resource": {
                "download_link": "/api/v2/well_known_file_download/3",
                "well_known_file_type": {"name": "NWBDownload"},
            },
        },
        "request_url": "https://api.brain-map.org/api/v2/well_known_file_download/3",
        "status": "size-resolved",
        "content_length": 4,
        "response_headers": [["Content-Length", "4"], ["ETag", '"v1"']],
    }
    reg = {
        "source_base": "https://api.brain-map.org",
        "maximum_nwb_bytes": 10,
        "maximum_morphology_bytes": 10,
        "minimum_free_disk_bytes": 0,
        "timeout_seconds": 30,
        "chunk_bytes": 2,
    }
    return head, reg, {"registration_sha256": "sealed"}


def transport(monkeypatch, collector, head, *, changed=None, body=b"abcd"):
    calls = []

    class Response(io.BytesIO):
        status = 200
        headers = Message()
        headers["Content-Length"] = "4"
        headers["ETag"] = '"v1"'

        def geturl(self):
            return head["request_url"]

    if changed:
        key, value = changed
        if key in Response.headers:
            Response.headers.replace_header(key, value)
        else:
            Response.headers[key] = value

    class Opener:
        def open(self, request, timeout):
            calls.append(request)
            return Response(body)

    monkeypatch.setattr(collector, "build_opener", lambda *args: Opener())
    return calls


def test_exact_checkpoint_reuse_no_request(collector, monkeypatch, tmp_path, inputs):
    head, reg, context = inputs
    calls = transport(monkeypatch, collector, head)
    first = collector.collect_one(head, reg=reg, context=context, directory=tmp_path)
    second = collector.collect_one(head, reg=reg, context=context, directory=tmp_path)
    assert first == second and len(calls) == 1
    assert first["status"] == "complete-receipt-awaiting-format-validation"


@pytest.mark.parametrize(
    "changed", [("Content-Length", "5"), ("ETag", '"v2"'), ("Content-Encoding", "gzip")]
)
def test_header_drift_saved_failure_no_retry(collector, monkeypatch, tmp_path, inputs, changed):
    head, reg, context = inputs
    calls = transport(monkeypatch, collector, head, changed=changed)
    result = collector.collect_one(head, reg=reg, context=context, directory=tmp_path)
    assert result["status"] == "failed" and "raw_path" not in result
    assert not (tmp_path / "3.nwb").exists()
    assert collector.collect_one(head, reg=reg, context=context, directory=tmp_path) == result
    assert len(calls) == 1


def test_partial_receipt_failure_retained(collector, monkeypatch, tmp_path, inputs):
    head, reg, context = inputs
    calls = transport(monkeypatch, collector, head, body=b"ab")
    result = collector.collect_one(head, reg=reg, context=context, directory=tmp_path)
    assert result["status"] == "failed" and result["raw_bytes"] == 2
    assert collector.collect_one(head, reg=reg, context=context, directory=tmp_path) == result
    assert len(calls) == 1


def test_orphan_not_overwritten(collector, tmp_path, inputs):
    head, reg, context = inputs
    with (tmp_path / "3.nwb").open("xb") as handle:
        handle.write(b"original")
    with pytest.raises(ValueError, match="orphan"):
        collector.collect_one(head, reg=reg, context=context, directory=tmp_path)
    assert (tmp_path / "3.nwb").read_bytes() == b"original"


def test_checkpoint_tampering_rejected(collector, monkeypatch, tmp_path, inputs):
    head, reg, context = inputs
    transport(monkeypatch, collector, head)
    saved = collector.collect_one(head, reg=reg, context=context, directory=tmp_path)
    with pytest.raises(ValueError, match="implementation"):
        collector.check_saved(
            saved, head=head, context={"registration_sha256": "changed"}, path=tmp_path / "3.nwb"
        )
    with (tmp_path / "3.nwb").open("r+b") as handle:
        handle.write(b"zzzz")
    with pytest.raises(ValueError, match="receipt changed"):
        collector.check_saved(saved, head=head, context=context, path=tmp_path / "3.nwb")
