"""Pinned source identity and static inspection boundaries."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture
def collector(monkeypatch):
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    name = "native_launch_source_test"
    spec = importlib.util.spec_from_file_location(
        name, Path("scripts/collect_sst_vip_native_launch_sources.py")
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def test_source_is_not_executed(collector):
    raw = b"raise RuntimeError('never execute')\ndef launch(): pass\n"
    blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    result = collector.inspect_source(raw, entry={"sha": blob}, python=True)
    assert result["definitions"] == ["launch"]
    assert not result["source_executed"]


def test_blob_mismatch_rejected(collector):
    with pytest.raises(ValueError, match="pinned"):
        collector.inspect_source(b"x", entry={"sha": "wrong"}, python=False)
