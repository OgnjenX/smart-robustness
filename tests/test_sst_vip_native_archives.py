"""Native source/archive inspection is bounded and never executes members."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path

import pytest
import yaml

from smart_robustness.validation.sst_vip_native_archives import inspect_channel, inspect_package

RULES = {
    "maximum_compressed_bytes_per_package": 10000,
    "maximum_archive_entries": 10,
    "maximum_total_uncompressed_bytes": 10000,
    "maximum_manifest_json_bytes": 1000,
    "expected_manifest_basename": "manifest.json",
}


def archive(entries):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        for name, contents in entries:
            z.writestr(name, contents)
    return buffer.getvalue()


def test_manifest_read_without_extracting_or_executing_code():
    description = {
        "biophys": [{"model_type": "Biophysical - all active"}, {"axon_type": "stub"}],
        "neuron": [{"hoc": ["stdgui.hoc"]}],
    }
    raw = archive(
        [
            ("manifest.json", json.dumps(description)),
            ("evil.py", "raise RuntimeError('never execute')"),
        ]
    )
    result = inspect_package(raw, RULES)
    assert result["description"] == description and result["setup_resolved"]
    assert not result["source_executed"] and not result["archive_extracted"]


@pytest.mark.parametrize(
    "path", ["../manifest.json", "/manifest.json", "C:/manifest.json", "folder\\manifest.json"]
)
def test_unsafe_paths_rejected(path):
    with pytest.raises(ValueError, match="unsafe"):
        inspect_package(archive([(path, "{}")]), RULES)


def test_missing_and_ambiguous_manifests_not_silently_selected():
    for entries in ([("other.json", "{}")], [("a/manifest.json", "{}"), ("b/manifest.json", "{}")]):
        result = inspect_package(archive(entries), RULES)
        assert result["setup_error"] == "missing-or-ambiguous-manifest"
        assert not result["setup_resolved"]


def test_absent_axon_mode_is_not_defaulted():
    raw = archive([("manifest.json", '{"biophys": [{"model_type": "Biophysical - perisomatic"}]}')])
    result = inspect_package(raw, RULES)
    assert result["setup_error"] == "missing-authoritative-axon-type"
    assert result["declared_axon_types"] == []


def test_entry_compressed_expanded_and_manifest_limits():
    raw = archive([("manifest.json", '{"a": 1}')])
    for key in (
        "maximum_compressed_bytes_per_package",
        "maximum_archive_entries",
        "maximum_total_uncompressed_bytes",
        "maximum_manifest_json_bytes",
    ):
        with pytest.raises(ValueError):
            inspect_package(raw, RULES | {key: 0})


def test_source_inspection_does_not_claim_equation_validation():
    result = inspect_channel(b"NEURON { SUFFIX NaTs USEION na READ ena WRITE ina }", 1000)
    assert result["declared_mechanisms"] == ["NaTs"]
    assert result["dependency_constructs"] == ["USEION"]
    assert not result["equations_validated"]


@pytest.fixture
def collector(monkeypatch):
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    name = "native_archives_collector_test"
    spec = importlib.util.spec_from_file_location(
        name, Path("scripts/collect_sst_vip_channels_and_packages.py")
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def test_bad_zip_preserves_received_bytes_and_failure(collector, monkeypatch, tmp_path):
    class Response(io.BytesIO):
        def geturl(self):
            return "https://api.brain-map.org/neuronal_model/download/1"

    class Opener:
        def open(self, url, timeout):
            return Response(b"not a zip")

    monkeypatch.setattr(collector, "build_opener", lambda *args: Opener())
    task = {
        "url": "https://api.brain-map.org/neuronal_model/download/1",
        "kind": "package",
        "filename": "package-1.zip",
        "maximum_bytes": 100,
    }
    reg = {
        "channels": {"source_base": "https://api.brain-map.org"},
        "transport": {"timeout_seconds": 30},
        "packages": RULES,
    }
    result = collector.collect_one(task, reg=reg, directory=tmp_path)
    assert result["status"] == "failed" and result["error_type"] == "BadZipFile"
    assert (tmp_path / task["filename"]).read_bytes() == b"not a zip"
    assert yaml.safe_load((tmp_path / "package-1.zip.provenance.yaml").read_text()) == result


def test_unregistered_host_rejected_before_transport(collector, monkeypatch, tmp_path):
    def forbidden(*args):
        pytest.fail("must not create transport for an unregistered host")

    monkeypatch.setattr(collector, "build_opener", forbidden)
    task = {"url": "https://example.org/model", "filename": "bad.zip"}
    reg = {"channels": {"source_base": "https://api.brain-map.org"}}
    result = collector.collect_one(task, reg=reg, directory=tmp_path)
    assert result["status"] == "failed" and result["error_type"] == "ValueError"
    assert "raw_path" not in result
