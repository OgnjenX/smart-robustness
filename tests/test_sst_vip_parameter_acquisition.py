"""Parameter source and byte/JSON boundaries, without any remote requests."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from smart_robustness.validation.sst_vip_parameter_acquisition import (
    acquisition_plan,
    validate_json,
    validate_url,
)


@pytest.mark.parametrize(
    "url",
    [
        "http://api.brain-map.org/file",
        "https://example.com/file",
        "https://api.brain-map.org:444/file",
        "https://user@api.brain-map.org/file",
        "https://api.brain-map.org.example.com/file",
    ],
)
def test_unregistered_source_rejected(url):
    with pytest.raises(ValueError):
        validate_url(url, "https://api.brain-map.org")


def test_registered_https_and_json_object():
    validate_url("https://api.brain-map.org/api/v2/file", "https://api.brain-map.org")
    assert validate_json(b'{"b": 1, "a": [2]}', 100) == ["a", "b"]


@pytest.mark.parametrize("raw,limit", [(b"[]", 100), (b"not JSON", 100), (b"{}", 1)])
def test_invalid_shape_parse_or_size_rejected(raw, limit):
    with pytest.raises((ValueError, TypeError)):
        validate_json(raw, limit)


def test_plan_uses_exact_attachment_and_no_missing_file_replacement():
    rules = {
        "path_format": "/api/v2/well_known_file_download/{file_id}",
        "source_base": "https://api.brain-map.org",
        "required_file_type": "NeuronalModelParameters",
        "expected_unique_file_count": 1,
        "expected_model_count": 1,
    }
    resource = {
        "id": 3,
        "download_link": "/api/v2/well_known_file_download/3",
        "attachable_id": 2,
        "attachable_type": "NeuronalModel",
        "well_known_file_type": {"name": "NeuronalModelParameters"},
    }
    source = {
        "records": [
            {
                "id": 1,
                "neuronal_models": [
                    {"id": 2, "neuronal_model_template": {}, "well_known_files": [resource]}
                ],
            }
        ]
    }
    assert acquisition_plan(source, rules)[0]["specimen_id"] == 1
    resource["attachable_id"] = 99
    with pytest.raises(ValueError):
        acquisition_plan(source, rules)


@pytest.fixture
def collector():
    spec = importlib.util.spec_from_file_location(
        "parameter_collector_test", Path("scripts/collect_sst_vip_parameters.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def collect_fixture(collector, tmp_path, monkeypatch, raw=b'{"a": 1}', error=False):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def geturl(self):
            return "https://api.brain-map.org/api/v2/well_known_file_download/3"

        def read(self, size):
            return raw[:size]

    class Opener:
        def open(self, url, timeout):
            if error:
                raise OSError("fixture request failure")
            return Response()

    monkeypatch.setattr(collector, "build_opener", lambda *args: Opener())
    args = {
        "directory": tmp_path,
        "rules": {
            "maximum_bytes_per_file": 100,
            "timeout_seconds": 30,
            "source_base": "https://api.brain-map.org",
        },
        "identity": {"test": "sealed"},
    }
    record = {
        "file_id": 3,
        "request_url": "https://api.brain-map.org/api/v2/well_known_file_download/3",
    }
    return record, args, collector.collect_one(record, **args)


def test_saved_success_resume_checks_hash_and_identity(collector, tmp_path, monkeypatch):
    record, args, saved = collect_fixture(collector, tmp_path, monkeypatch)
    assert saved["status"] == "downloaded-valid-json"
    monkeypatch.setattr(collector, "build_opener", lambda *args: pytest.fail("resume requested"))
    assert collector.collect_one(record, **args) == saved
    args["identity"] = {"test": "changed"}
    with pytest.raises(ValueError, match="identity"):
        collector.collect_one(record, **args)


def test_tampered_raw_resume_rejected(collector, tmp_path, monkeypatch):
    record, args, _ = collect_fixture(collector, tmp_path, monkeypatch)
    (tmp_path / "3.raw.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="hash"):
        collector.collect_one(record, **args)


def test_failed_request_not_silently_retried(collector, tmp_path, monkeypatch):
    record, args, saved = collect_fixture(collector, tmp_path, monkeypatch, error=True)
    assert saved["status"] == "request-failed"
    assert saved["error_type"] == "OSError"
    monkeypatch.setattr(collector, "build_opener", lambda *args: pytest.fail("failure retried"))
    assert collector.collect_one(record, **args) == saved


def test_invalid_json_bytes_preserved(collector, tmp_path, monkeypatch):
    _, _, saved = collect_fixture(collector, tmp_path, monkeypatch, raw=b"not JSON")
    assert saved["status"] == "request-failed"
    assert (tmp_path / "3.raw.json").read_bytes() == b"not JSON"


def test_orphan_raw_not_overwritten(collector, tmp_path):
    path = tmp_path / "3.raw.json"
    path.write_bytes(b"{}")
    with pytest.raises(ValueError, match="orphan"):
        collector.collect_one({"file_id": 3}, directory=tmp_path, rules={}, identity={})
    assert path.read_bytes() == b"{}"
