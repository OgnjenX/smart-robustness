"""No network, SDK import or physiological data in catalog parser controls."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def collector(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "release_catalog", scripts / "collect_synphys_release_catalog.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_revision_immutable(collector):
    assert collector.revision(json.dumps({"sha": "a" * 40})) == "a" * 40
    with pytest.raises(ValueError):
        collector.revision(json.dumps({"sha": "current-release"}))


def test_every_catalog_entry_and_alias_retained(collector):
    entry = {
        "file": "synphys_r2.0_small.sqlite",
        "aliases": ["fixture-alias"],
        "schema_version": "22",
        "unknown_future_field": "retained",
    }
    parsed = collector.inspect_catalog(
        json.dumps({"default_url_path": "https://public.invalid/", "databases": [entry]})
    )
    assert parsed["entries"][0]["declared_metadata"] == entry
    assert not parsed["database_objects_downloaded"]


@pytest.mark.parametrize("name", ["../private.sqlite", "source.nwb", "nested/database.sqlite"])
def test_no_path_guessing_or_non_database_catalog_entry(collector, name):
    with pytest.raises(ValueError):
        collector.inspect_catalog(
            json.dumps(
                {"default_url_path": "https://public.invalid/", "databases": [{"file": name}]}
            )
        )


def test_duplicate_database_rejected(collector):
    with pytest.raises(ValueError):
        collector.inspect_catalog(
            json.dumps(
                {"default_url_path": "x", "databases": [{"file": "x.sqlite"}, {"file": "x.sqlite"}]}
            )
        )


def test_source_parsed_not_executed(collector):
    raw = b'download_info_url = "https://public.invalid/catalog.json"\nraise RuntimeError("must never run")'
    assert collector.config_catalog_url(raw) == "https://public.invalid/catalog.json"
    with pytest.raises(ValueError):
        collector.config_catalog_url(b"download_info_url = some_function()")
