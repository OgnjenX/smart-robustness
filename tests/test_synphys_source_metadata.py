"""Bounded metadata-only acquisition controls; no live network in tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

SPEC = importlib.util.spec_from_file_location(
    "synphys_metadata", Path(__file__).parents[1] / "scripts/collect_synphys_source_metadata.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def registration():
    return yaml.safe_load(MODULE.REGISTRATION.read_text())


def test_registration_bounds():
    MODULE.validate(registration())


@pytest.mark.parametrize(
    "key",
    [
        "database_download_authorized",
        "new_parameter_fitting_authorized",
        "cell_or_network_simulation_authorized",
    ],
)
def test_no_broader_authorization(key):
    reg = registration()
    reg[key] = True
    with pytest.raises(ValueError, match="scope"):
        MODULE.validate(reg)


@pytest.mark.parametrize(
    "url",
    [
        "http://aisynphys.readthedocs.io/",
        "https://unregistered.invalid/",
        "https://aisynphys.readthedocs.io/file.sqlite",
    ],
)
def test_host_and_content_bounds(url):
    reg = registration()
    reg["resources"][0]["url"] = url
    with pytest.raises(ValueError):
        MODULE.validate(reg)


def test_listing_preserves_truncation_and_only_metadata():
    raw = b'<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><Name>allen-synphys</Name><IsTruncated>true</IsTruncated><CommonPrefixes><Prefix>database/</Prefix></CommonPrefixes><Contents><Key>fixture.sqlite</Key><Size>123</Size><LastModified>fixture</LastModified></Contents></ListBucketResult>'
    out = MODULE.inspect_listing(raw)
    assert out["truncated"] and out["prefixes"] == ["database/"]
    assert out["objects"][0]["size"] == 123 and not out["numeric_synaptic_outcomes_read"]


def test_wrong_or_unbounded_listing_rejected():
    for raw in (
        b"<Error/>",
        b'<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"/>',
    ):
        with pytest.raises(ValueError):
            MODULE.inspect_listing(raw)


def test_acquisition_bounds_retains_oversized_receipt(monkeypatch, tmp_path):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def geturl(self):
            return "https://aisynphys.readthedocs.io/fixture"

        def read(self, size):
            assert size == 5
            return b"12345"

    monkeypatch.setattr(MODULE, "urlopen", lambda *args, **kwargs: Response())
    record = MODULE.acquire(
        {"name": "fixture", "url": "https://aisynphys.readthedocs.io/fixture", "maximum_bytes": 4},
        tmp_path,
        ["aisynphys.readthedocs.io"],
    )
    assert record["status"] == "failed-retained" and record["raw_bytes"] == 5
    assert (tmp_path / "fixture.raw").read_bytes() == b"12345"
