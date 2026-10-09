"""Bounded metadata-only acquisition controls; no live network in tests."""

from __future__ import annotations

import importlib.util
import ssl
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


def test_retry_inherits_original_resources_and_checks_lineage(tmp_path):
    original = tmp_path / "original.yaml"
    parent = registration()
    original.write_text(yaml.safe_dump(parent))
    previous, prior = tmp_path / "assessment.yaml", tmp_path / "prior.yaml"
    previous.write_text("failure assessment")
    prior.write_text("retained failure manifest")
    retry = {
        "parent_registration": str(original),
        "parent_registration_sha256": MODULE.digest(original),
        "previous_assessment": str(previous),
        "previous_assessment_sha256": MODULE.digest(previous),
        "prior_result": str(prior),
        "prior_result_sha256": MODULE.digest(prior),
        "output_directory": str(tmp_path / "new"),
        "certificate_authority_bundle": "fixture",
        "certificate_authority_sha256": "fixture",
        "insecure_transport_authorized": False,
    }
    for key in (
        "database_download_authorized",
        "physiological_response_or_numeric_synaptic_result_reads_authorized",
        "new_parameter_fitting_authorized",
        "cell_or_network_simulation_authorized",
    ):
        retry[key] = False
    path = tmp_path / "retry.yaml"
    path.write_text(yaml.safe_dump(retry))
    loaded = MODULE.load_registration(path)
    assert (
        loaded["resources"] == parent["resources"]
        and loaded["allowed_hosts"] == parent["allowed_hosts"]
    )
    assert loaded["output_directory"] != parent["output_directory"]
    prior.write_text("changed")
    with pytest.raises(ValueError, match="lineage"):
        MODULE.load_registration(path)


def test_changed_certificate_bundle_rejected_before_fetch(tmp_path):
    ca = tmp_path / "ca.pem"
    ca.write_text("fixture")
    with pytest.raises(ValueError, match="CA bytes changed"):
        MODULE.tls_context(
            {"certificate_authority_bundle": str(ca), "certificate_authority_sha256": "wrong"}
        )


@pytest.mark.parametrize("hostname,mode", [(False, ssl.CERT_REQUIRED), (True, ssl.CERT_NONE)])
def test_certificate_verification_cannot_be_disabled(monkeypatch, tmp_path, hostname, mode):
    ca = tmp_path / "ca.pem"
    ca.write_text("fixture")

    class Context:
        check_hostname = hostname
        verify_mode = mode

    monkeypatch.setattr(MODULE.ssl, "create_default_context", lambda **kwargs: Context())
    with pytest.raises(ValueError, match="verification must remain"):
        MODULE.tls_context(
            {
                "certificate_authority_bundle": str(ca),
                "certificate_authority_sha256": MODULE.digest(ca),
            }
        )


def test_verified_context_uses_registered_bundle(monkeypatch, tmp_path):
    ca = tmp_path / "ca.pem"
    ca.write_text("fixture")
    calls = []

    class Context:
        check_hostname = True
        verify_mode = ssl.CERT_REQUIRED

    def create(**kwargs):
        calls.append(kwargs)
        return Context()

    monkeypatch.setattr(MODULE.ssl, "create_default_context", create)
    MODULE.tls_context(
        {"certificate_authority_bundle": str(ca), "certificate_authority_sha256": MODULE.digest(ca)}
    )
    assert calls == [{"cafile": str(ca)}]
