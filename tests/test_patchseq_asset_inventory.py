"""Synthetic metadata catalog contracts; never download experiment recordings."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from uuid import UUID

import pytest
import yaml

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/inventory_patchseq_assets.py"
spec = importlib.util.spec_from_file_location("patchseq_assets", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def rows():
    return [{"asset_id": str(UUID(int=index + 1)), "path": f"sub-10/record-{index:04d}.nwb", "size": 20}
            for index in range(4435)]


def page_document(page):
    return {"count": 4435, "next": module.page_url(page + 1) if page < 9 else None,
            "previous": module.page_url(page - 1) if page > 1 else None,
            "results": rows()[(page-1)*500:page*500]}


@pytest.mark.parametrize("page", [1, 2, 9])
def test_complete_fixed_page_projection(page):
    document = page_document(page)
    assert module.parse_page(document, page) == document["results"]


def test_first_previous_link_may_use_default_page_one():
    document = page_document(2)
    document["previous"] = document["previous"].replace("&page=1", "")
    assert len(module.parse_page(document, 2)) == 500


@pytest.mark.parametrize("link", ["https://example.com/file.nwb", module.BASE.replace("0.210913.1639", "draft") + "assets/",
                                 module.page_url(2) + "&path=selected", module.page_url(2) + "&page=2",
                                 module.page_url(2).replace("https:", "http:"), module.page_url(2) + "#fragment"])
def test_server_link_cannot_change_scope(link):
    with pytest.raises(ValueError):
        module.check_link(link, 2)


@pytest.mark.parametrize("mutation", ["count", "short", "wrong-next", "wrong-previous"])
def test_page_count_and_completeness_fail(mutation):
    document = page_document(1)
    if mutation == "count":
        document["count"] = 4434
    elif mutation == "short":
        document["results"].pop()
    elif mutation == "wrong-next":
        document["next"] = module.page_url(3)
    else:
        document["previous"] = module.page_url(1)
    with pytest.raises(ValueError):
        module.parse_page(document, 1)


@pytest.mark.parametrize("field,value", [("asset_id", "not-UUID"), ("path", "/absolute"),
                                       ("path", "../escape"), ("path", "sub-10//file"),
                                       ("path", "a\\file"), ("path", "x"*4097),
                                       ("size", True), ("size", -1), ("size", "20")])
def test_asset_identity_path_and_size_bounds(field, value):
    document = page_document(1)
    document["results"][0][field] = value
    with pytest.raises(ValueError):
        module.parse_page(document, 1)


def test_unknown_fields_are_not_used_to_classify_physiology():
    document = page_document(1)
    document["results"][0]["metadata"] = {"VIP": True, "firing_rate": 100}
    assert module.parse_page(document, 1)[0] == rows()[0]


def old_records():
    return [{"specimen_id": identity, "included": identity <= 111} for identity in range(1, 2334)]


def test_overlap_is_only_a_path_hint_not_confirmed_identity():
    summary = module.catalog_summary(rows(), old_records())
    assert summary["total_asset_bytes"] == 4435 * 20
    assert summary["potential_prior_specimen_ID_matches"] == [10]
    assert summary["potential_prior_included_specimen_ID_matches"] == [10]
    assert not summary["specimen_identity_confirmed"]
    assert not summary["VIP_or_SST_identity_confirmed"]


@pytest.mark.parametrize("mutation", ["missing", "duplicate-id", "duplicate-path", "wrong-order", "old-inventory"])
def test_complete_unique_catalog_and_prior_inventory_required(mutation):
    assets, old = rows(), old_records()
    if mutation == "missing":
        assets.pop()
    elif mutation == "duplicate-id":
        assets[-1]["asset_id"] = assets[0]["asset_id"]
    elif mutation == "duplicate-path":
        assets[-1]["path"] = assets[0]["path"]
    elif mutation == "wrong-order":
        assets.reverse()
    else:
        old.pop()
    with pytest.raises(ValueError):
        module.catalog_summary(assets, old)


def test_missing_subject_path_not_inferred():
    assets = rows()
    assets[-1]["path"] = "unknown-file.nwb"
    summary = module.catalog_summary(assets, old_records())
    assert summary["unresolved_path_asset_ids"] == [assets[-1]["asset_id"]]


@pytest.mark.parametrize("key,value", [("asset_binary_download_authorized", True), ("parameter_fitting_authorized", True),
                                       ("maximum_requests", 11), ("version", "draft"), ("metadata_acquisition_authorized", 1)])
def test_fixed_registration_scope_cannot_expand(key, value):
    reg = yaml.safe_load(module.REGISTRATION.read_text())
    module.validate_registration(reg)
    altered = copy.deepcopy(reg)
    altered[key] = value
    with pytest.raises(ValueError):
        module.validate_registration(altered)


def test_fetch_refuses_binary_or_unregistered_asset_metadata_endpoints(tmp_path):
    reg = yaml.safe_load(module.REGISTRATION.read_text())
    with pytest.raises(ValueError, match="unregistered"):
        module.fetch(module.BASE + "assets/fake/download/", tmp_path / "not-created", reg)
    assert not (tmp_path / "not-created").exists()


def simulated_catalog(tmp_path, monkeypatch, fail_page=None):
    """Only fake public metadata responses; no network or experimental arrays."""
    reg = yaml.safe_load(module.REGISTRATION.read_text())
    monkeypatch.chdir(tmp_path)
    Path(reg["previous_specimen_inventory"]).parent.mkdir(parents=True)
    Path(reg["previous_specimen_inventory"]).write_text(yaml.safe_dump({"records": old_records()}))
    registration = tmp_path / "registration.yaml"
    registration.write_text(yaml.safe_dump(reg))
    monkeypatch.setattr(module, "REGISTRATION", registration)
    actual_digest = module.digest
    source_hashes = {str(reg[key]): reg["certificate_authority_sha256" if key == "certificate_authority_bundle" else key + "_sha256"]
                     for key in ("parent_audit", "previous_specimen_inventory", "certificate_authority_bundle")}
    monkeypatch.setattr(module, "digest", lambda path: source_hashes.get(str(path)) or actual_digest(path))
    calls = []

    def client(url, target, registration):
        calls.append(url)
        if url == module.BASE:
            document = {"version": reg["version"], "asset_count": 4435, "size": 4435 * 20, "name": "synthetic"}
        else:
            page = int(url.split("page=")[-1])
            document = page_document(page)
            if page == fail_page:
                document["count"] = 1
        target.write_bytes(json.dumps(document).encode())
        return {"url": url, "path": str(target), "bytes": target.stat().st_size,
                "sha256": actual_digest(target), "client_exit_code": 0}

    monkeypatch.setattr(module, "fetch", client)
    return reg, calls


def test_full_synthetic_catalog_and_exclusive_output(tmp_path, monkeypatch):
    reg, calls = simulated_catalog(tmp_path, monkeypatch)
    with pytest.raises(SystemExit) as completed:
        module.main()
    assert completed.value.code == 0
    root = Path(reg["output_directory"])
    report = yaml.safe_load((root / "manifest.yaml").read_text())
    assert len(calls) == 10
    assert report["summary"]["assets"] == 4435
    assert not report["raw_recordings_downloaded"]
    assert not report["summary"]["biological_eligibility_established"]
    with pytest.raises(FileExistsError):
        module.main()
    assert len(calls) == 10


def test_failed_page_keeps_raw_evidence_and_stops_without_retry(tmp_path, monkeypatch):
    reg, calls = simulated_catalog(tmp_path, monkeypatch, fail_page=2)
    with pytest.raises(SystemExit) as failed:
        module.main()
    assert failed.value.code == 1
    report = yaml.safe_load((Path(reg["output_directory"]) / "manifest.yaml").read_text())
    assert report["status"] == "patchseq-metadata-failure-retained"
    assert len(calls) == len(report["raw_response_evidence"]) == 3
    assert len(report["assets"]) == 500
    assert not report["physiological_arrays_read"]
