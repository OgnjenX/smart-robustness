"""Unreported page counts stay unreported; final catalog proof is unchanged."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from uuid import UUID

import pytest
import yaml

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("nullable_catalog", SCRIPTS / "complete_patchseq_catalog.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def page_document(page):
    all_assets = [{"asset_id": str(UUID(int=index + 1)), "path": f"sub-10/record-{index:04d}.nwb", "size": 20}
                  for index in range(4435)]
    return {"count": 4435, "results": all_assets[(page-1)*500:page*500],
            "next": module.page_url(page+1) if page < 9 else None,
            "previous": module.page_url(page-1) if page > 1 else None}


@pytest.mark.parametrize("missing", ["absent", "null"])
def test_later_count_missingness_preserved_without_fabricated_observation(missing):
    document = page_document(2)
    if missing == "absent":
        document.pop("count")
    else:
        document["count"] = None
    before = json.dumps(document)
    projected, evidence = module.project_page(document, 2)
    assert len(projected) == 500
    assert evidence["reported_count"] is None
    assert not evidence["count_is_reported"]
    assert evidence["count_key_present"] is (missing == "null")
    assert json.dumps(document) == before


@pytest.mark.parametrize("value", [None, 4434, True, 4435.0, "4435"])
def test_first_count_is_still_explicit_exact_integer(value):
    document = page_document(1)
    document["count"] = value
    with pytest.raises(ValueError):
        module.project_page(document, 1)


@pytest.mark.parametrize("value", [4434, True, 4435.0, "4435"])
def test_reported_later_conflicting_count_never_relaxed(value):
    document = page_document(2)
    document["count"] = value
    with pytest.raises(ValueError):
        module.project_page(document, 2)


@pytest.mark.parametrize("fault", ["short", "wrong-next", "wrong-UUID"])
def test_nullable_count_does_not_bypass_other_page_gates(fault):
    document = page_document(2)
    document.pop("count")
    if fault == "short":
        document["results"].pop()
    elif fault == "wrong-next":
        document["next"] = module.page_url(4)
    else:
        document["results"][0]["asset_id"] = "not-a-UUID"
    with pytest.raises(ValueError):
        module.project_page(document, 2)


@pytest.mark.parametrize("key,value", [("maximum_requests", 9), ("parameter_fitting_authorized", True),
                                       ("version", "draft"), ("new_page_numbers", [1, 2, 3]),
                                       ("saved_metadata_mapping_reads_authorized", 1)])
def test_fixed_seven_request_registration_and_scope(key, value):
    reg = yaml.safe_load(module.REGISTRATION.read_text())
    module.validate_registration(reg)
    reg[key] = value
    with pytest.raises(ValueError):
        module.validate_registration(reg)


def simulation(tmp_path, monkeypatch, fault=None):
    reg = yaml.safe_load(module.REGISTRATION.read_text())
    monkeypatch.chdir(tmp_path)
    saved = Path(reg["saved_version_response"])
    saved.parent.mkdir(parents=True)
    saved.write_text(json.dumps({"identifier": "DANDI:000020", "version": "0.210913.1639",
                                 "assetsSummary": {"numberOfFiles": 4435, "numberOfBytes": 88700, "numberOfSubjects": 1}}))
    for item in reg["saved_pages"]:
        path = Path(item["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        document = page_document(item["page"])
        if item["page"] > 1:
            document.pop("count")
        path.write_text(json.dumps(document))
    parent = Path(reg["parent_manifest"])
    parent.write_text(yaml.safe_dump({"status": "patchseq-catalog-continuation-failure-retained", "requests": [{}, {}],
                                     "assets": [{}]*500, "raw_recordings_downloaded": False, "physiological_arrays_read": False}))
    old = Path(reg["previous_specimen_inventory"])
    old.parent.mkdir(parents=True, exist_ok=True)
    old.write_text(yaml.safe_dump({"records": [{"specimen_id": identity, "included": identity <= 111} for identity in range(1, 2334)]}))
    registration = tmp_path / "registration.yaml"
    registration.write_text(yaml.safe_dump(reg))
    monkeypatch.setattr(module, "REGISTRATION", registration)
    monkeypatch.setattr(module, "validate_registration", lambda _: None)
    pinned = {reg[key]: reg["certificate_authority_sha256" if key == "certificate_authority_bundle" else key + "_sha256"]
              for key in ("parent_stop", "parent_manifest", "saved_version_response", "previous_specimen_inventory",
                          "catalog_helper", "version_mapper", "certificate_authority_bundle")}
    pinned.update({item["path"]: item["sha256"] for item in reg["saved_pages"]})
    original = {path: Path(path).read_bytes() for path in pinned if Path(path).is_file()}
    actual_digest = module.digest

    def checksum(path):
        text_path = str(path)
        if text_path in pinned:
            if text_path in original and Path(path).read_bytes() != original[text_path]:
                return actual_digest(path)
            return pinned[text_path]
        return actual_digest(path)

    monkeypatch.setattr(module, "digest", checksum)
    calls = []

    def client(url, target, registration):
        page = int(url.split("page=")[-1])
        assert 3 <= page <= 9
        calls.append(page)
        document = page_document(page)
        document.pop("count")
        if fault == "count" and page == 3:
            document["count"] = 1
        if fault == "duplicate" and page == 3:
            document["results"][0]["asset_id"] = str(UUID(int=1))
        if fault == "changed-prefix":
            saved.write_bytes(saved.read_bytes() + b" ")
        target.write_text(json.dumps(document))
        return {"url": url, "path": str(target), "bytes": target.stat().st_size,
                "sha256": actual_digest(target), "client_exit_code": 0}

    monkeypatch.setattr(module, "fetch", client)
    return reg, calls, original


def test_complete_nullable_catalog_reuses_pages_and_rejects_duplicate_invocation(tmp_path, monkeypatch):
    reg, calls, original = simulation(tmp_path, monkeypatch)
    with pytest.raises(SystemExit) as completed:
        module.main()
    assert completed.value.code == 0
    assert calls == [3, 4, 5, 6, 7, 8, 9]
    report = yaml.safe_load((Path(reg["output_directory"]) / "manifest.yaml").read_text())
    assert report["summary"]["assets"] == 4435
    assert [page["count_is_reported"] for page in report["pages"]] == [True] + [False]*8
    assert all(Path(path).read_bytes() == raw for path, raw in original.items())
    with pytest.raises(FileExistsError):
        module.main()
    assert len(calls) == 7


@pytest.mark.parametrize("fault", ["count", "duplicate", "changed-prefix"])
def test_conflict_or_source_change_retained_without_catalog_promotion(tmp_path, monkeypatch, fault):
    reg, calls, _ = simulation(tmp_path, monkeypatch, fault)
    with pytest.raises(SystemExit) as failed:
        module.main()
    assert failed.value.code == 1
    report = yaml.safe_load((Path(reg["output_directory"]) / "manifest.yaml").read_text())
    assert report["status"] == "nullable-page-catalog-failure-retained"
    assert calls == ([3] if fault == "count" else [3, 4, 5, 6, 7, 8, 9])
    assert not report["raw_recordings_downloaded"]
