"""Synthetic saved-metadata mapping; original evidence is never overwritten."""

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
spec = importlib.util.spec_from_file_location("patchseq_continuation", SCRIPTS / "continue_patchseq_catalog.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def version():
    return {"identifier": "DANDI:000020", "version": "0.210913.1639",
            "assetsSummary": {"numberOfFiles": 4435, "numberOfBytes": 88700, "numberOfSubjects": 1}}


def test_explicit_saved_metadata_mapping_not_server_summary_or_cell_count():
    result = module.map_saved_version(version())
    assert result["asset_count"] == 4435
    assert result["total_asset_bytes"] == 88700
    assert result["archive_subject_count"] == 1
    assert result["subject_count_is_confirmed_cell_count"] is False


@pytest.mark.parametrize("field,value", [("identifier", "DANDI:000021"), ("version", "draft"),
                                       ("assetsSummary", None), ("assetsSummary", [])])
def test_wrong_identity_or_container_rejected(field, value):
    document = version()
    document[field] = value
    with pytest.raises(ValueError):
        module.map_saved_version(document)


@pytest.mark.parametrize("field,value", [("numberOfFiles", 4434), ("numberOfFiles", 4435.0),
                                       ("numberOfBytes", True), ("numberOfBytes", 0),
                                       ("numberOfBytes", 200000000001), ("numberOfSubjects", 0),
                                       ("numberOfSubjects", True), ("numberOfSubjects", 4436)])
def test_typed_counts_and_bounds(field, value):
    document = version()
    document["assetsSummary"][field] = value
    with pytest.raises(ValueError):
        module.map_saved_version(document)


def test_top_level_old_assumptions_do_not_override_registered_mapping():
    document = version()
    document.update(asset_count=1, size=1, cell_class="Vip")
    result = module.map_saved_version(document)
    assert result["asset_count"] == 4435
    assert "cell_class" not in result


@pytest.mark.parametrize("key,value", [("maximum_requests", 10), ("asset_binary_download_authorized", True),
                                       ("version", "draft"), ("saved_version_response_bytes", 0),
                                       ("metadata_acquisition_authorized", 1)])
def test_continuation_identity_and_permissions_fixed(key, value):
    reg = yaml.safe_load(module.REGISTRATION.read_text())
    module.validate_registration(reg)
    reg[key] = value
    with pytest.raises(ValueError):
        module.validate_registration(reg)


def simulated_continuation(tmp_path, monkeypatch, fail_page=None):
    reg = yaml.safe_load(module.REGISTRATION.read_text())
    monkeypatch.chdir(tmp_path)
    saved = Path(reg["saved_version_response"])
    saved.parent.mkdir(parents=True)
    raw = json.dumps(version()).encode()
    saved.write_bytes(raw + b" " * (3192 - len(raw)))
    original = saved.read_bytes()
    old = Path(reg["previous_specimen_inventory"])
    old.parent.mkdir(parents=True, exist_ok=True)
    old.write_text(yaml.safe_dump({"records": [{"specimen_id": i, "included": i <= 111} for i in range(1, 2334)]}))
    parent = Path(reg["parent_manifest"])
    parent.write_text(yaml.safe_dump({"status": "patchseq-metadata-failure-retained", "assets": [],
                                     "requests": [{"url": module.BASE, "sha256": reg["saved_version_response_sha256"], "client_exit_code": 0}],
                                     "raw_recordings_downloaded": False, "physiological_arrays_read": False}))
    registration = tmp_path / "registration.yaml"
    registration.write_text(yaml.safe_dump(reg))
    monkeypatch.setattr(module, "REGISTRATION", registration)
    actual_digest = module.digest
    source_hashes = {reg[key]: reg["certificate_authority_sha256" if key == "certificate_authority_bundle" else key + "_sha256"]
                     for key in ("parent_manifest", "parent_stop", "saved_version_response", "catalog_helper",
                                 "previous_specimen_inventory", "certificate_authority_bundle")}
    monkeypatch.setattr(module, "digest", lambda path: source_hashes.get(str(path)) or actual_digest(path))
    calls = []

    def client(url, target, registration):
        assert url != module.BASE
        calls.append(url)
        page = int(url.split("page=")[-1])
        all_assets = [{"asset_id": str(UUID(int=i+1)), "path": f"sub-10/record-{i:04d}.nwb", "size": 20} for i in range(4435)]
        document = {"count": 4435, "results": all_assets[(page-1)*500:page*500],
                    "next": module.page_url(page+1) if page < 9 else None,
                    "previous": module.page_url(page-1) if page > 1 else None}
        if page == fail_page:
            document["count"] = 1
        target.write_bytes(json.dumps(document).encode())
        return {"url": url, "path": str(target), "bytes": target.stat().st_size,
                "sha256": actual_digest(target), "client_exit_code": 0}

    monkeypatch.setattr(module, "fetch", client)
    return reg, calls, original


def test_saved_version_reused_with_exactly_nine_requests_and_no_overwrite(tmp_path, monkeypatch):
    reg, calls, original = simulated_continuation(tmp_path, monkeypatch)
    with pytest.raises(SystemExit) as completed:
        module.main()
    assert completed.value.code == 0
    assert len(calls) == 9
    assert Path(reg["saved_version_response"]).read_bytes() == original
    report = yaml.safe_load((Path(reg["output_directory"]) / "manifest.yaml").read_text())
    assert report["summary"]["assets"] == 4435
    assert not report["summary"]["biological_eligibility_established"]
    with pytest.raises(FileExistsError):
        module.main()
    assert len(calls) == 9


def test_continuation_stops_on_failed_page_without_requesting_version_or_retry(tmp_path, monkeypatch):
    reg, calls, original = simulated_continuation(tmp_path, monkeypatch, fail_page=2)
    with pytest.raises(SystemExit) as failed:
        module.main()
    assert failed.value.code == 1
    assert len(calls) == 2
    assert Path(reg["saved_version_response"]).read_bytes() == original
    report = yaml.safe_load((Path(reg["output_directory"]) / "manifest.yaml").read_text())
    assert len(report["raw_response_evidence"]) == 2
    assert len(report["assets"]) == 500
