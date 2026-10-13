"""Preserve original metadata stop and reuse version bytes for asset listing."""

from __future__ import annotations

import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml
from inventory_patchseq_assets import BASE, catalog_summary, digest, fetch, page_url, parse_page

REGISTRATION = Path("docs/validation-results/post2008-patchseq-catalog-continuation-registration-1159.yaml")


def map_saved_version(document):
    if (type(document) is not dict or document.get("identifier") != "DANDI:000020"
        or document.get("version") != "0.210913.1639" or type(document.get("assetsSummary")) is not dict):
        raise ValueError("saved published version identity or summary shape differs")
    summary = document["assetsSummary"]
    count, size, subjects = (summary.get(key) for key in ("numberOfFiles", "numberOfBytes", "numberOfSubjects"))
    if (type(count) is not int or count != 4435 or type(size) is not int or not 0 < size <= 200000000000
        or type(subjects) is not int or not 0 < subjects <= 4435):
        raise ValueError("saved archive count, bytes or subject-count metadata differs")
    return {"identifier": document["identifier"], "version": document["version"],
            "asset_count": count, "total_asset_bytes": size, "archive_subject_count": subjects,
            "subject_count_is_confirmed_cell_count": False}


def validate_registration(reg):
    fixed = {"dandiset": "000020", "version": "0.210913.1639", "expected_assets": 4435,
             "maximum_declared_asset_bytes": 200000000000, "page_size": 500,
             "maximum_asset_pages": 9, "maximum_requests": 9, "maximum_response_bytes": 2097152,
             "maximum_aggregate_response_bytes": 18874368, "maximum_seconds_per_request": 60,
             "maximum_total_seconds": 1200, "output_directory": "results/patchseq-catalog-continuation-1160",
             "metadata_acquisition_authorized": True, "saved_version_mapping_reads_authorized": True,
             "asset_binary_download_authorized": False, "experimental_array_reads_authorized": False,
             "parameter_fitting_authorized": False, "cell_or_network_execution_authorized": False,
             "parent_manifest": "results/patchseq-asset-metadata-1156/manifest.yaml",
             "parent_manifest_sha256": "a98bee0008d7f7de88fbfbd16c7363667a2189df9f739de866bf5eac61271da3",
             "saved_version_response": "results/patchseq-asset-metadata-1156/version.json",
             "saved_version_response_sha256": "bf637119b3e254e70ca2847cf138d04a001e8c9fa5d8c27edcbf2cd7da8b884f",
             "saved_version_response_bytes": 3192,
             "parent_stop": "docs/validation-results/post2008-patchseq-metadata-schema-stop-1158.yaml",
             "parent_stop_sha256": "a8f83430c626879a2e783c1d220f337847506790e069927aee0ce21890525cb0",
             "catalog_helper": "scripts/inventory_patchseq_assets.py",
             "catalog_helper_sha256": "5e6e3f27280b91f0c1294b973e51a09aaba620f57552aced0048af6c59263dd9",
             "previous_specimen_inventory": "results/sst-vip-cell-source-inventory-1028.yaml",
             "previous_specimen_inventory_sha256": "f6d4ef1e801b9dfc101b7e037b2f63f0d27ee2168b86ffd734976b5c9dd812ad",
             "certificate_authority_bundle": "/etc/ssl/cert.pem",
             "certificate_authority_sha256": "9dae8d76e55cb08991f2b672d58999ea15560d910759c16b544f843bdffbb994"}
    if any(type(reg.get(key)) is not type(value) or reg[key] != value for key, value in fixed.items()):
        raise ValueError("fixed continuation identity, bounds or permissions differ")


def main():
    reg = yaml.safe_load(REGISTRATION.read_text())
    validate_registration(reg)
    sources = ("parent_manifest", "parent_stop", "saved_version_response", "catalog_helper",
               "previous_specimen_inventory", "certificate_authority_bundle")
    for key in sources:
        field = "certificate_authority_sha256" if key == "certificate_authority_bundle" else key + "_sha256"
        if digest(reg[key]) != reg[field]:
            raise ValueError("continuation source changed: " + key)
    parent = yaml.safe_load(Path(reg["parent_manifest"]).read_text())
    if (parent.get("status") != "patchseq-metadata-failure-retained" or parent.get("assets") != []
        or len(parent.get("requests", [])) != 1 or parent["requests"][0]["url"] != BASE
        or parent["requests"][0]["sha256"] != reg["saved_version_response_sha256"]
        or parent["requests"][0]["client_exit_code"] != 0
        or parent.get("raw_recordings_downloaded") is not False or parent.get("physiological_arrays_read") is not False):
        raise ValueError("original stopped attempt differs")
    saved = Path(reg["saved_version_response"])
    if saved.stat().st_size != reg["saved_version_response_bytes"]:
        raise ValueError("saved version response length differs")
    old = yaml.safe_load(Path(reg["previous_specimen_inventory"]).read_text())["records"]
    root = Path(reg["output_directory"])
    root.mkdir()
    started = time.monotonic()
    report = {"registration_sha256": digest(REGISTRATION), "collector_sha256": digest(__file__),
              "catalog_helper_sha256": reg["catalog_helper_sha256"], "requests": [], "assets": [],
              "status": "patchseq-catalog-continuation-failure-retained", "saved_version_response": str(saved),
              "saved_version_response_sha256": reg["saved_version_response_sha256"],
              "started_at_utc": datetime.now(UTC).isoformat(), "original_stop_modified": False,
              "raw_recordings_downloaded": False, "physiological_arrays_read": False}
    try:
        report["version_projection"] = map_saved_version(json.loads(saved.read_bytes()))
        for page in range(1, 10):
            if time.monotonic() - started > reg["maximum_total_seconds"]:
                raise ValueError("total metadata time bound exceeded")
            path = root / f"assets-{page:02d}.json"
            report["requests"].append(fetch(page_url(page), path, reg))
            if sum(item["bytes"] for item in report["requests"]) > reg["maximum_aggregate_response_bytes"]:
                raise ValueError("aggregate metadata byte bound exceeded")
            report["assets"].extend(parse_page(json.loads(path.read_bytes()), page))
        report["summary"] = catalog_summary(report["assets"], old)
        if report["summary"]["total_asset_bytes"] != report["version_projection"]["total_asset_bytes"]:
            raise ValueError("declared asset bytes and catalog sum differ")
        for key in sources:
            field = "certificate_authority_sha256" if key == "certificate_authority_bundle" else key + "_sha256"
            if digest(reg[key]) != reg[field]:
                raise ValueError("source changed during metadata continuation")
        if time.monotonic() - started > reg["maximum_total_seconds"]:
            raise ValueError("total metadata time bound exceeded")
        report["status"] = "patchseq-catalog-acquired-independent-verification-pending"
    except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as exc:
        report["error"] = str(exc)
    report.update(finished_at_utc=datetime.now(UTC).isoformat(), elapsed_seconds=time.monotonic() - started)
    report["raw_response_evidence"] = [{"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}
                                       for path in sorted(root.glob("*.json"))]
    with (root / "manifest.yaml").open("x") as handle:
        yaml.safe_dump(report, handle, sort_keys=False)
    print(report["status"])
    raise SystemExit(0 if report["status"] == "patchseq-catalog-acquired-independent-verification-pending" else 1)


if __name__ == "__main__":
    main()
