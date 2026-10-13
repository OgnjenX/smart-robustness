"""Explicitly preserve unreported page counts; prove complete catalog separately."""

from __future__ import annotations

import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml
from continue_patchseq_catalog import map_saved_version
from inventory_patchseq_assets import catalog_summary, digest, fetch, page_url, parse_page

REGISTRATION = Path("docs/validation-results/post2008-patchseq-nullable-pagination-registration-1164.yaml")


def project_page(document, page):
    if type(document) is not dict:
        raise ValueError("page object required")
    reported = document.get("count")
    if (page == 1 or reported is not None) and (type(reported) is not int or reported != 4435):
        raise ValueError("reported source count conflicts with fixed version")
    # Supply a separately established scope total only to the legacy shape
    # validator; the raw source observation and its missingness remain explicit.
    shape = dict(document)
    shape["count"] = 4435
    projected = parse_page(shape, page)
    return projected, {"page": page, "count_key_present": "count" in document,
                       "reported_count": reported, "count_is_reported": reported is not None,
                       "scope_total_source": "published-version-and-first-page"}


def validate_registration(reg):
    previous = Path("docs/validation-results/post2008-patchseq-catalog-continuation-registration-1159.yaml")
    if digest(previous) != "28e93a3b34d251a118c042cd74d399ebf0a866824811ef2f2a3858bd10fc7d56":
        raise ValueError("original continuation registration changed")
    original = yaml.safe_load(previous.read_text())
    for key in ("saved_version_response", "saved_version_response_sha256", "previous_specimen_inventory",
                "previous_specimen_inventory_sha256", "catalog_helper", "catalog_helper_sha256",
                "certificate_authority_bundle", "certificate_authority_sha256", "dandiset", "version",
                "expected_assets", "page_size", "maximum_response_bytes", "maximum_seconds_per_request",
                "maximum_total_seconds", "metadata_acquisition_authorized", "asset_binary_download_authorized",
                "experimental_array_reads_authorized", "parameter_fitting_authorized", "cell_or_network_execution_authorized"):
        if type(reg.get(key)) is not type(original[key]) or reg[key] != original[key]:
            raise ValueError("preserved source scope differs: " + key)
    fixed = {"parent_stop": "docs/validation-results/post2008-patchseq-pagination-schema-stop-1163.yaml",
             "parent_stop_sha256": "7b7da715d2d5cfc5f27d3b24e85c7c63c08d2102a5b8abef6280818fd4a0148e",
             "parent_manifest": "results/patchseq-catalog-continuation-1160/manifest.yaml",
             "parent_manifest_sha256": "d1a255ee110189d5f60eeeb8159aa82d7747f2c1530cd4bcabbebefff08b80b3",
             "version_mapper": "scripts/continue_patchseq_catalog.py",
             "version_mapper_sha256": "8c613bb010e23a7071d4ca0b7643f4b6b4275aa7395e4cf1799e0ee905279fd5",
             "maximum_requests": 7, "maximum_aggregate_response_bytes": 14680064,
             "new_page_numbers": [3, 4, 5, 6, 7, 8, 9],
             "output_directory": "results/patchseq-nullable-page-catalog-1165", "saved_metadata_mapping_reads_authorized": True}
    if any(type(reg.get(key)) is not type(value) or reg[key] != value for key, value in fixed.items()):
        raise ValueError("bounded continuation scope differs")
    expected_pages = [
        {"page": 1, "path": "results/patchseq-catalog-continuation-1160/assets-01.json", "sha256": "2f4266712c1b665fe7cd51353d4ec24de94386da08bdf33e36073dec99bd9735"},
        {"page": 2, "path": "results/patchseq-catalog-continuation-1160/assets-02.json", "sha256": "04e3f47b83c6a30fc09e3f329e8c5383712beb845a1685d5887b9e38daa5c75b"},
    ]
    if json.dumps(reg.get("saved_pages"), sort_keys=True) != json.dumps(expected_pages, sort_keys=True):
        raise ValueError("fixed original pages differ")


def main():
    reg = yaml.safe_load(REGISTRATION.read_text())
    validate_registration(reg)
    pinned = {reg[key]: reg["certificate_authority_sha256" if key == "certificate_authority_bundle" else key + "_sha256"]
              for key in ("parent_stop", "parent_manifest", "saved_version_response", "previous_specimen_inventory",
                          "catalog_helper", "version_mapper", "certificate_authority_bundle")}
    pinned.update({row["path"]: row["sha256"] for row in reg["saved_pages"]})
    for path, expected in pinned.items():
        if digest(path) != expected:
            raise ValueError("saved source integrity failed: " + path)
    parent = yaml.safe_load(Path(reg["parent_manifest"]).read_text())
    if (parent.get("status") != "patchseq-catalog-continuation-failure-retained"
        or len(parent.get("requests", [])) != 2 or len(parent.get("assets", [])) != 500
        or parent.get("raw_recordings_downloaded") is not False or parent.get("physiological_arrays_read") is not False):
        raise ValueError("original terminal metadata stop differs")
    root = Path(reg["output_directory"])
    root.mkdir()
    started = time.monotonic()
    report = {"registration_sha256": digest(REGISTRATION), "collector_sha256": digest(__file__),
              "status": "nullable-page-catalog-failure-retained", "new_requests": [], "pages": [], "assets": [],
              "started_at_utc": datetime.now(UTC).isoformat(), "original_stops_modified": False,
              "raw_recordings_downloaded": False, "physiological_arrays_read": False}
    try:
        report["version_projection"] = map_saved_version(json.loads(Path(reg["saved_version_response"]).read_bytes()))
        old = yaml.safe_load(Path(reg["previous_specimen_inventory"]).read_text())["records"]
        for page in range(1, 10):
            if time.monotonic() - started > reg["maximum_total_seconds"]:
                raise ValueError("metadata total time limit exceeded")
            if page <= 2:
                item = reg["saved_pages"][page-1]
                path = Path(item["path"])
            else:
                path = root / f"assets-{page:02d}.json"
                report["new_requests"].append(fetch(page_url(page), path, reg))
                if sum(item["bytes"] for item in report["new_requests"]) > reg["maximum_aggregate_response_bytes"]:
                    raise ValueError("aggregate new response byte limit exceeded")
            projected, evidence = project_page(json.loads(path.read_bytes()), page)
            evidence.update(path=str(path), sha256=digest(path), reused=page <= 2)
            report["pages"].append(evidence)
            report["assets"].extend(projected)
        report["summary"] = catalog_summary(report["assets"], old)
        if report["summary"]["total_asset_bytes"] != report["version_projection"]["total_asset_bytes"]:
            raise ValueError("declared bytes and complete asset-size sum differ")
        if any(digest(path) != expected for path, expected in pinned.items()):
            raise ValueError("original metadata evidence changed during continuation")
        if time.monotonic() - started > reg["maximum_total_seconds"]:
            raise ValueError("metadata total time limit exceeded")
        report["status"] = "patchseq-catalog-complete-independent-verification-pending"
    except (ValueError, OSError, KeyError, TypeError, subprocess.TimeoutExpired) as exc:
        report["error"] = str(exc)
    report.update(finished_at_utc=datetime.now(UTC).isoformat(), elapsed_seconds=time.monotonic() - started)
    report["raw_new_response_evidence"] = [{"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}
                                          for path in sorted(root.glob("*.json"))]
    with (root / "manifest.yaml").open("x") as handle:
        yaml.safe_dump(report, handle, sort_keys=False)
    print(report["status"])
    raise SystemExit(0 if report["status"] == "patchseq-catalog-complete-independent-verification-pending" else 1)


if __name__ == "__main__":
    main()
