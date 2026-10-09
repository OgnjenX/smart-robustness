"""HEAD-only download sizing. Never read response bodies or run native code."""

from __future__ import annotations

import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, build_opener

import yaml
from collect_sst_vip_native_resources import save_yaml, sha
from collect_sst_vip_parameters import RegisteredRedirect

from smart_robustness.validation.sst_vip_physiology_resources import content_length, resource_plan

REGISTRATION = Path(
    "docs/validation-results/post2008-sst-vip-physiology-size-registration-1051.yaml"
)


def head_one(resource, *, reg):
    url = reg["source_base"] + resource["resource"]["download_link"]
    result = {
        "source": resource,
        "request_url": url,
        "request_method": "HEAD",
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "status": "unresolved",
        "body_read": False,
    }
    try:
        with build_opener(RegisteredRedirect(reg["source_base"])).open(
            Request(url, method="HEAD"), timeout=reg["timeout_seconds"]
        ) as response:
            result.update(
                final_url=response.geturl(),
                response_status=response.status,
                response_headers=list(response.headers.items()),
            )
            if response.geturl() != url:
                raise ValueError("resource URL changed")
            result["content_length"] = content_length(response.headers.get("Content-Length"))
            result["status"] = "size-resolved"
    except (OSError, ValueError) as exc:
        result.update(error_type=type(exc).__name__, error=str(exc))
    return result


def main():
    reg = yaml.safe_load(REGISTRATION.read_text())
    for key in ("parent", "physiology_manifest"):
        if sha(Path(reg[key])) != reg[key + "_sha256"]:
            raise ValueError("preflight lineage changed")
    if reg["physiology_body_download_authorized"] is not False:
        raise ValueError("HEAD-only protocol changed")
    source = yaml.safe_load(Path(reg["physiology_manifest"]).read_text())
    plan = resource_plan(source["physiology_records"])
    counts = {
        kind: sum(r["resource"]["well_known_file_type"]["name"] == kind for r in plan["resources"])
        for kind in reg["resource_types"]
    }
    if (
        len(plan["specimen_ids"]) != reg["expected_specimens"]
        or counts["NWBDownload"] != reg["expected_nwb_files"]
        or counts["3DNeuronReconstruction"] != reg["expected_morphology_files"]
        or len(plan["missing_morphology_specimen_ids"])
        != reg["expected_missing_morphology_specimens"]
    ):
        raise ValueError("full registered resource cohort changed")
    directory = Path(reg["output_directory"])
    directory.mkdir()

    def collect(resource):
        result = head_one(resource, reg=reg)
        save_yaml(directory / f"{resource['file_id']}.head.yaml", result)
        return result

    with ThreadPoolExecutor(max_workers=reg["workers"]) as pool:
        records = list(pool.map(collect, plan["resources"]))
    unresolved = sum(r["status"] != "size-resolved" for r in records)
    total = sum(r.get("content_length", 0) for r in records)
    free = shutil.disk_usage(directory).free
    limits = reg["download_admission_for_later_protocol"]
    oversize = [
        r["source"]["file_id"]
        for r in records
        if r.get("content_length", 0)
        > limits[
            "maximum_nwb_bytes"
            if r["source"]["resource"]["well_known_file_type"]["name"] == "NWBDownload"
            else "maximum_morphology_bytes"
        ]
    ]
    eligible = (
        not unresolved
        and not oversize
        and total <= limits["maximum_total_bytes"]
        and free - total >= limits["minimum_free_disk_after_download_bytes"]
    )
    manifest = {
        "schema_version": 1,
        "registration": str(REGISTRATION),
        "registration_sha256": sha(REGISTRATION),
        "collector_sha256": sha(Path(__file__)),
        "plan": plan,
        "records": records,
        "unresolved_resources": unresolved,
        "known_total_bytes": total,
        "free_disk_bytes_at_preflight": free,
        "oversize_resource_ids": oversize,
        "size_admission_passed": bool(eligible),
        "body_download_performed": False,
        "physiology_validated": False,
    }
    save_yaml(directory / "manifest.yaml", manifest)
    print(
        f"{len(records) - unresolved}/{len(records)} sizes resolved; known bytes={total}; admission={eligible}"
    )


if __name__ == "__main__":
    main()
