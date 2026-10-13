"""Bounded published DANDI asset catalog only; no binary or physiological reads."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlsplit
from uuid import UUID

import yaml

REGISTRATION = Path("docs/validation-results/post2008-patchseq-asset-metadata-registration-1155.yaml")
BASE = "https://api.dandiarchive.org/api/dandisets/000020/versions/0.210913.1639/"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def page_url(page):
    if type(page) is not int or not 1 <= page <= 9:
        raise ValueError("fixed page bound exceeded")
    return BASE + "assets/?" + urlencode({"metadata": "false", "order": "path", "page_size": 500, "page": page})


def check_link(link, page):
    if page is None:
        if link is not None:
            raise ValueError("terminal pagination link differs")
        return
    if type(link) is not str:
        raise ValueError("pagination URL missing")
    url = urlsplit(link)
    expected = urlsplit(page_url(page))
    query = parse_qs(url.query, keep_blank_values=True)
    if page == 1 and "page" not in query:
        query["page"] = ["1"]
    if (url.scheme != expected.scheme or url.netloc != expected.netloc or url.path != expected.path
        or url.fragment or query != parse_qs(expected.query)):
        raise ValueError("pagination URL leaves fixed version or query scope")


def parse_page(document, page):
    if type(document) is not dict or type(document.get("count")) is not int or document["count"] != 4435:
        raise ValueError("fixed published asset count differs")
    rows = document.get("results")
    expected_count = 500 if page < 9 else 435
    if type(rows) is not list or len(rows) != expected_count:
        raise ValueError("complete page extent differs")
    check_link(document.get("next"), page + 1 if page < 9 else None)
    check_link(document.get("previous"), page - 1 if page > 1 else None)
    projected = []
    for row in rows:
        if type(row) is not dict or type(row.get("asset_id")) is not str:
            raise ValueError("asset identifier missing")
        if str(UUID(row["asset_id"])) != row["asset_id"]:
            raise ValueError("canonical asset UUID required")
        path, size = row.get("path"), row.get("size")
        if (type(path) is not str or not path or len(path.encode()) > 4096
            or path.startswith("/") or "\\" in path or any(part in ("", ".", "..") for part in path.split("/"))
            or type(size) is not int or size < 0):
            raise ValueError("bounded relative asset path or byte size differs")
        projected.append({"asset_id": row["asset_id"], "path": path, "size": size})
    return projected


def catalog_summary(rows, old_records):
    if len(rows) != 4435 or len({row["asset_id"] for row in rows}) != len(rows) or len({row["path"] for row in rows}) != len(rows):
        raise ValueError("complete unique published catalog required")
    paths = [row["path"] for row in rows]
    if paths != sorted(paths):
        raise ValueError("registered path order differs")
    old = {row["specimen_id"] for row in old_records}
    included = {row["specimen_id"] for row in old_records if row["included"] is True}
    if len(old) != 2333 or len(included) != 111:
        raise ValueError("prior source inventory identity or inclusion count differs")
    subjects, unresolved = set(), []
    for row in rows:
        match = re.fullmatch(r"sub-([1-9][0-9]*)", row["path"].split("/")[0])
        if match:
            subjects.add(int(match.group(1)))
        else:
            unresolved.append(row["asset_id"])
    return {"assets": len(rows), "total_asset_bytes": sum(row["size"] for row in rows),
            "numeric_path_subject_hints": sorted(subjects), "unresolved_path_asset_ids": unresolved,
            "potential_prior_specimen_ID_matches": sorted(subjects & old),
            "potential_prior_included_specimen_ID_matches": sorted(subjects & included),
            "specimen_identity_confirmed": False, "VIP_or_SST_identity_confirmed": False,
            "biological_eligibility_established": False}


def fetch(url, target, reg):
    if url != BASE and url not in [page_url(page) for page in range(1, 10)]:
        raise ValueError("unregistered request URL")
    stderr = target.with_suffix(".stderr.txt")
    args = ["/usr/bin/curl", "--disable", "--silent", "--show-error", "--fail", "--proto", "=https",
            "--max-time", str(reg["maximum_seconds_per_request"]), "--max-filesize", str(reg["maximum_response_bytes"]),
            "--cacert", reg["certificate_authority_bundle"], "--header", "Accept: application/json", url]
    count, h = 0, hashlib.sha256()
    with target.open("xb") as output, stderr.open("xb") as error, subprocess.Popen(args, stdout=subprocess.PIPE, stderr=error) as proc:
        try:
            while count <= reg["maximum_response_bytes"]:
                block = proc.stdout.read(min(65536, reg["maximum_response_bytes"] + 1 - count))
                if not block:
                    break
                output.write(block)
                count += len(block)
                h.update(block)
            if count > reg["maximum_response_bytes"]:
                proc.kill()
            code = proc.wait(timeout=reg["maximum_seconds_per_request"] + 5)
        except BaseException:
            proc.kill()
            proc.wait()
            raise
    if code != 0 or count > reg["maximum_response_bytes"]:
        raise ValueError("metadata transport exit or byte bound failed")
    return {"url": url, "path": str(target), "bytes": count, "sha256": h.hexdigest(), "client_exit_code": code}


def validate_registration(reg):
    fixed = {"dandiset": "000020", "version": "0.210913.1639", "version_url": BASE,
             "parent_audit": "docs/validation-results/post2008-inhibitory-alternative-primary-source-audit-1154.yaml",
             "parent_audit_sha256": "71a25804a69345b223f64208d374ad8dfd69658c0c5e4b002e1e9b0ff02898f2",
             "previous_specimen_inventory": "results/sst-vip-cell-source-inventory-1028.yaml",
             "previous_specimen_inventory_sha256": "f6d4ef1e801b9dfc101b7e037b2f63f0d27ee2168b86ffd734976b5c9dd812ad",
             "asset_list_url": BASE + "assets/", "expected_assets": 4435, "page_size": 500,
             "maximum_asset_pages": 9, "maximum_requests": 10, "maximum_response_bytes": 2097152,
             "maximum_aggregate_response_bytes": 20971520, "maximum_seconds_per_request": 60,
             "maximum_total_seconds": 1200, "output_directory": "results/patchseq-asset-metadata-1156",
             "metadata_acquisition_authorized": True, "asset_binary_download_authorized": False,
             "experimental_array_reads_authorized": False, "parameter_fitting_authorized": False,
             "cell_or_network_execution_authorized": False,
             "certificate_authority_bundle": "/etc/ssl/cert.pem",
             "certificate_authority_sha256": "9dae8d76e55cb08991f2b672d58999ea15560d910759c16b544f843bdffbb994"}
    if any(type(reg.get(key)) is not type(value) or reg[key] != value for key, value in fixed.items()):
        raise ValueError("registered metadata scope or bound differs")


def main():
    reg = yaml.safe_load(REGISTRATION.read_text())
    validate_registration(reg)
    for key in ("parent_audit", "previous_specimen_inventory", "certificate_authority_bundle"):
        field = "certificate_authority_sha256" if key == "certificate_authority_bundle" else key + "_sha256"
        if digest(reg[key]) != reg[field]:
            raise ValueError("metadata source lineage changed")
    old = yaml.safe_load(Path(reg["previous_specimen_inventory"]).read_text())["records"]
    root = Path(reg["output_directory"])
    root.mkdir()
    report = {"registration_sha256": digest(REGISTRATION), "collector_sha256": digest(__file__),
              "status": "patchseq-metadata-failure-retained", "requests": [], "assets": [],
              "started_at_utc": datetime.now(UTC).isoformat(), "raw_recordings_downloaded": False,
              "physiological_arrays_read": False, "parameter_fitting_performed": False}
    started = time.monotonic()
    try:
        response = fetch(BASE, root / "version.json", reg)
        report["requests"].append(response)
        version = json.loads((root / "version.json").read_bytes())
        if (type(version) is not dict or version.get("version") != reg["version"] or type(version.get("asset_count")) is not int
            or version["asset_count"] != reg["expected_assets"]):
            raise ValueError("published version identity or asset count differs")
        report["version_projection"] = {key: version.get(key) for key in ("version", "asset_count", "size", "name")}
        for page in range(1, 10):
            if time.monotonic() - started > reg["maximum_total_seconds"]:
                raise ValueError("total metadata time bound exceeded")
            path = root / f"assets-{page:02d}.json"
            report["requests"].append(fetch(page_url(page), path, reg))
            if sum(item["bytes"] for item in report["requests"]) > reg["maximum_aggregate_response_bytes"]:
                raise ValueError("aggregate response byte bound exceeded")
            report["assets"].extend(parse_page(json.loads(path.read_bytes()), page))
        report["summary"] = catalog_summary(report["assets"], old)
        size = report["version_projection"]["size"]
        if type(size) is not int or size != report["summary"]["total_asset_bytes"]:
            raise ValueError("independent declared size and asset sum differ")
        if time.monotonic() - started > reg["maximum_total_seconds"]:
            raise ValueError("total metadata time bound exceeded")
        report["status"] = "patchseq-catalog-acquired-independent-verification-pending"
    except (ValueError, OSError, KeyError, TypeError, subprocess.TimeoutExpired) as exc:
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
