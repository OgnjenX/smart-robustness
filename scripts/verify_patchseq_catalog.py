"""Independently reconstruct the public catalog; no collector or network imports."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from uuid import UUID

import yaml

REGISTRATION = Path("docs/validation-results/post2008-patchseq-nullable-pagination-registration-1164.yaml")
MANIFEST = Path("results/patchseq-nullable-page-catalog-1165/manifest.yaml")
OUTPUT = Path("results/patchseq-catalog-independent-1167")
BASE = "https://api.dandiarchive.org/api/dandisets/000020/versions/0.210913.1639/"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("ambiguous duplicate JSON key")
        result[key] = value
    return result


def read_json(path):
    if path.stat().st_size > 2097152:
        raise ValueError("response exceeds independent byte bound")
    return json.loads(path.read_bytes(), object_pairs_hook=unique_object)


def check_url(url, page):
    if page is None:
        if url is not None:
            raise ValueError("unexpected terminal pagination URL")
        return
    if type(url) is not str:
        raise ValueError("required page URL absent")
    actual = urlsplit(url)
    query = parse_qs(actual.query, keep_blank_values=True)
    if page == 1 and "page" not in query:
        query["page"] = ["1"]
    if (actual.scheme != "https" or actual.netloc != "api.dandiarchive.org" or actual.fragment
        or actual.path != "/api/dandisets/000020/versions/0.210913.1639/assets/"
        or query != {"metadata": ["false"], "order": ["path"], "page_size": ["500"], "page": [str(page)]}):
        raise ValueError("page URL leaves fixed version or scope")


def reconstruct(version, pages, old_records):
    if (type(version) is not dict or version.get("identifier") != "DANDI:000020"
        or version.get("version") != "0.210913.1639" or type(version.get("assetsSummary")) is not dict):
        raise ValueError("published version identity differs")
    declared = version["assetsSummary"]
    if (type(declared.get("numberOfFiles")) is not int or declared["numberOfFiles"] != 4435
        or type(declared.get("numberOfBytes")) is not int or not 0 < declared["numberOfBytes"] <= 200000000000
        or type(declared.get("numberOfSubjects")) is not int or not 0 < declared["numberOfSubjects"] <= 4435
        or len(pages) != 9):
        raise ValueError("published catalog count or byte scope differs")
    assets, count_evidence, ids, paths = [], [], set(), set()
    for number, document in enumerate(pages, start=1):
        if type(document) is not dict:
            raise ValueError("page object required")
        count = document.get("count")
        if (number == 1 or count is not None) and (type(count) is not int or count != 4435):
            raise ValueError("reported count conflicts with fixed version")
        rows = document.get("results")
        if type(rows) is not list or len(rows) != (500 if number < 9 else 435):
            raise ValueError("incomplete page extent")
        check_url(document.get("previous"), number-1 if number > 1 else None)
        check_url(document.get("next"), number+1 if number < 9 else None)
        count_evidence.append({"page": number, "count_key_present": "count" in document,
                               "reported_count": count, "count_is_reported": count is not None,
                               "scope_total_source": "published-version-and-first-page"})
        for row in rows:
            if type(row) is not dict or type(row.get("asset_id")) is not str:
                raise ValueError("asset UUID missing")
            identity, path, size = row["asset_id"], row.get("path"), row.get("size")
            if (str(UUID(identity)) != identity or identity in ids or type(path) is not str or not path
                or len(path.encode()) > 4096 or path.startswith("/") or "\\" in path
                or any(part in ("", ".", "..") for part in path.split("/")) or path in paths
                or type(size) is not int or size < 0):
                raise ValueError("asset identity, path or typed size differs")
            ids.add(identity)
            paths.add(path)
            assets.append({"asset_id": identity, "path": path, "size": size})
    if len(assets) != 4435 or [row["path"] for row in assets] != sorted(paths):
        raise ValueError("complete unique path-ordered catalog required")
    total = sum(row["size"] for row in assets)
    if total != declared["numberOfBytes"]:
        raise ValueError("complete byte sum disagrees with published declaration")
    old, included = set(), set()
    for row in old_records:
        value = row["specimen_id"]
        if type(value) is not int or value <= 0 or value in old:
            raise ValueError("old specimen identity differs")
        old.add(value)
        if row["included"] is True:
            included.add(value)
    if len(old) != 2333 or len(included) != 111:
        raise ValueError("prior source cohort differs")
    subjects, unresolved = set(), []
    for row in assets:
        head = row["path"].split("/")[0]
        suffix = head[4:] if head.startswith("sub-") else ""
        if suffix and suffix.isascii() and suffix.isdigit() and suffix[0] != "0":
            subjects.add(int(suffix))
        else:
            unresolved.append(row["asset_id"])
    return {"assets": assets, "count_evidence": count_evidence,
            "version_projection": {"identifier": version["identifier"], "version": version["version"],
                                   "asset_count": 4435, "total_asset_bytes": total,
                                   "archive_subject_count": declared["numberOfSubjects"],
                                   "subject_count_is_confirmed_cell_count": False},
            "summary": {"assets": 4435, "total_asset_bytes": total, "numeric_path_subject_hints": sorted(subjects),
                        "unresolved_path_asset_ids": unresolved, "potential_prior_specimen_ID_matches": sorted(subjects & old),
                        "potential_prior_included_specimen_ID_matches": sorted(subjects & included),
                        "specimen_identity_confirmed": False, "VIP_or_SST_identity_confirmed": False,
                        "biological_eligibility_established": False}}


def typed_equal(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def snapshot(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ino


def main():
    if not MANIFEST.is_file():
        raise ValueError("terminal catalog evidence absent; never restart acquisition")
    OUTPUT.mkdir()
    report = {"verifier_sha256": digest(__file__), "acquisition_manifest_sha256": digest(MANIFEST),
              "collector_imported": False, "network_requests": 0, "physiological_arrays_read": False,
              "status": "catalog-verification-failure-retained", "started_at_utc": datetime.now(UTC).isoformat()}
    try:
        reg = yaml.safe_load(REGISTRATION.read_text())
        result = yaml.safe_load(MANIFEST.read_text())
        if (digest(REGISTRATION) != "ad6858cc534f80a0c3308ed8c2ca5c08fb006d9e4a2d628fdcda7499893b4f78"
            or result["registration_sha256"] != digest(REGISTRATION)
            or result["collector_sha256"] != digest(Path(__file__).with_name("complete_patchseq_catalog.py"))
            or result.get("status") != "patchseq-catalog-complete-independent-verification-pending"
            or result.get("raw_recordings_downloaded") is not False or result.get("physiological_arrays_read") is not False
            or result.get("original_stops_modified") is not False or len(result.get("new_requests", [])) != 7
            or len(result.get("pages", [])) != 9 or type(result.get("elapsed_seconds")) not in (int, float)
            or not 0 <= result["elapsed_seconds"] <= 1200):
            raise ValueError("source integrity or terminal acquisition scope differs")
        pinned = {Path(reg[key]): reg["certificate_authority_sha256" if key == "certificate_authority_bundle" else key + "_sha256"]
                  for key in ("parent_stop", "parent_manifest", "saved_version_response", "previous_specimen_inventory",
                              "catalog_helper", "version_mapper", "certificate_authority_bundle")}
        pinned.update({Path(item["path"]): item["sha256"] for item in reg["saved_pages"]})
        root = Path(reg["output_directory"])
        for page in range(3, 10):
            request = result["new_requests"][page-3]
            path = root / f"assets-{page:02d}.json"
            check_url(request["url"], page)
            if (Path(request["path"]) != path or type(request["client_exit_code"]) is not int or request["client_exit_code"] != 0
                or type(request["bytes"]) is not int or request["bytes"] != path.stat().st_size):
                raise ValueError("new request ledger differs")
            pinned[path] = request["sha256"]
        if sum(item["bytes"] for item in result["new_requests"]) > 14680064:
            raise ValueError("aggregate new response bound exceeded")
        pinned[REGISTRATION] = digest(REGISTRATION)
        pinned[MANIFEST] = digest(MANIFEST)
        pinned[Path(__file__).with_name("complete_patchseq_catalog.py")] = result["collector_sha256"]
        raw_evidence = [{"path": str(root / f"assets-{page:02d}.json"),
                         "bytes": (root / f"assets-{page:02d}.json").stat().st_size,
                         "sha256": digest(root / f"assets-{page:02d}.json")} for page in range(3, 10)]
        if not typed_equal(result["raw_new_response_evidence"], raw_evidence):
            raise ValueError("raw response inventory differs")
        before = {path: snapshot(path) for path in pinned}
        for path, checksum in pinned.items():
            if digest(path) != checksum:
                raise ValueError("pinned source hash differs: " + str(path))
        documents = []
        for number, entry in enumerate(result["pages"], start=1):
            path = Path(reg["saved_pages"][number-1]["path"]) if number <= 2 else root / f"assets-{number:02d}.json"
            if (entry["page"] != number or Path(entry["path"]) != path or entry["sha256"] != digest(path)
                or entry["reused"] is not (number <= 2)):
                raise ValueError("saved or new page ledger differs")
            documents.append(read_json(path))
        old = yaml.safe_load(Path(reg["previous_specimen_inventory"]).read_text())["records"]
        rebuilt = reconstruct(read_json(Path(reg["saved_version_response"])), documents, old)
        for key in ("assets", "summary", "version_projection"):
            if not typed_equal(result[key], rebuilt[key]):
                raise ValueError("independent projection differs: " + key)
        for actual, expected in zip(result["pages"], rebuilt["count_evidence"], strict=True):
            if not typed_equal({key: actual[key] for key in expected}, expected):
                raise ValueError("page-count missingness reconstruction differs")
        if any(snapshot(path) != state for path, state in before.items()):
            raise ValueError("source evidence changed during verification")
        report.update(rebuilt, status="patchseq-catalog-independently-verified")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report["error"] = str(exc)
    report["finished_at_utc"] = datetime.now(UTC).isoformat()
    with (OUTPUT / "manifest.yaml").open("x") as handle:
        yaml.safe_dump(report, handle, sort_keys=False)
    print(report["status"])
    raise SystemExit(0 if report["status"] == "patchseq-catalog-independently-verified" else 1)


if __name__ == "__main__":
    main()
