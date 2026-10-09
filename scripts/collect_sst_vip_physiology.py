"""Checkpointed full-cohort acquisition; no native execution or data mutation."""

from __future__ import annotations

import fcntl
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, build_opener

import yaml
from collect_sst_vip_native_resources import save_yaml, sha
from collect_sst_vip_parameters import RegisteredRedirect

from smart_robustness.validation.sst_vip_physiology_download import hash_file, receive_file
from smart_robustness.validation.sst_vip_physiology_resources import content_length

REGISTRATION = Path(
    "docs/validation-results/post2008-sst-vip-physiology-acquisition-registration-1054.yaml"
)


def receipt_path(head, directory):
    kind = head["source"]["resource"]["well_known_file_type"]["name"]
    extension = {"NWBDownload": ".nwb", "3DNeuronReconstruction": ".swc"}[kind]
    return directory / (str(head["source"]["file_id"]) + extension)


def check_saved(saved, *, head, context, path):
    if saved["preflight_record"] != head or saved["context"] != context:
        raise ValueError("checkpoint source or implementation changed")
    if "raw_path" in saved:
        if saved["raw_path"] != str(path) or not path.is_file():
            raise ValueError("checkpoint receipt identity changed")
        if path.stat().st_size != saved["raw_bytes"] or hash_file(path) != saved["raw_sha256"]:
            raise ValueError("checkpoint receipt changed")
        if (
            saved["status"] == "complete-receipt-awaiting-format-validation"
            and saved["raw_bytes"] != head["content_length"]
        ):
            raise ValueError("checkpoint complete length changed")
    elif path.exists():
        raise ValueError("orphan receipt not represented by checkpoint")


def collect_one(head, *, reg, directory, context):
    path = receipt_path(head, directory)
    sidecar = directory / f"{head['source']['file_id']}.provenance.yaml"
    if sidecar.exists():
        saved = yaml.safe_load(sidecar.read_text())
        check_saved(saved, head=head, context=context, path=path)
        return saved
    if path.exists():
        raise ValueError("orphan receipt requires assessment, not overwrite")
    result = {
        "preflight_record": head,
        "context": context,
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "status": "failed",
    }
    expected = head["content_length"]
    kind = head["source"]["resource"]["well_known_file_type"]["name"]
    maximum = reg["maximum_nwb_bytes" if kind == "NWBDownload" else "maximum_morphology_bytes"]
    try:
        if head["status"] != "size-resolved" or not 0 < expected <= maximum:
            raise ValueError("preflight receipt bounds invalid")
        if shutil.disk_usage(directory).free - expected < reg["minimum_free_disk_bytes"]:
            raise ValueError("registered disk reserve would be violated")
        url = reg["source_base"] + head["source"]["resource"]["download_link"]
        if url != head["request_url"]:
            raise ValueError("preflight source URL changed")
        request = Request(url, headers={"Accept-Encoding": "identity"})
        with build_opener(RegisteredRedirect(reg["source_base"])).open(
            request, timeout=reg["timeout_seconds"]
        ) as response:
            result.update(
                final_url=response.geturl(),
                response_status=response.status,
                response_headers=[[key, value] for key, value in response.headers.items()],
            )
            if response.status != 200 or response.geturl() != url:
                raise ValueError("GET status or source URL changed")
            if content_length(response.headers.get("Content-Length")) != expected:
                raise ValueError("GET content length differs from sealed HEAD")
            if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                raise ValueError("unexpected encoded response")
            headers = {k.lower(): v for k, v in head["response_headers"]}
            for key in ("etag", "last-modified"):
                if key in headers and response.headers.get(key) != headers[key]:
                    raise ValueError("GET resource version differs from sealed HEAD")
            result.update(
                receive_file(
                    response,
                    path,
                    expected_bytes=expected,
                    maximum_bytes=maximum,
                    chunk_bytes=reg["chunk_bytes"],
                )
            )
    except (OSError, ValueError) as exc:
        result.update(error_type=type(exc).__name__, error=str(exc))
    save_yaml(sidecar, result)
    return result


def main():
    reg = yaml.safe_load(REGISTRATION.read_text())
    for key in ("parent", "preflight"):
        if sha(Path(reg[key])) != reg[key + "_sha256"]:
            raise ValueError("registered acquisition lineage changed")
    if reg["physiology_body_download_authorized"] is not True or any(
        reg[k] is not False
        for k in (
            "network_execution_authorized",
            "isolated_simulation_authorized",
            "parameter_fitting_authorized",
        )
    ):
        raise ValueError("acquisition-only authorization changed")
    preflight = yaml.safe_load(Path(reg["preflight"]).read_text())
    heads = preflight["records"]
    counts = {
        kind: sum(h["source"]["resource"]["well_known_file_type"]["name"] == kind for h in heads)
        for kind in ("NWBDownload", "3DNeuronReconstruction")
    }
    if (
        not preflight["size_admission_passed"]
        or len(heads) != reg["expected_files"]
        or len({h["source"]["file_id"] for h in heads}) != len(heads)
        or counts["NWBDownload"] != reg["expected_nwb_files"]
        or counts["3DNeuronReconstruction"] != reg["expected_morphology_files"]
        or sum(h["content_length"] for h in heads) != reg["expected_total_bytes"]
    ):
        raise ValueError("registered complete cohort changed")
    context = {
        "registration_sha256": sha(REGISTRATION),
        "implementation_sha256": {
            p: sha(Path(p))
            for p in (
                str(Path(__file__).resolve().relative_to(Path.cwd())),
                "scripts/collect_sst_vip_native_resources.py",
                "scripts/collect_sst_vip_parameters.py",
                "src/smart_robustness/validation/sst_vip_parameter_acquisition.py",
                "src/smart_robustness/validation/sst_vip_physiology_download.py",
                "src/smart_robustness/validation/sst_vip_physiology_resources.py",
            )
        },
    }
    directory = Path(reg["output_directory"])
    directory.mkdir(exist_ok=True)
    with (directory / "collector.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = directory / "manifest.yaml"
        if manifest_path.exists():
            saved = yaml.safe_load(manifest_path.read_text())
            if (
                saved["context"] != context
                or [r["preflight_record"] for r in saved["records"]] != heads
            ):
                raise ValueError("terminal acquisition lineage changed")
            for receipt in saved["records"]:
                head = receipt["preflight_record"]
                sidecar = directory / f"{head['source']['file_id']}.provenance.yaml"
                if yaml.safe_load(sidecar.read_text()) != receipt:
                    raise ValueError("terminal provenance changed")
                check_saved(receipt, head=head, context=context, path=receipt_path(head, directory))
            print("Verified terminal receipts; no new requests")
            return
        pending = [
            h
            for h in heads
            if not (directory / f"{h['source']['file_id']}.provenance.yaml").exists()
        ]
        if (
            shutil.disk_usage(directory).free - sum(h["content_length"] for h in pending)
            < reg["minimum_free_disk_bytes"]
        ):
            raise ValueError("full pending cohort would violate disk reserve")
        with ThreadPoolExecutor(max_workers=reg["workers"]) as pool:
            records = list(
                pool.map(
                    lambda h: collect_one(h, reg=reg, directory=directory, context=context), heads
                )
            )
        failed = sum(r["status"] != "complete-receipt-awaiting-format-validation" for r in records)
        save_yaml(
            manifest_path,
            {
                "schema_version": 1,
                "registration": str(REGISTRATION),
                "context": context,
                "records": records,
                "failed_records": failed,
                "status": "complete-with-failures"
                if failed
                else "acquisition-complete-awaiting-format-validation",
                "native_models_executed": False,
                "physiology_validated": False,
                "missing_morphology_specimen_ids": preflight["plan"][
                    "missing_morphology_specimen_ids"
                ],
            },
        )
        print(
            f"{len(records) - failed}/{len(records)} complete receipts; {failed} failures retained"
        )


if __name__ == "__main__":
    main()
