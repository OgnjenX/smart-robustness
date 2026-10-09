"""Bounded, checkpointed, JSON-only model-parameter acquisition."""

from __future__ import annotations

import fcntl
import hashlib
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import HTTPRedirectHandler, build_opener

import yaml

from smart_robustness.validation.sst_vip_parameter_acquisition import (
    acquisition_plan,
    validate_json,
    validate_url,
)

REGISTRATION = Path(
    "docs/validation-results/post2008-sst-vip-parameter-acquisition-registration-1033.yaml"
)
VALIDATOR = Path("src/smart_robustness/validation/sst_vip_parameter_acquisition.py")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RegisteredRedirect(HTTPRedirectHandler):
    def __init__(self, base: str):
        self.base = base

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validate_url(newurl, self.base)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def collect_one(record: dict, *, directory: Path, rules: dict, identity: dict) -> dict:
    raw_path = directory / f"{record['file_id']}.raw.json"
    provenance_path = directory / f"{record['file_id']}.provenance.yaml"
    if provenance_path.exists():
        saved = yaml.safe_load(provenance_path.read_text())
        if saved["identity"] != identity or saved["source"] != record:
            raise ValueError("resume identity changed")
        if saved.get("raw_sha256") is not None:
            if sha(raw_path) != saved["raw_sha256"]:
                raise ValueError("resume raw hash changed")
            if saved["status"] == "downloaded-valid-json":
                validate_json(raw_path.read_bytes(), rules["maximum_bytes_per_file"])
        return saved  # Explicit failed requests are not silently retried.
    if raw_path.exists():
        raise ValueError("orphan raw file requires inspection, not overwrite")
    result = {
        "identity": identity,
        "source": record,
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "status": "request-failed",
    }
    try:
        validate_url(record["request_url"], rules["source_base"])
        opener = build_opener(RegisteredRedirect(rules["source_base"]))
        with opener.open(record["request_url"], timeout=rules["timeout_seconds"]) as response:
            result["final_url"] = response.geturl()
            validate_url(result["final_url"], rules["source_base"])
            raw = response.read(rules["maximum_bytes_per_file"] + 1)
        with raw_path.open("xb") as handle:
            handle.write(raw)
        result.update(
            {
                "raw_path": str(raw_path),
                "raw_bytes": len(raw),
                "raw_sha256": sha(raw_path),
                "received_bytes_may_be_truncated": len(raw) > rules["maximum_bytes_per_file"],
            }
        )
        result["top_level_keys"] = validate_json(raw, rules["maximum_bytes_per_file"])
        result["status"] = "downloaded-valid-json"
    except (OSError, ValueError, TypeError) as exc:
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
    with provenance_path.open("x") as handle:
        yaml.safe_dump(result, handle, sort_keys=False)
    return result


def main() -> None:
    reg = yaml.safe_load(REGISTRATION.read_text())
    source_path = Path(reg["source_manifest"])
    if (
        sha(source_path) != reg["source_manifest_sha256"]
        or sha(Path(reg["parent"])) != reg["parent_sha256"]
    ):
        raise ValueError("registered source lineage changed")
    for key in (
        "network_execution_authorized",
        "isolated_simulation_authorized",
        "parameter_fitting_authorized",
    ):
        if reg[key] is not False:
            raise ValueError("acquisition-only authorization changed")
    source = yaml.safe_load(source_path.read_text())
    rules = reg["acquisition"]
    plan = acquisition_plan(source, rules)
    identity = {
        "registration_sha256": sha(REGISTRATION),
        "source_sha256": sha(source_path),
        "collector_sha256": sha(Path(__file__)),
        "validator_sha256": sha(VALIDATOR),
    }
    directory = Path(rules["output_directory"])
    directory.mkdir(exist_ok=True)
    with (directory / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with ThreadPoolExecutor(max_workers=rules["workers"]) as pool:
            records = list(
                pool.map(
                    lambda record: collect_one(
                        record, directory=directory, rules=rules, identity=identity
                    ),
                    plan,
                )
            )
        passed = sum(r["status"] == "downloaded-valid-json" for r in records)
        manifest = {
            "schema_version": 1,
            "identity": identity,
            "status": "complete" if passed == len(plan) else "complete-with-acquisition-failures",
            "records": records,
            "valid_json_records": passed,
            "failed_records": len(plan) - passed,
            "no_model_specimens": source["summary"]["no_model_specimens"],
            "network_execution": False,
            "isolated_simulation": False,
            "parameter_fitting": False,
        }
        path = directory / "manifest.yaml"
        if path.exists():
            if yaml.safe_load(path.read_text()) != manifest:
                raise ValueError("terminal acquisition manifest changed")
        else:
            with path.open("x") as handle:
                yaml.safe_dump(manifest, handle, sort_keys=False)
        print(f"{manifest['status']}: {passed}/{len(plan)} valid parameter JSON records")


if __name__ == "__main__":
    main()
