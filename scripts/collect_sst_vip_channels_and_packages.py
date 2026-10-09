"""Acquire every registered native channel/package; no extraction or execution."""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import build_opener
from zipfile import BadZipFile

import yaml
from collect_sst_vip_native_resources import save_yaml, sha
from collect_sst_vip_parameters import RegisteredRedirect

from smart_robustness.validation.sst_vip_native_archives import inspect_channel, inspect_package
from smart_robustness.validation.sst_vip_parameter_acquisition import validate_url

REGISTRATION = Path(
    "docs/validation-results/post2008-sst-vip-channel-package-acquisition-registration-1045.yaml"
)


def collect_one(task: dict, *, reg: dict, directory: Path) -> dict:
    result = {"task": task, "retrieved_at_utc": datetime.now(UTC).isoformat(), "status": "failed"}
    base = reg["channels"]["source_base"]
    try:
        validate_url(task["url"], base)
        opener = build_opener(RegisteredRedirect(base))
        with opener.open(task["url"], timeout=reg["transport"]["timeout_seconds"]) as response:
            result["final_url"] = response.geturl()
            validate_url(result["final_url"], base)
            raw = response.read(task["maximum_bytes"] + 1)
        path = directory / task["filename"]
        with path.open("xb") as handle:
            handle.write(raw)
        result.update(
            {
                "raw_path": str(path),
                "raw_sha256": hashlib.sha256(raw).hexdigest(),
                "raw_bytes": len(raw),
                "received_bytes_may_be_truncated": len(raw) > task["maximum_bytes"],
            }
        )
        if task["kind"] == "channel":
            result["inspection"] = inspect_channel(raw, task["maximum_bytes"])
        else:
            result["inspection"] = inspect_package(raw, reg["packages"])
        result["status"] = "acquired-and-statically-inspected"
    except (OSError, ValueError, TypeError, RuntimeError, BadZipFile, NotImplementedError) as exc:
        result.update({"error_type": type(exc).__name__, "error": str(exc)})
    save_yaml(directory / (task["filename"] + ".provenance.yaml"), result)
    return result


def main() -> None:
    reg = yaml.safe_load(REGISTRATION.read_text())
    for key in ("parent", "dependency_manifest"):
        if sha(Path(reg[key])) != reg[key + "_sha256"]:
            raise ValueError("native channel/package source lineage changed")
    for key in (
        "network_execution_authorized",
        "isolated_simulation_authorized",
        "parameter_fitting_authorized",
        "compile_or_import_native_code_authorized",
    ):
        if reg[key] is not False:
            raise ValueError("native channel/package acquisition is static only")
    source = yaml.safe_load(Path(reg["dependency_manifest"]).read_text())
    files = {}
    for model in source["models"]:
        template = model["neuronal_model_template"]
        for resource in template["well_known_files"]:
            file_id = resource["id"]
            path = reg["channels"]["path_format"].format(file_id=file_id)
            if (
                resource["download_link"] != path
                or resource["well_known_file_type"]["name"] != reg["channels"]["file_type"]
            ):
                raise ValueError("registered native channel source identity changed")
            files.setdefault(file_id, []).append(
                {"model_id": model["id"], "template_id": template["id"], "resource": resource}
            )
    if len(files) != reg["channels"]["expected_unique_resources"] or len(source["models"]) != 25:
        raise ValueError("registered native resource cohort changed")
    tasks = [
        {
            "kind": "channel",
            "file_id": file_id,
            "references": files[file_id],
            "filename": f"channel-{file_id}.txt",
            "url": reg["channels"]["source_base"]
            + reg["channels"]["path_format"].format(file_id=file_id),
            "maximum_bytes": reg["channels"]["maximum_bytes_per_file"],
        }
        for file_id in sorted(files)
    ]
    tasks.extend(
        {
            "kind": "package",
            "model_id": model["id"],
            "template": model["neuronal_model_template"]["name"],
            "filename": f"package-{model['id']}.zip",
            "url": reg["packages"]["endpoint_format"].format(model_id=model["id"]),
            "maximum_bytes": reg["packages"]["maximum_compressed_bytes_per_package"],
        }
        for model in sorted(source["models"], key=lambda m: m["id"])
    )
    directory = Path(reg["output_directory"])
    directory.mkdir()
    with ThreadPoolExecutor(max_workers=reg["transport"]["workers"]) as pool:
        records = list(
            pool.map(lambda task: collect_one(task, reg=reg, directory=directory), tasks)
        )
    failed = sum(r["status"] == "failed" for r in records)
    result = {
        "schema_version": 1,
        "registration": str(REGISTRATION),
        "registration_sha256": sha(REGISTRATION),
        "collector_sha256": sha(Path(__file__)),
        "inspection_sha256": sha(
            Path("src/smart_robustness/validation/sst_vip_native_archives.py")
        ),
        "helper_sha256": {
            path: sha(Path(path))
            for path in (
                "scripts/collect_sst_vip_native_resources.py",
                "scripts/collect_sst_vip_parameters.py",
                "src/smart_robustness/validation/sst_vip_parameter_acquisition.py",
            )
        },
        "status": "complete-with-failures"
        if failed
        else "acquisition-complete-awaiting-equation-audit",
        "records": records,
        "failed_records": failed,
        "native_code_executed": False,
        "archive_extracted": False,
    }
    save_yaml(directory / "manifest.yaml", result)
    print(
        f"{result['status']}: {len(records) - failed}/{len(records)} acquired and statically inspected"
    )


if __name__ == "__main__":
    main()
