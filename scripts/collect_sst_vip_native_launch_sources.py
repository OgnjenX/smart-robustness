"""Acquire six pinned launch sources without importing or executing them."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from urllib.request import build_opener

import yaml
from collect_sst_vip_native_resources import save_yaml, sha
from collect_sst_vip_native_source_correction import validate_tree
from collect_sst_vip_parameters import RegisteredRedirect

REGISTRATION = Path(
    "docs/validation-results/post2008-sst-vip-native-launch-audit-registration-1048.yaml"
)


def inspect_source(raw: bytes, *, entry: dict, python: bool) -> dict:
    blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    if blob != entry["sha"]:
        raise ValueError("source bytes differ from pinned tree blob")
    text = raw.decode("utf-8")
    definitions = []
    if python:
        definitions = sorted(
            node.name
            for node in ast.walk(ast.parse(text))
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        )
    return {"blob_sha1": blob, "definitions": definitions, "source_executed": False}


def main():
    reg = yaml.safe_load(REGISTRATION.read_text())
    for key in ("parent", "tree"):
        if sha(Path(reg[key])) != reg[key + "_sha256"]:
            raise ValueError("registered source lineage changed")
    for key in (
        "network_execution_authorized",
        "isolated_simulation_authorized",
        "parameter_fitting_authorized",
        "native_code_execution_authorized",
    ):
        if reg[key] is not False:
            raise ValueError("static-only authorization changed")
    tree = json.loads(Path(reg["tree"]).read_text())
    entries = [validate_tree(tree, revision=reg["revision"], path=p) for p in reg["paths"]]
    directory = Path(reg["output_directory"])
    directory.mkdir()
    records = []
    for number, (path, entry) in enumerate(zip(reg["paths"], entries, strict=True)):
        record = {"repository_path": path, "pinned_blob": entry, "status": "failed"}
        url = reg["source_base"] + path
        record["request_url"] = url
        try:
            with build_opener(RegisteredRedirect("https://raw.githubusercontent.com")).open(
                url, timeout=reg["timeout_seconds"]
            ) as response:
                record["final_url"] = response.geturl()
                if record["final_url"] != url:
                    raise ValueError("source URL changed")
                raw = response.read(reg["maximum_bytes_per_file"] + 1)
            output = directory / f"source-{number:02d}.txt"
            with output.open("xb") as handle:
                handle.write(raw)
            record.update(raw_path=str(output), raw_sha256=sha(output), raw_bytes=len(raw))
            if len(raw) > reg["maximum_bytes_per_file"]:
                raise ValueError("source exceeds registered byte limit")
            record["inspection"] = inspect_source(raw, entry=entry, python=path.endswith(".py"))
            record["status"] = "archived-statically-inspected"
        except (OSError, ValueError, SyntaxError) as exc:
            record.update(error_type=type(exc).__name__, error=str(exc))
        save_yaml(directory / f"source-{number:02d}.provenance.yaml", record)
        records.append(record)
    result = {
        "schema_version": 1,
        "registration": str(REGISTRATION),
        "registration_sha256": sha(REGISTRATION),
        "collector_sha256": sha(Path(__file__)),
        "records": records,
        "source_executed": False,
    }
    save_yaml(directory / "manifest.yaml", result)
    print(f"{sum(r['status'] == 'archived-statically-inspected' for r in records)}/6 archived")


if __name__ == "__main__":
    main()
