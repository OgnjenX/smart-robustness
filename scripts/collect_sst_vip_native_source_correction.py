"""Archive explicitly corrected native source; parse text without importing it."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import yaml
from collect_sst_vip_native_resources import acquire, save_yaml, sha

REGISTRATION = Path(
    "docs/validation-results/post2008-sst-vip-native-source-correction-registration-1039.yaml"
)


def validate_tree(tree: dict, *, revision: str, path: str) -> dict:
    if tree.get("sha") != revision or tree.get("truncated") is not False:
        raise ValueError("native source tree is incomplete or wrong revision")
    matches = [entry for entry in tree["tree"] if entry["path"] == path]
    if len(matches) != 1 or matches[0]["type"] != "blob":
        raise ValueError("corrected native source is not uniquely identified in tree")
    return matches[0]


def inspect_classes(text: str) -> dict[str, list[str]]:
    tree = ast.parse(text)
    return {
        node.name: sorted(
            child.name
            for child in node.body
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    }


def main() -> None:
    reg = yaml.safe_load(REGISTRATION.read_text())
    for key in ("parent", "original_manifest"):
        if sha(Path(reg[key])) != reg[key + "_sha256"]:
            raise ValueError("source correction lineage changed")
    for key in (
        "network_execution_authorized",
        "isolated_simulation_authorized",
        "parameter_fitting_authorized",
    ):
        if reg[key] is not False:
            raise ValueError("source correction is inspection only")
    directory = Path(reg["output_directory"])
    directory.mkdir()
    source = reg["source"]
    limit = source["maximum_bytes_per_file"]
    tree_record = acquire(source["tree_request"], directory / "tree.json", maximum_bytes=limit)
    if tree_record.get("oversized_or_partial_receipt"):
        raise ValueError("native tree exceeds registered byte limit")
    entry = validate_tree(
        json.loads(Path(tree_record["raw_path"]).read_bytes()),
        revision=source["revision"],
        path=source["corrected_path"],
    )
    source_record = acquire(
        source["raw_source_base"] + source["corrected_path"],
        directory / "corrected-source.txt",
        maximum_bytes=limit,
    )
    if source_record.get("oversized_or_partial_receipt"):
        raise ValueError("corrected native source exceeds registered byte limit")
    raw = Path(source_record["raw_path"]).read_bytes()
    blob_sha = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    if blob_sha != entry["sha"]:
        raise ValueError("corrected bytes do not match pinned repository tree blob")
    classes = inspect_classes(raw.decode())
    result = {
        "schema_version": 1,
        "status": "archived-awaiting-setup-audit",
        "registration": str(REGISTRATION),
        "registration_sha256": sha(REGISTRATION),
        "collector_sha256": sha(Path(__file__)),
        "acquisition_helper_sha256": sha(Path("scripts/collect_sst_vip_native_resources.py")),
        "tree": tree_record,
        "source": source_record,
        "pinned_blob": entry,
        "classes": classes,
        "original_failures_preserved": True,
        "native_source_executed": False,
    }
    save_yaml(directory / "manifest.yaml", result)
    print(json.dumps({"blob_identity_verified": True, "class_names": sorted(classes)}, indent=2))


if __name__ == "__main__":
    main()
