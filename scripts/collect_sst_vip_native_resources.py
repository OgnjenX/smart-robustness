"""Archive pinned source text and physiology metadata; never import native code."""

from __future__ import annotations

import ast
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import yaml

from smart_robustness.validation.sst_vip_native_resources import (
    reconcile_physiology,
    validate_physiology_batch,
)

REGISTRATION = Path(
    "docs/validation-results/post2008-sst-vip-native-resource-audit-registration-1036.yaml"
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_yaml(path: Path, value: dict | list) -> None:
    with path.open("x") as handle:
        yaml.safe_dump(value, handle, sort_keys=False)


def acquire(url: str, path: Path, *, maximum_bytes: int | None = None) -> dict:
    record = {"request_url": url, "retrieved_at_utc": datetime.now(UTC).isoformat()}
    with urlopen(url, timeout=30) as response:
        raw = response.read() if maximum_bytes is None else response.read(maximum_bytes + 1)
        record["final_url"] = response.geturl()
    with path.open("xb") as handle:
        handle.write(raw)
    record.update({"raw_path": str(path), "raw_sha256": sha(path), "raw_bytes": len(raw)})
    if maximum_bytes is not None and len(raw) > maximum_bytes:
        record["oversized_or_partial_receipt"] = True
    save_yaml(path.with_suffix(".provenance.yaml"), record)
    return record


def main() -> None:
    reg = yaml.safe_load(REGISTRATION.read_text())
    for key in ("parent", "cell_inventory", "parameter_manifest"):
        if sha(Path(reg[key])) != reg[key + "_sha256"]:
            raise ValueError("registered audit source lineage changed")
    for key in (
        "network_execution_authorized",
        "isolated_simulation_authorized",
        "parameter_fitting_authorized",
    ):
        if reg[key] is not False:
            raise ValueError("inspection-only authorization changed")
    inventory = yaml.safe_load(Path(reg["cell_inventory"]).read_text())
    included = [r for r in inventory["records"] if r["included"]]
    ids = [r["specimen_id"] for r in included]
    if len(ids) != 111 or ids != sorted(set(ids)):
        raise ValueError("registered native audit cohort changed")
    directory = Path(reg["output_directory"])
    directory.mkdir()  # Exclusive; interrupted evidence requires explicit inspection.
    sources = []
    for number, source_path in enumerate(reg["native_source"]["paths"]):
        entry = {"repository_path": source_path, "status": "unresolved"}
        try:
            entry.update(
                acquire(
                    reg["native_source"]["source_base"] + source_path,
                    directory / f"source-{number:02d}.txt",
                    maximum_bytes=reg["native_source"]["maximum_bytes_per_source_file"],
                )
            )
            if entry.get("oversized_or_partial_receipt"):
                raise ValueError("native source exceeded registered size limit")
            tree = ast.parse(Path(entry["raw_path"]).read_text())
            entry["definitions"] = sorted(
                {
                    node.name
                    for node in ast.walk(tree)
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                }
            )
            entry["status"] = "archived-and-statically-parsed-not-executed"
        except (OSError, ValueError, SyntaxError) as exc:
            entry.update({"error_type": type(exc).__name__, "error": str(exc)})
        sources.append(entry)
        save_yaml(directory / f"source-{number:02d}.inspection.yaml", entry)
    metadata = reg["physiology_metadata"]
    rows, batches = [], []
    size = metadata["batch_size"]
    for number, offset in enumerate(range(0, len(ids), size)):
        requested = ids[offset : offset + size]
        query = (
            "model::Specimen,rma::criteria[id$in"
            + ",".join(map(str, requested))
            + "],rma::include,"
            + metadata["include"]
            + ",rma::options[num_rows$eq'all']"
        )
        entry = acquire(
            metadata["endpoint"] + "?" + urlencode({"q": query}),
            directory / f"physiology-{number:02d}.json",
        )
        entry["requested_ids"] = requested
        rows.extend(
            validate_physiology_batch(json.loads(Path(entry["raw_path"]).read_bytes()), requested)
        )
        batches.append(entry)
    summary = reconcile_physiology(rows, included)
    result = {
        "schema_version": 1,
        "registration": str(REGISTRATION),
        "registration_sha256": sha(REGISTRATION),
        "collector_sha256": sha(Path(__file__)),
        "validator_sha256": sha(
            Path("src/smart_robustness/validation/sst_vip_native_resources.py")
        ),
        "status": "collected-awaiting-manual-native-equation-audit",
        "source_revision": reg["native_source"]["revision"],
        "sources": sources,
        "physiology_batches": batches,
        "physiology_records": rows,
        "physiology_summary": summary,
        "native_code_executed": False,
        "model_simulation": False,
    }
    save_yaml(directory / "manifest.yaml", result)
    print(
        json.dumps(
            {
                "archived_native_sources": sum(
                    e["status"] == "archived-and-statically-parsed-not-executed" for e in sources
                ),
                **summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
