"""Acquire model identity metadata only; preserve complete raw batch lineage."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import yaml

from smart_robustness.validation.sst_vip_model_identity import reconcile, validate_batch

REGISTRATION = Path(
    "docs/validation-results/post2008-sst-vip-model-identity-registration-1030.yaml"
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    reg = yaml.safe_load(REGISTRATION.read_text())
    source_path = Path(reg["inventory"])
    if sha(source_path) != reg["inventory_sha256"]:
        raise ValueError("registered source inventory hash changed")
    if sha(Path(reg["parent"])) != reg["parent_sha256"]:
        raise ValueError("registered parent assessment hash changed")
    source = yaml.safe_load(source_path.read_text())
    included = [row for row in source["records"] if row["included"]]
    ids = [row["specimen_id"] for row in included]
    if len(ids) != reg["expected_specimens"] or ids != sorted(set(ids)):
        raise ValueError("registered specimen cohort changed")
    for key in (
        "network_execution_authorized",
        "isolated_simulation_authorized",
        "parameter_fitting_authorized",
    ):
        if reg[key] is not False:
            raise ValueError("metadata-only authorization changed")
    directory = Path(reg["outputs"]["directory"])
    directory.mkdir()  # Exclusive creation also prevents concurrent collection.
    rows = []
    batches = []
    batch_size = reg["protocol"]["batch_size"]
    for number, offset in enumerate(range(0, len(ids), batch_size)):
        requested = ids[offset : offset + batch_size]
        query = (
            "model::Specimen,rma::criteria[id$in"
            + ",".join(map(str, requested))
            + "],rma::include,"
            + reg["source"]["include"]
            + ",rma::options[num_rows$eq'all']"
        )
        url = reg["source"]["endpoint"] + "?" + urlencode({"q": query})
        with urlopen(url, timeout=60) as response:
            raw = response.read()
        path = directory / f"batch-{number:02d}.raw.json"
        with path.open("xb") as handle:
            handle.write(raw)
        provenance = {
            "request_url": url,
            "requested_ids": requested,
            "retrieved_at_utc": datetime.now(UTC).isoformat(),
            "raw_path": str(path),
            "raw_sha256": sha(path),
            "raw_bytes": len(raw),
        }
        with (directory / f"batch-{number:02d}.provenance.yaml").open("x") as handle:
            yaml.safe_dump(provenance, handle, sort_keys=False)
        batches.append(provenance)
        rows.extend(validate_batch(json.loads(raw), requested))
    summary = reconcile(rows, included)
    status = (
        "complete-with-source-discrepancies"
        if summary["count_discrepancies"] or summary["unknown_templates"]
        else "complete-counts-reconciled"
    )
    manifest = {
        "schema_version": 1,
        "status": status,
        "registration": str(REGISTRATION),
        "registration_sha256": sha(REGISTRATION),
        "collector_sha256": sha(Path(__file__)),
        "validator_sha256": sha(Path("src/smart_robustness/validation/sst_vip_model_identity.py")),
        "batches": batches,
        "summary": summary,
        "records": rows,
        "network_execution": False,
        "parameter_fitting": False,
    }
    with (directory / "manifest.yaml").open("x") as handle:
        yaml.safe_dump(manifest, handle, sort_keys=False)
    print(json.dumps({"status": status, **summary}, indent=2))


if __name__ == "__main__":
    main()
