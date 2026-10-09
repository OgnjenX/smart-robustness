"""Independent receipt audit. HDF5 signature is not an NWB schema/physiology gate."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from pathlib import Path

import yaml


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def swc_inventory(path):
    nodes = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.split("#", 1)[0].strip()
        if not text:
            continue
        fields = text.split()
        require(len(fields) == 7, "SWC row must contain seven fields")
        nid, kind, parent = int(fields[0]), int(fields[1]), int(fields[6])
        values = [float(x) for x in fields[2:6]]
        require(nid > 0 and nid not in nodes, "duplicate or invalid SWC node id")
        require(all(math.isfinite(x) for x in values), "nonfinite SWC coordinate or radius")
        require(values[-1] >= 0, "negative SWC radius")
        nodes[nid] = (parent, kind)
    require(bool(nodes), "empty SWC")
    roots = [n for n, (parent, _) in nodes.items() if parent == -1]
    require(bool(roots), "SWC has no root")
    resolved = set()
    for start in nodes:
        current, chain = start, set()
        while current != -1 and current not in resolved:
            require(current in nodes, "SWC parent id missing")
            require(current not in chain, "SWC parent cycle")
            chain.add(current)
            current = nodes[current][0]
        resolved.update(chain)
    return {
        "nodes": len(nodes),
        "roots": roots,
        "node_type_counts": dict(Counter(kind for _, kind in nodes.values())),
        "physiological_morphology_validated": False,
    }


def verify_receipt(record, path):
    require(str(path) == record["raw_path"], "receipt path mismatch")
    require(path.stat().st_size == record["raw_bytes"], "receipt length changed")
    require(digest(path) == record["raw_sha256"], "receipt hash changed")
    complete = record["status"] == "complete-receipt-awaiting-format-validation"
    require(
        not complete or record["raw_bytes"] == record["preflight_record"]["content_length"],
        "complete receipt length mismatch",
    )
    result = {"raw_hash_and_length_verified": True, "complete": complete}
    if complete:
        kind = record["preflight_record"]["source"]["resource"]["well_known_file_type"]["name"]
        if kind == "NWBDownload":
            with path.open("rb") as handle:
                require(handle.read(8) == b"\x89HDF\r\n\x1a\n", "missing expected HDF5 signature")
            result.update(hdf5_signature_verified=True, nwb_schema_validated=False)
        else:
            require(kind == "3DNeuronReconstruction", "unexpected resource type")
            result["swc"] = swc_inventory(path)
    return result


def verify(manifest_path):
    m = yaml.safe_load(manifest_path.read_text())
    reg_path = Path(m["registration"])
    reg = yaml.safe_load(reg_path.read_text())
    require(digest(reg_path) == m["context"]["registration_sha256"], "registration changed")
    for key in ("parent", "preflight"):
        require(digest(Path(reg[key])) == reg[key + "_sha256"], "acquisition lineage changed")
    for path, expected in m["context"]["implementation_sha256"].items():
        require(digest(Path(path)) == expected, "sealed implementation changed")
    preflight = yaml.safe_load(Path(reg["preflight"]).read_text())
    require(
        [r["preflight_record"] for r in m["records"]] == preflight["records"],
        "full receipt cohort changed",
    )
    require(len(m["records"]) == reg["expected_files"], "receipt cohort incomplete")
    inventories = []
    for r in m["records"]:
        head = r["preflight_record"]
        fid = head["source"]["file_id"]
        require(r["context"] == m["context"], "receipt context changed")
        require(
            yaml.safe_load((manifest_path.parent / f"{fid}.provenance.yaml").read_text()) == r,
            "receipt sidecar changed",
        )
        kind = head["source"]["resource"]["well_known_file_type"]["name"]
        path = manifest_path.parent / (str(fid) + (".nwb" if kind == "NWBDownload" else ".swc"))
        item = {"file_id": fid, "resource_type": kind, "receipt_status": r["status"]}
        if "raw_path" in r:
            item.update(verify_receipt(r, path))
        else:
            require(r["status"] == "failed" and not path.exists(), "unrecorded receipt bytes")
        inventories.append(item)
    failed = sum(r["status"] != "complete-receipt-awaiting-format-validation" for r in m["records"])
    require(failed == m["failed_records"], "failure count changed")
    require(
        m["missing_morphology_specimen_ids"]
        == preflight["plan"]["missing_morphology_specimen_ids"],
        "missing morphology cohort changed",
    )
    require(
        m["native_models_executed"] is False and m["physiology_validated"] is False,
        "acquisition boundary changed",
    )
    return {
        "independent_receipt_audit": "passed",
        "manifest_sha256": digest(manifest_path),
        "manifest_bytes": manifest_path.stat().st_size,
        "failed_records": failed,
        "total_preserved_raw_bytes": sum(r.get("raw_bytes", 0) for r in m["records"]),
        "inventories": inventories,
        "nwb_schema_validated": False,
        "physiological_validation_performed": False,
    }


if __name__ == "__main__":
    print(
        yaml.safe_dump(
            verify(Path("results/sst-vip-physiology-acquisition-1055/manifest.yaml")),
            sort_keys=False,
        )
    )
