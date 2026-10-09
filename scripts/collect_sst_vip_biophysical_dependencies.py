"""Resolve native template files for all registered biophysical models."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlencode

import yaml
from collect_sst_vip_native_resources import acquire, save_yaml, sha

REGISTRATION = Path(
    "docs/validation-results/post2008-sst-vip-biophysical-dependency-registration-1042.yaml"
)
BIOPHYSICAL_NAMES = frozenset(("Biophysical - all active", "Biophysical - perisomatic"))


def validate_batch(payload: dict, requested: list[int], expected: dict) -> list[dict]:
    rows = payload.get("msg")
    if (
        payload.get("success") is not True
        or not isinstance(rows, list)
        or payload.get("total_rows") != len(rows)
        or len(rows) != len(requested)
    ):
        raise ValueError("incomplete or failed native dependency response")
    ids = [r.get("id") for r in rows]
    if len(set(ids)) != len(ids) or set(ids) != set(requested):
        raise ValueError("native dependency model identities changed")
    for r in rows:
        original = expected[r["id"]]
        template = r.get("neuronal_model_template")
        if (
            r.get("specimen_id") != original["specimen_id"]
            or not isinstance(template, dict)
            or template.get("id") != original["neuronal_model_template"]["id"]
            or template.get("name") != original["neuronal_model_template"]["name"]
        ):
            raise ValueError("native model specimen or template identity changed")
        if not isinstance(template.get("well_known_files"), list):
            raise TypeError("native template file metadata missing")
        expected_ids = {f["id"] for f in original["well_known_files"]}
        actual_ids = {
            f["id"]
            for f in r["well_known_files"]
            if f["well_known_file_type"]["name"] == "NeuronalModelParameters"
        }
        if actual_ids != expected_ids:
            raise ValueError("native parameter file identity changed")
        if r.get("specimen", {}).get("id") != original["specimen_id"]:
            raise ValueError("nested native specimen identity changed")
    return sorted(rows, key=lambda r: r["id"])


def mechanism_candidates(model: dict, parameter_object: dict) -> dict:
    names = sorted({g["mechanism"] for g in parameter_object["genome"] if g["mechanism"]})
    files = model["neuronal_model_template"]["well_known_files"]
    mapped = {}
    missing, ambiguous = [], []
    for name in names:
        candidates = [
            f
            for f in files
            if Path(f["path"]).name == name + ".mod"
            and f["well_known_file_type"]["name"] == "BiophysicalModelDescription"
        ]
        mapped[name] = candidates
        if not candidates:
            missing.append(name)
        elif len(candidates) > 1:
            ambiguous.append(name)
    return {
        "model_id": model["id"],
        "specimen_id": model["specimen_id"],
        "template_id": model["neuronal_model_template"]["id"],
        "mechanism_candidates": mapped,
        "missing_mechanisms": missing,
        "ambiguous_mechanisms": ambiguous,
        "channel_equations_validated": False,
        "axon_type_resolved": False,
    }


def main() -> None:
    reg = yaml.safe_load(REGISTRATION.read_text())
    for key in ("parent", "model_manifest", "parameter_manifest"):
        if sha(Path(reg[key])) != reg[key + "_sha256"]:
            raise ValueError("native dependency source lineage changed")
    for key in (
        "network_execution_authorized",
        "isolated_simulation_authorized",
        "parameter_fitting_authorized",
    ):
        if reg[key] is not False:
            raise ValueError("native dependency query is metadata only")
    source = yaml.safe_load(Path(reg["model_manifest"]).read_text())
    expected = {
        n["id"]: n
        for r in source["records"]
        for n in r["neuronal_models"]
        if n["neuronal_model_template"]["name"] in BIOPHYSICAL_NAMES
    }
    ids = sorted(expected)
    if len(ids) != 25:
        raise ValueError("registered biophysical model cohort changed")
    parameter_manifest = yaml.safe_load(Path(reg["parameter_manifest"]).read_text())
    parameters = {r["source"]["model_id"]: r for r in parameter_manifest["records"]}
    directory = Path(reg["output_directory"])
    directory.mkdir()
    size = reg["query"]["batch_size"]
    rows, batches = [], []
    for number, offset in enumerate(range(0, len(ids), size)):
        requested = ids[offset : offset + size]
        query = (
            "model::NeuronalModel,rma::criteria[id$in"
            + ",".join(map(str, requested))
            + "],rma::include,"
            + reg["query"]["include"]
            + ",rma::options[num_rows$eq'all']"
        )
        entry = acquire(
            reg["query"]["endpoint"] + "?" + urlencode({"q": query}),
            directory / f"batch-{number:02d}.json",
        )
        entry["requested_ids"] = requested
        rows.extend(
            validate_batch(json.loads(Path(entry["raw_path"]).read_bytes()), requested, expected)
        )
        batches.append(entry)
    candidates = []
    for row in rows:
        record = parameters[row["id"]]
        path = Path(record["raw_path"])
        if sha(path) != record["raw_sha256"]:
            raise ValueError("acquired native parameter bytes changed")
        candidates.append(mechanism_candidates(row, json.loads(path.read_bytes())))
    result = {
        "schema_version": 1,
        "status": "metadata-complete-native-setup-not-validated",
        "registration": str(REGISTRATION),
        "registration_sha256": sha(REGISTRATION),
        "collector_sha256": sha(Path(__file__)),
        "acquisition_helper_sha256": sha(Path("scripts/collect_sst_vip_native_resources.py")),
        "batches": batches,
        "models": rows,
        "mechanism_inventory": candidates,
        "original_no_model_specimens": source["summary"]["no_model_specimens"],
        "network_execution": False,
        "native_code_executed": False,
    }
    save_yaml(directory / "manifest.yaml", result)
    print(
        json.dumps(
            {
                "models": len(rows),
                "templates": len({r["neuronal_model_template"]["id"] for r in rows}),
                "missing_mechanism_mappings": sum(len(c["missing_mechanisms"]) for c in candidates),
                "ambiguous_mechanism_mappings": sum(
                    len(c["ambiguous_mechanisms"]) for c in candidates
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
