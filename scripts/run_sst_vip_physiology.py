"""Sealed full-cohort physiology runner; no fitting or SMART network execution."""

from __future__ import annotations

import copy
import json
import logging
import platform
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path

import yaml
from run_sst_vip_glif_parity import load_reference

from smart_robustness.validation.native_glif_parity import compare, exact_repeat
from smart_robustness.validation.native_glif_state_machine import simulate_glif
from smart_robustness.validation.physiology_attempts import attempt_outputs, stored_attempt
from smart_robustness.validation.physiology_recordings import (
    digest,
    group_key,
    load_recording,
    score_group,
)
from smart_robustness.validation.physiology_storage import ArrayStore, require_space

REGISTRATION = Path(
    "docs/validation-results/post2008-sst-vip-physiology-protocol-registration-1068.yaml"
)
SEAL = Path("docs/validation-results/post2008-sst-vip-physiology-execution-seal-1073.yaml")
ELIGIBILITY_REG = Path(
    "docs/validation-results/post2008-sst-vip-physiology-eligibility-registration-1065.yaml"
)
GLIF_REG = Path("docs/validation-results/post2008-sst-vip-glif-parity-registration-1061.yaml")
OUTPUT = Path("results/sst-vip-physiology-validation-1072")


def verify_seal(registration_path, seal_path):
    reg, seal = (yaml.safe_load(p.read_text()) for p in (registration_path, seal_path))
    if seal["registration_sha256"] != digest(registration_path):
        raise ValueError("physiology registration changed")
    if not all(
        seal.get(key) is True
        for key in ("response_trace_read_authorized", "cell_simulation_authorized")
    ):
        raise ValueError("physiology execution not authorized")
    if (
        seal.get("network_execution_authorized") is not False
        or seal.get("parameter_fitting_authorized") is not False
    ):
        raise ValueError("scope changed")
    for path, expected in seal["implementation_sha256"].items():
        if digest(path) != expected:
            raise ValueError("sealed implementation changed")
    for key in ("parent", "eligibility_result"):
        if digest(reg[key]) != reg[key + "_sha256"]:
            raise ValueError("protocol lineage changed")
    for key in ("native_reader", "native_simulation"):
        if digest(reg[key + "_source"]) != reg[key + "_sha256"]:
            raise ValueError("protocol native source changed")
    for path, expected in reg["metric_implementation_sha256"].items():
        if digest(path) != expected:
            raise ValueError("registered metric implementation changed")
    if version("h5py") != "3.12.1":
        raise ValueError("registered HDF5 parser version changed")
    return reg, seal


def build_plan(eligibility, schema, parameter_records):
    by_specimen = {r["source"]["specimen_ids"][0]: r for r in schema["records"]}
    models = {
        r["source"]["model_id"]: r
        for r in parameter_records
        if not r["source"]["template"]["name"].startswith("Biophysical")
    }
    plan = []
    for specimen in sorted(eligibility["records"], key=lambda r: r["specimen_id"]):
        if not specimen["primary_candidate"]:
            continue
        source = by_specimen[specimen["specimen_id"]]
        inventories = {s["sweep_number"]: s for s in source["inspection"]["sweeps"]}
        for sweep in sorted(specimen["sweeps"], key=lambda s: s["sweep_number"]):
            if not sweep["eligible_noise_sweep"] or sweep["stimulus_name"] != "Noise 2":
                continue
            for model_id in sorted(specimen["model_ids"]):
                model = models[model_id]
                if model["source"]["specimen_id"] != specimen["specimen_id"]:
                    raise ValueError("model/specimen attachment mismatch")
                plan.append(
                    {
                        "specimen_id": specimen["specimen_id"],
                        "sweep_number": sweep["sweep_number"],
                        "model_id": model_id,
                        "source": source,
                        "inventory": inventories[sweep["sweep_number"]],
                        "model": model,
                    }
                )
    return plan


def recording_checkpoint(store, case, context):
    name = f"recording-{case['specimen_id']}-{case['sweep_number']}"
    path = store.directory / (name + ".json")
    source = case["source"]
    identity = {
        "context": context,
        "path": source["inspection"]["path"],
        "source_sha256": source["raw_sha256"],
        "inventory": case["inventory"],
    }
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["identity"] != identity or digest(identity["path"]) != identity["source_sha256"]:
            raise ValueError("recording checkpoint or source changed")
        if saved["status"] == "complete":
            for receipt in saved["arrays"].values():
                store.get(receipt)
        return saved
    samples = case["inventory"]["stimulus"]["data_shape"][0]
    require_space(
        store.directory, samples * 16 + samples * 16 // 100 + 1048576, store.reserve_bytes
    )
    try:
        recording = load_recording(
            identity["path"],
            identity["source_sha256"],
            case["inventory"],
            source["inspection"]["pipeline_version"],
            authorized=True,
        )
    except (ValueError, OSError, KeyError, TypeError) as exc:
        saved = {
            "identity": identity,
            "status": "recording-failure",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    else:
        saved = {
            "identity": identity,
            "status": "complete",
            "metadata": {k: v for k, v in recording.items() if k not in ("stimulus", "response")},
            "arrays": {k: store.put(recording[k]) for k in ("stimulus", "response")},
        }
    store.checkpoint(name, saved)
    return saved


def restore_recording(store, receipt):
    return {**receipt["metadata"], **{k: store.get(v) for k, v in receipt["arrays"].items()}}


def run_case(store, number, case, parameters, recording, reference, context):
    name = f"case-{number:04d}"
    if recording["status"] != "complete":
        result = {
            "context": context,
            "specimen_id": case["specimen_id"],
            "model_id": case["model_id"],
            "sweep_number": case["sweep_number"],
            "recording": recording,
            "status": "recording-failure",
            "parity_passed": False,
        }
        store.checkpoint(name, result)
        return result
    restored = restore_recording(store, recording)
    p = copy.deepcopy(parameters)
    p["dt"] = 1.0 / restored["rate"]
    attempts = {}
    for label, function in (
        ("native0", reference),
        ("native1", reference),
        ("candidate0", simulate_glif),
        ("candidate1", simulate_glif),
    ):
        attempts[label] = stored_attempt(
            store, name + "-" + label, p, restored["stimulus"], function, context
        )
    result = {
        "context": context,
        "specimen_id": case["specimen_id"],
        "model_id": case["model_id"],
        "sweep_number": case["sweep_number"],
        "recording": recording,
        "attempts": attempts,
        "source_dt": parameters["dt"],
        "run_dt": p["dt"],
        "parity_passed": False,
        "status": "attempts-complete",
    }
    if all(a["status"] == "complete" for a in attempts.values()):
        outputs = {k: attempt_outputs(store, a) for k, a in attempts.items()}
        result["comparison"] = compare(outputs["native0"], outputs["candidate0"])
        result["native_exact_repeat"] = exact_repeat(outputs["native0"], outputs["native1"])
        result["candidate_exact_repeat"] = exact_repeat(
            outputs["candidate0"], outputs["candidate1"]
        )
        result["bad_reset_stop"] = any(a["state"]["bad_reset_stop"] for a in attempts.values())
        result["parity_passed"] = (
            result["comparison"]["passed"]
            and result["native_exact_repeat"]
            and result["candidate_exact_repeat"]
            and not result["bad_reset_stop"]
        )
    store.checkpoint(name, result)
    return result


def summarize_groups(store, records):
    grouped, failures = defaultdict(list), []
    for result in records:
        if result["recording"]["status"] != "complete":
            failures.append(
                {
                    "specimen_id": result["specimen_id"],
                    "model_id": result["model_id"],
                    "sweep_number": result["sweep_number"],
                    "status": "recording-failure",
                }
            )
        else:
            key = (
                result["specimen_id"],
                result["model_id"],
                group_key(result["recording"]["metadata"]),
            )
            grouped[key].append(result)
    groups = []
    for (specimen, model, _), repeats in grouped.items():
        if any(r["attempts"]["candidate0"]["status"] != "complete" for r in repeats):
            report = {"passed": False, "status": "candidate-exception"}
        else:
            recordings = [restore_recording(store, r["recording"]) for r in repeats]
            predictions = [
                store.get(r["attempts"]["candidate0"]["arrays"]["interpolated_spike_times"])
                for r in repeats
            ]
            try:
                report = score_group(recordings, predictions, [r["parity_passed"] for r in repeats])
            except ValueError as exc:
                report = {"passed": False, "status": "invalid-scoring-events", "error": str(exc)}
        groups.append(
            {
                "specimen_id": specimen,
                "model_id": model,
                "sweep_numbers": [r["sweep_number"] for r in repeats],
                **report,
            }
        )
    model_keys = sorted({(r["specimen_id"], r["model_id"]) for r in records})
    models = [
        {
            "specimen_id": s,
            "model_id": m,
            "passed": not any(f["specimen_id"] == s and f["model_id"] == m for f in failures)
            and all(g["passed"] for g in groups if g["specimen_id"] == s and g["model_id"] == m)
            and any(g["specimen_id"] == s and g["model_id"] == m for g in groups),
        }
        for s, m in model_keys
    ]
    return {"groups": groups, "recording_failures": failures, "models": models}


def main():
    logging.disable(logging.INFO)
    reg, seal = verify_seal(REGISTRATION, SEAL)
    eligibility_reg = yaml.safe_load(ELIGIBILITY_REG.read_text())
    for key in ("schema_manifest", "parameter_manifest"):
        if digest(eligibility_reg[key]) != eligibility_reg[key + "_sha256"]:
            raise ValueError("cohort source changed")
    eligibility = yaml.safe_load(Path(reg["eligibility_result"]).read_text())
    schema = yaml.safe_load(Path(eligibility_reg["schema_manifest"]).read_text())
    parameter_records = yaml.safe_load(Path(eligibility_reg["parameter_manifest"]).read_text())[
        "records"
    ]
    plan = build_plan(eligibility, schema, parameter_records)
    if len(plan) != 509 or len({c["specimen_id"] for c in plan}) != 41:
        raise ValueError("registered primary case coverage changed")
    glif_reg = yaml.safe_load(GLIF_REG.read_text())
    for key in ("native_neuron", "native_methods"):
        if digest(glif_reg[key + "_source"]) != glif_reg[key + "_sha256"]:
            raise ValueError("pinned native numerical source changed")
    parameters = {}
    for case in plan:
        model = case["model"]
        if digest(model["raw_path"]) != model["raw_sha256"]:
            raise ValueError("model parameter bytes changed")
        parameters[case["model_id"]] = json.loads(Path(model["raw_path"]).read_text())
    context = {"registration_sha256": digest(REGISTRATION), "seal_sha256": digest(SEAL)}
    with ArrayStore(OUTPUT, reserve_bytes=seal["disk_reserve_bytes"]) as store:
        reference, import_audit = load_reference(glif_reg)
        records = []
        for number, case in enumerate(plan):
            recording = recording_checkpoint(store, case, context)
            records.append(
                run_case(
                    store, number, case, parameters[case["model_id"]], recording, reference, context
                )
            )
            print(f"Completed checkpoint {number + 1}/509", flush=True)
        summary = summarize_groups(store, records)
        store.checkpoint(
            "manifest",
            {
                "context": context,
                "records": records,
                "summary": summary,
                "native_import_audit": import_audit,
                "environment": {
                    "python": platform.python_version(),
                    "numpy": version("numpy"),
                    "h5py": version("h5py"),
                },
                "fitting_performed": False,
                "network_executed": False,
            },
        )
        print("Terminal physiology matrix saved; independent assessment required", flush=True)


if __name__ == "__main__":
    main()
