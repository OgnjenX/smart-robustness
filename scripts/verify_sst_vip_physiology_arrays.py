"""Independent immutable-blob and parity reconstruction; never imports simulators."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import yaml
from verify_sst_vip_glif_parity import FIELDS, reconstruct, require

ROOT = Path("results/sst-vip-physiology-validation-1072")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def source_plan(reg):
    eligibility_path = Path(
        "docs/validation-results/post2008-sst-vip-physiology-eligibility-registration-1065.yaml"
    )
    er = yaml.safe_load(eligibility_path.read_text())
    for label in ("schema_manifest", "parameter_manifest"):
        require(digest(er[label]) == er[label + "_sha256"], "source manifest changed")
    require(
        digest(reg["eligibility_result"]) == reg["eligibility_result_sha256"], "eligibility changed"
    )
    eligibility = yaml.safe_load(Path(reg["eligibility_result"]).read_text())
    schema = yaml.safe_load(Path(er["schema_manifest"]).read_text())
    source = {r["source"]["specimen_ids"][0]: r for r in schema["records"]}
    models = {}
    for r in yaml.safe_load(Path(er["parameter_manifest"]).read_text())["records"]:
        if r["source"]["template"]["name"].startswith("Biophysical"):
            continue
        require(digest(r["raw_path"]) == r["raw_sha256"], "parameter file changed")
        models[r["source"]["model_id"]] = (
            r["source"]["specimen_id"],
            json.loads(Path(r["raw_path"]).read_text()),
        )
    plan = []
    for specimen in sorted(eligibility["records"], key=lambda r: r["specimen_id"]):
        if not specimen["primary_candidate"]:
            continue
        record = source[specimen["specimen_id"]]
        sweeps = {s["sweep_number"]: s for s in record["inspection"]["sweeps"]}
        for sweep in sorted(specimen["sweeps"], key=lambda s: s["sweep_number"]):
            if not sweep["eligible_noise_sweep"] or sweep["stimulus_name"] != "Noise 2":
                continue
            for model in sorted(specimen["model_ids"]):
                model_specimen, parameters = models[model]
                require(model_specimen == specimen["specimen_id"], "model attachment differs")
                plan.append(
                    {
                        "specimen_id": specimen["specimen_id"],
                        "sweep_number": sweep["sweep_number"],
                        "model_id": model,
                        "source_path": record["inspection"]["path"],
                        "source_sha256": record["raw_sha256"],
                        "inventory": sweeps[sweep["sweep_number"]],
                        "parameters": parameters,
                    }
                )
    require(
        len(plan) == 509 and len({p["specimen_id"] for p in plan}) == 41,
        "registered source coverage differs",
    )
    return plan


def verify_alignment(result, expected):
    for label in ("specimen_id", "sweep_number", "model_id"):
        require(result[label] == expected[label], "case source identity differs")
    receipt = result["recording"]
    require(
        receipt["identity"]["path"] == expected["source_path"]
        and receipt["identity"]["source_sha256"] == expected["source_sha256"]
        and receipt["identity"]["inventory"] == expected["inventory"],
        "recording source alignment differs",
    )
    if receipt["status"] != "complete":
        return
    inv = expected["inventory"]
    meta = receipt["metadata"]
    require(
        meta["rate"] == inv["stimulus"]["sampling_rate"]
        and meta["start"] == inv["experiment_stimulus"]["index_start"]
        and meta["count"] == inv["experiment_stimulus"]["count"],
        "recording analysis epoch differs",
    )
    parameters = {**expected["parameters"], "dt": 1.0 / meta["rate"]}
    require(
        result["source_dt"] == expected["parameters"]["dt"]
        and result["run_dt"] == parameters["dt"],
        "source/run dt differs",
    )
    for attempt in result["attempts"].values():
        require(attempt["identity"]["parameters"] == parameters, "attempt source parameters differ")
        require(
            attempt["identity"]["stimulus_content_sha256"]
            == receipt["arrays"]["stimulus"]["content_sha256"],
            "attempt stimulus differs from recording",
        )


def read_array(directory, receipt):
    key = receipt["content_sha256"]
    require(re.fullmatch(r"[0-9a-f]{64}", key) is not None, "unsafe content key")
    path = directory / (key + ".npz")
    require(digest(path) == receipt["file_sha256"], "blob file digest differs")
    with np.load(path, allow_pickle=False) as archive:
        require(archive.files == ["values"], "blob field coverage differs")
        array = archive["values"]
    require(array.dtype.kind in "biufc", "non-numeric blob")
    meta = {"dtype": array.dtype.str, "shape": list(array.shape)}
    require(meta == receipt["metadata"], "blob dtype/shape differs")
    h = hashlib.sha256(json.dumps(meta, sort_keys=True).encode() + b"\0")
    data = memoryview(np.ascontiguousarray(array).reshape(-1).view(np.uint8))
    for start in range(0, len(data), 1048576):
        h.update(data[start : start + 1048576])
    require(h.hexdigest() == key, "blob content digest differs")
    require(array.nbytes == receipt["uncompressed_bytes"], "blob size differs")
    require(path.stat().st_size == receipt["stored_bytes"], "archive size differs")
    return array


def verify_case(directory, path, context):
    result = json.loads(path.read_text())
    require(result["context"] == context, "case execution context differs")
    recording = result["recording"]
    require(recording["identity"]["context"] == context, "recording context differs")
    require(
        digest(recording["identity"]["path"]) == recording["identity"]["source_sha256"],
        "source recording changed",
    )
    if recording["status"] != "complete":
        require(result["parity_passed"] is False, "failed recording promoted")
        return False
    stimulus = read_array(directory, recording["arrays"]["stimulus"])
    response = read_array(directory, recording["arrays"]["response"])
    require(
        stimulus.shape == response.shape
        and np.isfinite(stimulus).all()
        and np.isfinite(response).all(),
        "invalid recording arrays",
    )
    require(
        hashlib.sha256(stimulus.tobytes()).hexdigest() == recording["metadata"]["stimulus_sha256"],
        "stimulus identity differs",
    )
    arrays = {}
    failed = False
    for label in ("native0", "native1", "candidate0", "candidate1"):
        attempt = result["attempts"][label]
        saved = json.loads((directory / (path.stem + "-" + label + ".json")).read_text())
        require(attempt == saved, "attempt receipt differs from case")
        require(attempt["identity"]["context"] == context, "attempt context differs")
        if attempt["status"] == "simulation-exception":
            failed = True
            continue
        require(
            attempt["status"] == "complete" and set(attempt["arrays"]) == set(FIELDS),
            "attempt status or fields differ",
        )
        for field in FIELDS:
            arrays[label + "_" + field] = read_array(directory, attempt["arrays"][field])
    if failed:
        require(result["parity_passed"] is False, "exception promoted")
        return False
    comparison, repeats = reconstruct(arrays)
    require(comparison == result["comparison"], "raw-array parity gates differ")
    for kind, repeat in repeats.items():
        require(repeat == result[kind + "_exact_repeat"], "exact-repeat gate differs")
    bad = any(a["state"]["bad_reset_stop"] for a in result["attempts"].values())
    require(bad == result["bad_reset_stop"], "recorded reset flags differ")
    passed = comparison["passed"] and all(repeats.values()) and not bad
    require(passed == result["parity_passed"], "combined parity gate differs")
    return passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoints",
        action="store_true",
        help="verify completed cases only, never claim terminal completeness",
    )
    args = parser.parse_args()
    registration = Path(
        "docs/validation-results/post2008-sst-vip-physiology-protocol-registration-1068.yaml"
    )
    seal_path = Path("docs/validation-results/post2008-sst-vip-physiology-execution-seal-1073.yaml")
    seal = yaml.safe_load(seal_path.read_text())
    require(seal["registration_sha256"] == digest(registration), "registration changed")
    for path, expected in seal["implementation_sha256"].items():
        require(digest(path) == expected, "sealed implementation changed")
    context = {"registration_sha256": digest(registration), "seal_sha256": digest(seal_path)}
    reg = yaml.safe_load(registration.read_text())
    plan = source_plan(reg)
    paths = sorted(p for p in ROOT.glob("case-*.json") if re.fullmatch(r"case-\d{4}", p.stem))
    terminal = ROOT / "manifest.json"
    if not args.checkpoints:
        manifest = json.loads(terminal.read_text())
        require(
            manifest["context"] == context and len(paths) == len(manifest["records"]) == 509,
            "terminal case coverage incomplete",
        )
        for path, record in zip(paths, manifest["records"], strict=True):
            require(json.loads(path.read_text()) == record, "terminal case record differs")
    passed = 0
    for path in paths:
        number = int(path.stem.removeprefix("case-"))
        require(number < len(plan), "unexpected case number")
        verify_alignment(json.loads(path.read_text()), plan[number])
        passed += verify_case(ROOT, path, context)
    print(
        json.dumps(
            {
                "completed_cases_verified": len(paths),
                "parity_passed_cases": passed,
                "partial_checkpoint_mode": args.checkpoints,
                "physiological_scoring_independently_verified": False,
                "recorded_stop_flags_not_independently_reexecuted": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
