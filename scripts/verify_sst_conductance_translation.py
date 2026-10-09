"""Independent saved-array verification; imports neither simulator nor runner."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from verify_sst_vip_glif_parity import FIELDS, reconstruct

ROOT = Path("results/sst-conductance-translation-1084")
REGISTRATION = Path(
    "docs/validation-results/post2008-sst-conductance-translation-registration-1080.yaml"
)
SEAL = Path("docs/validation-results/post2008-sst-conductance-translation-execution-seal-1083.yaml")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_report(arrays, report, *, null):
    errors = report["errors"]
    backends = {"native0", "native1", "candidate0", "candidate1"}
    require(
        set(errors) <= backends | ({"current_only"} if null else set()), "unknown error backend"
    )
    require(set(report["states"]) == backends - set(errors), "state coverage changed")
    expected = {"bias", "excitation", "inhibition", "report_json"}
    for backend in ("native0", "native1", "candidate0", "candidate1"):
        if backend not in errors:
            expected.update(backend + "_" + field for field in FIELDS)
            require(backend in report["states"], "successful attempt lacks state")
        else:
            require(backend not in report["states"], "failed attempt has state")
    if null and not errors:
        expected.update("current_only_" + field for field in FIELDS)
    require(set(arrays.files) == expected, "output field coverage changed")
    if errors:
        require(report["passed"] is False, "error case promoted")
        return False
    comparison, repeats = reconstruct(arrays)
    require(report["comparison"] == comparison, "array parity assessment differs")
    require(report["native_exact_repeat"] == repeats["native"], "native repeat differs")
    require(report["candidate_exact_repeat"] == repeats["candidate"], "candidate repeat differs")
    bad_reset = any(s["bad_reset_stop"] for s in report["states"].values())
    require(report["bad_reset_stop"] == bad_reset, "stop flag summary differs")
    null_pass = not null or all(
        np.array_equal(arrays["current_only_" + f], arrays["candidate0_" + f], equal_nan=True)
        for f in FIELDS
    )
    require(report["null_equivalence"] == null_pass, "current-only null differs")
    passed = comparison["passed"] and all(repeats.values()) and not bad_reset and null_pass
    require(report["passed"] == passed, "overall gate differs")
    return passed


def main():
    reg = yaml.safe_load(REGISTRATION.read_text())
    seal = yaml.safe_load(SEAL.read_text())
    context = {"registration_sha256": digest(REGISTRATION), "seal_sha256": digest(SEAL)}
    require(seal["registration_sha256"] == context["registration_sha256"], "registration changed")
    require(seal["synthetic_execution_authorized"] is True, "not authorized")
    require(
        seal["network_execution_authorized"] is False
        and seal["parameter_fitting_authorized"] is False,
        "scope changed",
    )
    for path, expected in seal["implementation_sha256"].items():
        require(digest(path) == expected, "implementation changed")
    require(digest(reg["parent"]) == reg["parent_sha256"], "identity lineage changed")
    parent = yaml.safe_load(Path(reg["parent"]).read_text())
    for key in ("cell_inventory", "parameter_manifest"):
        require(digest(parent[key]) == parent[key + "_sha256"], "source lineage changed")
    native_path = Path(
        "docs/validation-results/post2008-sst-vip-glif-parity-registration-1061.yaml"
    )
    require(
        digest(native_path) == seal["native_registration_sha256"], "native registration changed"
    )
    native = yaml.safe_load(native_path.read_text())
    for key in ("native_neuron", "native_methods"):
        require(digest(native[key + "_source"]) == native[key + "_sha256"], "native source changed")
    maps = {}
    for record in yaml.safe_load(Path(parent["parameter_manifest"]).read_text())["records"]:
        model = record["source"]["model_id"]
        if model in reg["cohort"]["model_ids"]:
            require(model not in maps, "duplicate source model")
            require(digest(record["raw_path"]) == record["raw_sha256"], "parameter source changed")
            maps[model] = json.loads(Path(record["raw_path"]).read_text())
    require(set(maps) == set(reg["cohort"]["model_ids"]), "source cohort changed")
    terminal = yaml.safe_load((ROOT / "manifest.yaml").read_text())
    require(
        terminal["context"] == context and terminal["cases"] == 90 and terminal["attempts"] == 360,
        "terminal coverage/context changed",
    )
    require(len(terminal["records"]) == 90, "incomplete matrix")
    identities = [
        (m, dt, arm)
        for m in reg["cohort"]["model_ids"]
        for dt in reg["synthetic_protocol"]["dt_seconds"]
        for arm in reg["synthetic_protocol"]["arms"]
    ]
    passed, failures = 0, []
    for number, (saved, (model, dt, arm)) in enumerate(
        zip(terminal["records"], identities, strict=True)
    ):
        require(
            yaml.safe_load((ROOT / f"case-{number:03d}.yaml").read_text()) == saved,
            "sidecar differs",
        )
        p = dict(maps[model], dt=dt)
        require(
            saved["case"] == {"model_id": model, "dt": dt, "arm": arm, "parameters": p},
            "case identity changed",
        )
        require(saved["context"] == context, "case context changed")
        raw = ROOT / f"case-{number:03d}.npz"
        require(
            saved["raw_path"] == str(raw) and saved["raw_sha256"] == digest(raw),
            "raw arrays changed",
        )
        with np.load(raw, allow_pickle=False) as arrays:
            require(
                json.loads(str(arrays["report_json"].item())) == saved["report"],
                "raw report differs",
            )
            n = round(0.1 / dt)
            active = (np.arange(n) >= round(0.02 / dt)) & (np.arange(n) < round(0.08 / dt))
            coeff = p.get("coeffs", {})
            gl = coeff.get("G", 1) / p["R_input"]
            scale = gl * (p["th_inf"] * coeff.get("th_inf", 1) - p["El"])
            recipes = {
                "bias": np.full(n, arm["bias_scale_multiple"] * scale),
                "excitation": active.astype(float) * arm["excitation_leak_multiple"] * gl,
                "inhibition": active.astype(float) * arm["inhibition_leak_multiple"] * gl,
            }
            for key, value in recipes.items():
                require(np.array_equal(arrays[key], value), "input recipe changed")
            for backend in ("native0", "native1", "candidate0", "candidate1", "current_only"):
                if backend + "_voltage" not in arrays.files:
                    continue
                for field in ("voltage", "threshold"):
                    require(arrays[backend + "_" + field].shape == (n,), "trace length changed")
                require(
                    arrays[backend + "_AScurrents"].shape == (n, len(p["init_AScurrents"])),
                    "ASC dimensions changed",
                )
                indices = arrays[backend + "_spike_time_steps"]
                require(
                    indices.ndim == 1 and np.all(indices == np.floor(indices)),
                    "noninteger spike indices",
                )
                require(np.all((indices >= 0) & (indices < n)), "spike indices outside trace")
                for field in FIELDS[3:]:
                    require(
                        arrays[backend + "_" + field].shape == indices.shape,
                        "spike field dimensions changed",
                    )
                require(
                    np.array_equal(arrays[backend + "_grid_spike_times"], indices * dt),
                    "grid timing inconsistent",
                )
            result = verify_report(
                arrays, saved["report"], null=arm["name"] == "zero-conductance-null"
            )
            passed += result
            if not result:
                failures.append(number)
    require(terminal["passed_cases"] == passed, "terminal pass count differs")
    print(
        json.dumps(
            {
                "verified_cases": 90,
                "attempts": 360,
                "passed_cases": passed,
                "nonpassing_case_indices": failures,
                "stop_flags_independently_reexecuted": False,
                "biological_validation": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
