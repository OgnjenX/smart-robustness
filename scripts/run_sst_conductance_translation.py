"""Sealed 90-case engineering matrix; never fit parameters or run SMART."""

from __future__ import annotations

import ast
import copy
import fcntl
import hashlib
import json
import logging
import sys
import types
from pathlib import Path

import numpy as np
import yaml

from smart_robustness.validation.conductance_glif_state_machine import simulate_conductance_glif
from smart_robustness.validation.glif_conductance_input import run_extended_native
from smart_robustness.validation.native_glif_parity import compare, exact_repeat
from smart_robustness.validation.native_glif_state_machine import simulate_glif
from smart_robustness.validation.sst_vip_glif_source import numerical_module_ast

REGISTRATION = Path(
    "docs/validation-results/post2008-sst-conductance-translation-registration-1080.yaml"
)
SEAL = Path("docs/validation-results/post2008-sst-conductance-translation-execution-seal-1083.yaml")
ROOT = Path("results/sst-conductance-translation-1084")
NATIVE_REGISTRATION = Path(
    "docs/validation-results/post2008-sst-vip-glif-parity-registration-1061.yaml"
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def save(path, value):
    with path.open("x") as handle:
        yaml.safe_dump(value, handle, sort_keys=False)


def inputs(parameters, arm):
    dt = parameters["dt"]
    samples = round(0.1 / dt)
    require(dt in (0.00005, 0.000005), "unregistered timestep")
    # Integer endpoints avoid floating-point boundary shifts of half-open windows.
    envelope = np.zeros(samples)
    envelope[round(0.02 / dt) : round(0.08 / dt)] = 1
    coeffs = parameters.get("coeffs", {})
    leak = coeffs.get("G", 1) / parameters["R_input"]
    scale = leak * (parameters["th_inf"] * coeffs.get("th_inf", 1) - parameters["El"])
    return (
        np.full(samples, arm["bias_scale_multiple"] * scale),
        envelope * arm["excitation_leak_multiple"] * leak,
        envelope * arm["inhibition_leak_multiple"] * leak,
    )


def prepare(registration=REGISTRATION, seal_path=SEAL):
    """All authorization and source checks precede numerical source execution."""
    reg = yaml.safe_load(registration.read_text())
    seal = yaml.safe_load(seal_path.read_text())
    require(seal["registration_sha256"] == digest(registration), "registration changed")
    require(seal["synthetic_execution_authorized"] is True, "execution not authorized")
    require(seal["network_execution_authorized"] is False, "network scope changed")
    require(seal["parameter_fitting_authorized"] is False, "fitting scope changed")
    for path, expected in seal["implementation_sha256"].items():
        require(digest(path) == expected, "sealed implementation changed")
    require(digest(reg["parent"]) == reg["parent_sha256"], "candidate lineage changed")
    parent = yaml.safe_load(Path(reg["parent"]).read_text())
    for key in ("cell_inventory", "parameter_manifest"):
        require(digest(parent[key]) == parent[key + "_sha256"], "source lineage changed")
    require(
        seal["native_registration_sha256"] == digest(NATIVE_REGISTRATION),
        "native registration changed",
    )
    native = yaml.safe_load(NATIVE_REGISTRATION.read_text())
    for key in ("native_neuron", "native_methods"):
        require(digest(native[key + "_source"]) == native[key + "_sha256"], "native source changed")
    manifest = yaml.safe_load(Path(parent["parameter_manifest"]).read_text())
    maps = {}
    for record in manifest["records"]:
        model = record["source"]["model_id"]
        if model not in reg["cohort"]["model_ids"]:
            continue
        require(model not in maps, "duplicate source model")
        require(digest(record["raw_path"]) == record["raw_sha256"], "parameter bytes changed")
        maps[model] = json.loads(Path(record["raw_path"]).read_text())
    require(set(maps) == set(reg["cohort"]["model_ids"]), "model coverage changed")
    protocol = reg["synthetic_protocol"]
    require(protocol["duration_seconds"] == 0.1, "duration changed")
    require(protocol["dt_seconds"] == [0.00005, 0.000005], "dt arms changed")
    require(
        protocol["excitation_reversal_volts"] == 0
        and protocol["inhibition_reversal_volts"] == -0.08,
        "reversals changed",
    )
    require(len(protocol["arms"]) == 9, "arm count changed")
    cases = []
    for model in reg["cohort"]["model_ids"]:
        for dt in protocol["dt_seconds"]:
            for arm in protocol["arms"]:
                p = copy.deepcopy(maps[model])
                p["dt"] = dt
                cases.append({"model_id": model, "dt": dt, "arm": arm, "parameters": p})
    require(len(cases) == 90, "case coverage changed")
    return (
        cases,
        native,
        {"registration_sha256": digest(registration), "seal_sha256": digest(seal_path)},
    )


def load_native(native):
    methods = types.ModuleType("glif_neuron_methods")
    methods.__file__ = native["native_methods_source"]
    previous = sys.modules.get("glif_neuron_methods")
    try:
        exec(  # noqa: S102 - execution only after explicit source/hash authorization
            compile(Path(methods.__file__).read_text(), methods.__file__, "exec"), methods.__dict__
        )
        sys.modules["glif_neuron_methods"] = methods
        tree, _ = numerical_module_ast(Path(native["native_neuron_source"]).read_text())
        namespace = {"__name__": "_sealed_conductance_reference"}
        exec(  # noqa: S102 - audited pinned numerical AST, never serialization
            compile(ast.fix_missing_locations(tree), native["native_neuron_source"], "exec"),
            namespace,
        )
    finally:
        if previous is None:
            sys.modules.pop("glif_neuron_methods", None)
        else:
            sys.modules["glif_neuron_methods"] = previous
    return namespace["GlifNeuron"]


def evaluate(case, native_class):
    p = case["parameters"]
    bias, ge, gi = inputs(p, case["arm"])
    outputs, states, errors = {}, {}, {}
    for name in ("native0", "native1", "candidate0", "candidate1"):
        try:
            q = copy.deepcopy(p)
            if name.startswith("native"):
                outputs[name], states[name] = run_extended_native(
                    native_class.from_dict(q), q, bias, ge, gi, e_exc=0, e_inh=-0.08
                )
            else:
                outputs[name], states[name] = simulate_conductance_glif(
                    q, bias, ge, gi, e_exc=0, e_inh=-0.08
                )
        except Exception as exc:  # noqa: BLE001 - preserve every registered failure
            errors[name] = {"type": type(exc).__name__, "message": str(exc)}
    report = {"errors": errors, "states": states, "passed": False}
    if not errors:
        report["comparison"] = compare(outputs["native0"], outputs["candidate0"])
        report["native_exact_repeat"] = exact_repeat(outputs["native0"], outputs["native1"])
        report["candidate_exact_repeat"] = exact_repeat(
            outputs["candidate0"], outputs["candidate1"]
        )
        report["bad_reset_stop"] = any(s["bad_reset_stop"] for s in states.values())
        report["null_equivalence"] = True
        if case["arm"]["name"] == "zero-conductance-null":
            try:
                outputs["current_only"], _ = simulate_glif(copy.deepcopy(p), bias.copy())
                report["null_equivalence"] = exact_repeat(
                    outputs["current_only"], outputs["candidate0"]
                )
            except Exception as exc:  # noqa: BLE001 - retain diagnostic failures too
                errors["current_only"] = {"type": type(exc).__name__, "message": str(exc)}
                report["null_equivalence"] = False
        report["passed"] = all(
            (
                report["comparison"]["passed"],
                report["native_exact_repeat"],
                report["candidate_exact_repeat"],
                not report["bad_reset_stop"],
                report["null_equivalence"],
            )
        )
    arrays = {
        name + "_" + field: value for name, out in outputs.items() for field, value in out.items()
    }
    arrays.update(bias=bias, excitation=ge, inhibition=gi)
    return report, arrays


def checkpoint(number, case, context, native_class, root=ROOT):
    sidecar, raw = root / f"case-{number:03d}.yaml", root / f"case-{number:03d}.npz"
    if sidecar.exists():
        saved = yaml.safe_load(sidecar.read_text())
        require(
            saved["case"] == case and saved["context"] == context, "checkpoint identity changed"
        )
        require(
            saved["raw_path"] == str(raw) and saved["raw_sha256"] == digest(raw),
            "checkpoint arrays changed",
        )
        with np.load(raw, allow_pickle=False) as arrays:
            require(
                json.loads(str(arrays["report_json"].item())) == saved["report"],
                "checkpoint report changed",
            )
            for key, expected in zip(
                ("bias", "excitation", "inhibition"),
                inputs(case["parameters"], case["arm"]),
                strict=True,
            ):
                require(np.array_equal(arrays[key], expected), "checkpoint stimulus changed")
        return saved
    require(not raw.exists(), "orphan arrays require assessment, not overwrite")
    report, arrays = evaluate(case, native_class)
    with raw.open("xb") as handle:
        np.savez_compressed(
            handle, report_json=np.asarray(json.dumps(report, sort_keys=True)), **arrays
        )
    saved = {
        "case": case,
        "context": context,
        "raw_path": str(raw),
        "raw_sha256": digest(raw),
        "report": report,
    }
    save(sidecar, saved)
    return saved


def main():
    logging.disable(logging.INFO)
    cases, native, context = prepare()
    ROOT.mkdir(exist_ok=True)
    with (ROOT / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        terminal = ROOT / "manifest.yaml"
        if terminal.exists():
            saved = yaml.safe_load(terminal.read_text())
            require(
                saved["context"] == context and len(saved["records"]) == 90,
                "terminal context changed",
            )
            require(
                all((ROOT / f"case-{n:03d}.yaml").exists() for n in range(90)),
                "terminal sidecar missing; do not rerun",
            )
            records = [checkpoint(n, c, context, None) for n, c in enumerate(cases)]
            require(saved["records"] == records, "terminal records changed")
            print("Already terminal; no simulations restarted", flush=True)
            return
        klass = load_native(native)
        records = []
        for number, case in enumerate(cases):
            records.append(checkpoint(number, case, context, klass))
            print(f"Completed {number + 1}/90", flush=True)
        save(
            terminal,
            {
                "context": context,
                "records": records,
                "cases": 90,
                "attempts": 360,
                "passed_cases": sum(r["report"]["passed"] for r in records),
            },
        )


if __name__ == "__main__":
    main()
