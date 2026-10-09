"""Sealed, checkpointed native/independent GLIF comparison; no experimental responses."""

from __future__ import annotations

import ast
import copy
import fcntl
import hashlib
import json
import logging
import platform
import sys
import types
from pathlib import Path

import numpy as np
import yaml
from collect_sst_vip_native_resources import save_yaml, sha

from smart_robustness.validation.native_glif_parity import compare, exact_repeat, fixtures, stimuli
from smart_robustness.validation.native_glif_state_machine import simulate_glif
from smart_robustness.validation.sst_vip_glif_source import numerical_module_ast

REGISTRATION = Path("docs/validation-results/post2008-sst-vip-glif-parity-registration-1061.yaml")
SEAL = Path("docs/validation-results/post2008-sst-vip-glif-parity-execution-seal-1063.yaml")


def load_reference(reg):
    # Hash checks and execution authorization occur before this function is called.
    methods = types.ModuleType("glif_neuron_methods")
    methods.__file__ = reg["native_methods_source"]
    previous = sys.modules.get("glif_neuron_methods")
    try:
        exec(  # noqa: S102 - explicitly sealed, hash-pinned native numerical source
            compile(Path(methods.__file__).read_text(), methods.__file__, "exec"), methods.__dict__
        )
        sys.modules["glif_neuron_methods"] = methods
        tree, audit = numerical_module_ast(Path(reg["native_neuron_source"]).read_text())
        namespace = {"__name__": "_sealed_glif_reference", "__file__": reg["native_neuron_source"]}
        exec(  # noqa: S102 - audited AST preserves pinned numerical definitions
            compile(ast.fix_missing_locations(tree), reg["native_neuron_source"], "exec"), namespace
        )
    finally:
        if previous is None:
            sys.modules.pop("glif_neuron_methods", None)
        else:
            sys.modules["glif_neuron_methods"] = previous
    klass = namespace["GlifNeuron"]

    def run(parameters, stimulus):
        neuron = klass.from_dict(copy.deepcopy(parameters))
        original_reset = neuron.reset
        flags = []

        def observed_reset(*args):
            output = original_reset(*args)
            flags.append(bool(output[-1]))
            return output

        # Observation-only wrapper returns the exact native reset tuple.
        neuron.reset = observed_reset
        output = neuron.run(stimulus.copy())
        return output, {"bad_reset_stop": any(flags)}

    audit.update(native_source_executed=True, reset_observer_changes_numerics=False)
    audit["omitted_imports"] = [list(pair) for pair in audit["omitted_imports"]]
    return run, audit


def evaluate_case(case, stimulus, reference):
    outputs, states, errors = {}, {}, {}
    for name, function in (
        ("native0", reference),
        ("native1", reference),
        ("candidate0", simulate_glif),
        ("candidate1", simulate_glif),
    ):
        try:
            outputs[name], states[name] = function(
                copy.deepcopy(case["parameters"]), stimulus.copy()
            )
        except Exception as exc:  # noqa: BLE001 - retain every native failure as evidence
            # All failures are evidence, including uncommon native exceptions.
            errors[name] = {"type": type(exc).__name__, "message": str(exc)}
    report = {"errors": errors, "states": states, "passed": False}
    if not errors:
        report["comparison"] = compare(outputs["native0"], outputs["candidate0"])
        report["native_exact_repeat"] = exact_repeat(outputs["native0"], outputs["native1"])
        report["candidate_exact_repeat"] = exact_repeat(
            outputs["candidate0"], outputs["candidate1"]
        )
        report["bad_reset_stop"] = any(s["bad_reset_stop"] for s in states.values())
        report["passed"] = (
            report["comparison"]["passed"]
            and report["native_exact_repeat"]
            and report["candidate_exact_repeat"]
            and not report["bad_reset_stop"]
        )
    return report, {
        name + "_" + key: value for name, data in outputs.items() for key, value in data.items()
    }


def run_case(number, case, label, stimulus, *, directory, context, reference):
    sidecar = directory / f"case-{number:04d}.yaml"
    raw = directory / f"case-{number:04d}.npz"
    identity = {
        "case": case,
        "stimulus": label,
        "stimulus_sha256": hashlib.sha256(stimulus.tobytes()).hexdigest(),
    }
    if sidecar.exists():
        saved = yaml.safe_load(sidecar.read_text())
        if (
            saved["context"] != context
            or saved["identity"] != identity
            or saved["raw_sha256"] != sha(raw)
        ):
            raise ValueError("parity checkpoint changed")
        with np.load(raw, allow_pickle=False) as arrays:
            if json.loads(str(arrays["report_json"].item())) != saved[
                "report"
            ] or not np.array_equal(arrays["stimulus"], stimulus):
                raise ValueError("checkpoint report or stimulus differs from sealed raw arrays")
        return saved
    if raw.exists():
        raise ValueError("orphan parity arrays require assessment, not overwrite")
    report, arrays = evaluate_case(case, stimulus, reference)
    with raw.open("xb") as handle:
        np.savez_compressed(
            handle,
            stimulus=stimulus,
            report_json=np.asarray(json.dumps(report, sort_keys=True)),
            **arrays,
        )
    result = {
        "context": context,
        "identity": identity,
        "raw_path": str(raw),
        "raw_sha256": sha(raw),
        "report": report,
    }
    save_yaml(sidecar, result)
    return result


def main():
    logging.disable(logging.INFO)
    reg = yaml.safe_load(REGISTRATION.read_text())
    seal = yaml.safe_load(SEAL.read_text())
    if (
        seal["registration_sha256"] != sha(REGISTRATION)
        or seal["native_reference_execution_authorized"] is not True
    ):
        raise ValueError("missing or changed native execution seal")
    for path, expected in seal["implementation_sha256"].items():
        if sha(Path(path)) != expected:
            raise ValueError("sealed implementation changed")
    for key in ("parent", "parameter_manifest"):
        if sha(Path(reg[key])) != reg[key + "_sha256"]:
            raise ValueError("source lineage changed")
    for key in ("native_neuron", "native_methods"):
        if sha(Path(reg[key + "_source"])) != reg[key + "_sha256"]:
            raise ValueError("pinned native source changed")
    if (
        reg["network_execution_authorized"] is not False
        or reg["parameter_fitting_authorized"] is not False
    ):
        raise ValueError("synthetic-only authorization changed")
    manifest = yaml.safe_load(Path(reg["parameter_manifest"]).read_text())
    cases = []
    for record in manifest["records"]:
        name = record["source"]["template"]["name"]
        if name.startswith("Biophysical"):
            continue
        if sha(Path(record["raw_path"])) != record["raw_sha256"]:
            raise ValueError("GLIF parameter bytes changed")
        p = json.loads(Path(record["raw_path"]).read_text())
        cases.append(
            {"identity": str(record["source"]["model_id"]), "family": name, "parameters": p}
        )
    if len(cases) != 227 or len({c["identity"] for c in cases}) != 227:
        raise ValueError("registered GLIF cohort changed")
    cases = sorted(cases, key=lambda c: int(c["identity"])) + fixtures()
    if len(cases) != 237:
        raise ValueError("controlled fixture cohort changed")
    context = {"registration_sha256": sha(REGISTRATION), "seal_sha256": sha(SEAL)}
    directory = Path(reg["output_directory"])
    directory.mkdir(exist_ok=True)
    with (directory / "collector.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        reference, audit = load_reference(reg)
        records = []
        for case in cases:
            for label, stimulus in stimuli(case["parameters"]).items():
                records.append(
                    run_case(
                        len(records),
                        case,
                        label,
                        stimulus,
                        directory=directory,
                        context=context,
                        reference=reference,
                    )
                )
        if len(records) != 948:
            raise ValueError("registered synthetic case coverage changed")
        result = {
            "schema_version": 1,
            "context": context,
            "native_import_audit": audit,
            "environment": {"python": platform.python_version(), "numpy": np.__version__},
            "records": records,
            "passed_cases": sum(r["report"]["passed"] for r in records),
            "experimental_responses_read": False,
            "network_executed": False,
        }
        final = directory / "manifest.yaml"
        if final.exists():
            if yaml.safe_load(final.read_text()) != result:
                raise ValueError("terminal parity checkpoint changed")
            print("Verified terminal parity checkpoints; no reruns")
        else:
            save_yaml(final, result)
            print(f"{result['passed_cases']}/948 parity cases pass; all failures retained")


if __name__ == "__main__":
    main()
