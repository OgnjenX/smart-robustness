"""Execute design 949 only after a separately committed implementation seal."""

from __future__ import annotations

import argparse
import fcntl
import os
import platform
import tempfile
from pathlib import Path

import numpy as np
import yaml

from smart_robustness.baseline import load_frozen_classic_baseline
from smart_robustness.models.active_apical_integration import ARMS
from smart_robustness.validation.active_apical_isolated import ARRIVALS_MS, assay_cases
from smart_robustness.validation.active_apical_metrics import (
    mechanism_points,
    numerical_gate,
    summarize_response,
)
from smart_robustness.validation.active_apical_recording import (
    exact_trace_repeat,
    file_sha256,
    load_trace,
    rest_assessment,
    save_trace,
    simulate,
)

STEPS = (0.01, 0.005)


def checkpoint(path: Path, payload: dict) -> None:
    descriptor, name = tempfile.mkstemp(prefix=".apical-checkpoint-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            yaml.safe_dump(payload, stream, sort_keys=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def verify_seal(path: Path) -> dict:
    seal = yaml.safe_load(path.read_text())
    if seal["status"] != "sealed-before-isolated-outcomes":
        raise ValueError("an execution seal is required")
    for filename, digest in seal["files"].items():
        if file_sha256(Path(filename)) != digest:
            raise ValueError(f"sealed file changed: {filename}")
    if "scripts/run_active_apical_isolated.py" not in seal["files"]:
        raise ValueError("runner is missing from execution seal")
    return seal


def execute(output: Path, seal_path: Path) -> None:
    import brian2
    import scipy

    seal = verify_seal(seal_path)
    baseline = load_frozen_classic_baseline(seal["baseline_manifest"])
    identity = {
        "seal_sha256": file_sha256(seal_path),
        "baseline_manifest_fingerprint": baseline.manifest_fingerprint,
        "runtime_fingerprint": baseline.runtime_fingerprint,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "brian2": brian2.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
    payload = {
        "schema_version": 1,
        "status": "running-isolated-apical",
        "identity": identity,
        "rest": None,
        "runs": {},
        "network_execution_authorized": False,
    }
    if output.exists():
        payload = yaml.safe_load(output.read_text())
        if payload["identity"] != identity:
            raise ValueError("checkpoint does not match sealed implementation and environment")
    raw_dir = output.with_suffix("")
    raw_dir.mkdir(parents=True, exist_ok=True)

    def recover(entry):
        path = raw_dir / entry["filename"]
        if path.parent.resolve() != raw_dir.resolve():
            raise ValueError("trace path escapes result directory")
        return load_trace(path, entry["sha256"])

    # Verify all existing completed raw records, even when returning a terminal
    # checkpoint. Do not resume based only on a status flag.
    if payload["rest"]:
        restored = recover(payload["rest"])
        if rest_assessment(restored, 0.01) != payload["rest"]["assessment"]:
            raise ValueError("rest checkpoint does not match raw measurement")
    expected_keys = {
        f"{arm}-dt{dt:g}-repeat{rep}" for arm in ARMS for dt in STEPS for rep in range(2)
    }
    if set(payload["runs"]) - expected_keys:
        raise ValueError("checkpoint contains unregistered runs")
    for key, entry in payload["runs"].items():
        if key != f"{entry['arm']}-dt{entry['dt_ms']:g}-repeat{entry['repetition']}":
            raise ValueError("checkpoint run identity mismatch")
        restored = recover(entry)
        numeric = {k: v for k, v in restored.items() if np.issubdtype(v.dtype, np.number)}
        if rest_assessment(numeric, entry["dt_ms"]) != entry["rest"]:
            raise ValueError("run rest checkpoint does not match raw measurement")
        if "metrics" in entry and summarize_response(restored, entry["dt_ms"]) != entry["metrics"]:
            raise ValueError("run metrics do not match raw measurement")
    if payload["status"] == "completed-isolated-apical" and set(payload["runs"]) != expected_keys:
        raise ValueError("incomplete checkpoint marked completed")
    if payload["status"].startswith(("completed-", "stopped-")):
        return

    if payload["rest"] is None:
        print("Running pre-stimulus reference rest", flush=True)
        arrays = simulate(
            baseline=baseline,
            arm="classic_ampa",
            resting_distal_mV=-65.0,
            dt_ms=0.01,
            rest_only=True,
        )
        path = raw_dir / "reference-rest.npz"
        digest = save_trace(path, arrays)
        payload["rest"] = {
            "filename": path.name,
            "sha256": digest,
            "assessment": rest_assessment(arrays, 0.01),
        }
        checkpoint(output, payload)
    rest = payload["rest"]["assessment"]
    if not rest["pass"]:
        payload["status"] = "stopped-rest-prerequisite-failed"
        checkpoint(output, payload)
        return
    fixed_rest = rest["resting_distal_mV"]
    for arm in ARMS:
        for dt in STEPS:
            for repetition in range(2):
                key = f"{arm}-dt{dt:g}-repeat{repetition}"
                if key in payload["runs"]:
                    continue
                path = raw_dir / f"{key}.npz"
                if path.exists():
                    raise ValueError(f"uncheckpointed raw result requires review: {path}")
                print(f"Running {key}", flush=True)
                arrays = simulate(
                    baseline=baseline, arm=arm, resting_distal_mV=fixed_rest, dt_ms=dt
                )
                arrays["arrivals_ms"] = ARRIVALS_MS.copy()
                arrays["case_drive_factor"] = np.array([c.drive_factor for c in assay_cases()])
                arrays["case_input_kind"] = np.array([c.input_kind for c in assay_cases()])
                arrays["fixed_rest_mV"] = np.array(fixed_rest)
                digest = save_trace(path, arrays)
                # Metadata strings are not numerical states.
                numeric = {k: v for k, v in arrays.items() if np.issubdtype(v.dtype, np.number)}
                entry = {
                    "filename": path.name,
                    "sha256": digest,
                    "arm": arm,
                    "dt_ms": dt,
                    "repetition": repetition,
                    "rest": rest_assessment(numeric, dt),
                }
                payload["runs"][key] = entry
                if not entry["rest"]["all_recorded_states_finite"]:
                    payload["status"] = "stopped-nonfinite-recording"
                    checkpoint(output, payload)
                    return
                entry["metrics"] = summarize_response(arrays, dt)
                checkpoint(output, payload)
                del arrays
    exact = {}
    convergence = {}
    for arm in ARMS:
        for dt in STEPS:
            key = f"{arm}-dt{dt:g}"
            exact[key] = exact_trace_repeat(
                recover(payload["runs"][key + "-repeat0"]),
                recover(payload["runs"][key + "-repeat1"]),
            )
        convergence[arm] = numerical_gate(
            payload["runs"][f"{arm}-dt0.01-repeat0"]["metrics"],
            payload["runs"][f"{arm}-dt0.005-repeat0"]["metrics"],
        )
    points = {
        str(dt): mechanism_points(
            payload["runs"][f"mixed_active_block-dt{dt:g}-repeat0"]["metrics"],
            payload["runs"][f"mixed_rest_block-dt{dt:g}-repeat0"]["metrics"],
        )
        for dt in STEPS
    }
    common = sorted(set(points["0.01"]) & set(points["0.005"]))
    rest_pass = all(e["rest"]["pass"] for e in payload["runs"].values())
    payload["assessment"] = {
        "exact_raw_repeats": exact,
        "numerical_gates": convergence,
        "mechanism_points_by_dt": points,
        "shared_mechanism_points": common,
        "all_run_rest_checks_pass": rest_pass,
        "isolated_promotion_gates_pass": bool(
            all(exact.values())
            and rest_pass
            and common
            and all(g["pass"] for g in convergence.values())
        ),
        "boundary": "engineering seal and independent assessment required before network registration",
    }
    payload["status"] = "completed-isolated-apical"
    checkpoint(output, payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seal", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(output.suffix + ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another runner owns this result") from error
        execute(output, Path(args.seal))


if __name__ == "__main__":
    main()
