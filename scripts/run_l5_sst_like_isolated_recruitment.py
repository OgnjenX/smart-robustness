"""Run the sealed, network-blind L5 SST-like fixed-event assay."""

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
from smart_robustness.validation.l5_sst_like_isolated import (
    DELAYS_MS,
    DT_MS,
    RESOURCE_FRACTIONS,
    TOTAL_CONDUCTANCES_NS,
    exact_trace_repeat,
    file_sha256,
    load_trace,
    numerical_gate,
    save_trace,
    simulate,
    summarize,
)


def checkpoint(path: Path, payload: dict) -> None:
    descriptor, name = tempfile.mkstemp(
        prefix=".l5-sst-like-isolated-checkpoint-", dir=path.parent
    )
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
    if seal["status"] != "sealed-before-isolated-nonzero-outcomes":
        raise ValueError("an isolated nonzero execution seal is required")
    for filename, digest in seal["files"].items():
        if file_sha256(Path(filename)) != digest:
            raise ValueError(f"sealed file changed: {filename}")
    if "scripts/run_l5_sst_like_isolated_recruitment.py" not in seal["files"]:
        raise ValueError("runner is missing from execution seal")
    return seal


def _label(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def run_key(
    *, delay_ms: float, resource_fraction: float, dt_ms: float, repetition: int
) -> str:
    return (
        f"delay{_label(delay_ms)}-resource{_label(resource_fraction)}-"
        f"dt{_label(dt_ms)}-repeat{repetition}"
    )


def registered_runs():
    for delay_ms in DELAYS_MS:
        for resource_fraction, total_nS in zip(
            RESOURCE_FRACTIONS, TOTAL_CONDUCTANCES_NS, strict=True
        ):
            for dt_ms in DT_MS:
                for repetition in range(2):
                    yield {
                        "delay_ms": delay_ms,
                        "resource_fraction": resource_fraction,
                        "total_conductance_nS": total_nS,
                        "dt_ms": dt_ms,
                        "repetition": repetition,
                    }


def execute(output: Path, seal_path: Path) -> None:
    import brian2

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
    }
    payload = {
        "schema_version": 1,
        "status": "running-l5-sst-like-isolated-recruitment",
        "identity": identity,
        "runs": {},
        "assessment": None,
        "smart_network_constructed": False,
        "smart_behavior_observed": False,
        "nonzero_network_execution_authorized": False,
        "frozen_baseline_modified": False,
    }
    if output.exists():
        payload = yaml.safe_load(output.read_text())
        if payload["identity"] != identity:
            raise ValueError("checkpoint does not match sealed implementation and environment")
    raw_dir = output.with_suffix("")
    raw_dir.mkdir(parents=True, exist_ok=True)

    expected = {run_key(**{k: run[k] for k in ("delay_ms", "resource_fraction", "dt_ms", "repetition")}) for run in registered_runs()}
    if set(payload["runs"]) - expected:
        raise ValueError("checkpoint contains unregistered runs")

    def recover(entry: dict) -> dict[str, np.ndarray]:
        path = (raw_dir / entry["filename"]).resolve()
        if path.parent != raw_dir.resolve():
            raise ValueError("trace path escapes result directory")
        return load_trace(path, entry["sha256"])

    for key, entry in payload["runs"].items():
        expected_key = run_key(
            delay_ms=entry["delay_ms"],
            resource_fraction=entry["resource_fraction"],
            dt_ms=entry["dt_ms"],
            repetition=entry["repetition"],
        )
        if key != expected_key:
            raise ValueError("checkpoint run identity mismatch")
        arrays = recover(entry)
        if summarize(arrays) != entry["metrics"]:
            raise ValueError("checkpoint metrics do not match raw trace")
    if payload["status"] == "completed-l5-sst-like-isolated-recruitment":
        if set(payload["runs"]) != expected:
            raise ValueError("incomplete checkpoint marked completed")
        return
    if payload["status"].startswith("stopped-"):
        return

    for run in registered_runs():
        key_fields = {
            name: run[name]
            for name in ("delay_ms", "resource_fraction", "dt_ms", "repetition")
        }
        key = run_key(**key_fields)
        if key in payload["runs"]:
            continue
        path = raw_dir / f"{key}.npz"
        if path.exists():
            raise ValueError(f"uncheckpointed raw result requires review: {path}")
        print(f"Running {key}", flush=True)
        arrays = simulate(
            baseline=baseline,
            total_conductance_nS=run["total_conductance_nS"],
            delay_ms=run["delay_ms"],
            dt_ms=run["dt_ms"],
        )
        digest = save_trace(path, arrays)
        metrics = summarize(arrays)
        entry = {
            "filename": path.name,
            "sha256": digest,
            **run,
            "metrics": metrics,
        }
        payload["runs"][key] = entry
        if not metrics["per_run_gates_pass"]:
            payload["status"] = "stopped-isolated-engineering-gate-failed"
            payload["assessment"] = {
                "failed_run": key,
                "isolated_promotion_gates_pass": False,
                "classification": "engineering_stop",
            }
            checkpoint(output, payload)
            return
        checkpoint(output, payload)
        del arrays

    exact_repeats = {}
    convergence = {}
    for delay_ms in DELAYS_MS:
        for resource_fraction in RESOURCE_FRACTIONS:
            point = f"delay{_label(delay_ms)}-resource{_label(resource_fraction)}"
            for dt_ms in DT_MS:
                base = run_key(
                    delay_ms=delay_ms,
                    resource_fraction=resource_fraction,
                    dt_ms=dt_ms,
                    repetition=0,
                )
                other = run_key(
                    delay_ms=delay_ms,
                    resource_fraction=resource_fraction,
                    dt_ms=dt_ms,
                    repetition=1,
                )
                exact_repeats[f"{point}-dt{_label(dt_ms)}"] = exact_trace_repeat(
                    recover(payload["runs"][base]), recover(payload["runs"][other])
                )
            coarse = payload["runs"][
                run_key(
                    delay_ms=delay_ms,
                    resource_fraction=resource_fraction,
                    dt_ms=DT_MS[0],
                    repetition=0,
                )
            ]["metrics"]
            fine = payload["runs"][
                run_key(
                    delay_ms=delay_ms,
                    resource_fraction=resource_fraction,
                    dt_ms=DT_MS[1],
                    repetition=0,
                )
            ]["metrics"]
            convergence[point] = numerical_gate(coarse, fine)
    all_run_gates = all(
        entry["metrics"]["per_run_gates_pass"] for entry in payload["runs"].values()
    )
    promoted = bool(
        all_run_gates
        and all(exact_repeats.values())
        and all(gate["pass"] for gate in convergence.values())
    )
    payload["assessment"] = {
        "completed_runs": len(payload["runs"]),
        "all_per_run_gates_pass": all_run_gates,
        "exact_raw_repeats": exact_repeats,
        "numerical_convergence": convergence,
        "isolated_promotion_gates_pass": promoted,
        "classification": "isolated_engineering_pass" if promoted else "engineering_stop",
        "boundary": "network execution requires a separate assessment, registration, and seal",
    }
    payload["status"] = "completed-l5-sst-like-isolated-recruitment"
    checkpoint(output, payload)


def main() -> None:
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
