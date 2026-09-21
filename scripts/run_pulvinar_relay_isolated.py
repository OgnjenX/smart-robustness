"""Execute the sealed isolated SMART V2 relay assay."""

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
from smart_robustness.validation.active_apical_recording import (
    exact_trace_repeat,
    file_sha256,
    load_trace,
    save_trace,
)
from smart_robustness.validation.pulvinar_relay_recording import (
    numerical_gate,
    simulate,
    summarize,
)

FREQUENCIES = (0.5, 2, 5, 10, 20)
STEPS = (0.01, 0.005)
REPETITIONS = 2


def checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".pulvinar-relay-", dir=path.parent)
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
    if seal["status"] != "sealed-before-isolated-relay-outcomes":
        raise ValueError("an isolated relay execution seal is required")
    for filename, digest in seal["files"].items():
        if file_sha256(Path(filename)) != digest:
            raise ValueError(f"sealed file changed: {filename}")
    if "scripts/run_pulvinar_relay_isolated.py" not in seal["files"]:
        raise ValueError("runner is missing from execution seal")
    return seal


def expected_run_keys() -> set[str]:
    return {
        f"frequency{frequency:g}-dt{dt:g}-repeat{repetition}"
        for frequency in FREQUENCIES
        for dt in STEPS
        for repetition in range(REPETITIONS)
    }


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
        "status": "running-isolated-pulvinar-relay",
        "identity": identity,
        "runs": {},
        "numerical_gates": {},
        "network_execution_authorized": False,
    }
    if output.exists():
        payload = yaml.safe_load(output.read_text())
        if payload["identity"] != identity:
            raise ValueError("checkpoint does not match sealed implementation")
    raw_dir = output.with_suffix("")
    raw_dir.mkdir(parents=True, exist_ok=True)

    def recover(entry):
        path = raw_dir / entry["filename"]
        if path.parent.resolve() != raw_dir.resolve():
            raise ValueError("trace path escapes result directory")
        return load_trace(path, entry["sha256"])

    expected = expected_run_keys()
    if set(payload["runs"]) - expected:
        raise ValueError("checkpoint contains unregistered runs")
    for key, entry in payload["runs"].items():
        expected_key = (
            f"frequency{entry['frequency_hz']:g}-dt{entry['dt_ms']:g}"
            f"-repeat{entry['repetition']}"
        )
        if key != expected_key:
            raise ValueError("checkpoint run identity mismatch")
        arrays = recover(entry)
        if summarize(arrays) != entry["summary"]:
            raise ValueError("checkpoint summary differs from raw trace")
    if payload["status"] == "completed-isolated-pulvinar-relay" and set(payload["runs"]) != expected:
        raise ValueError("incomplete checkpoint marked completed")
    if payload["status"] == "completed-isolated-pulvinar-relay":
        return

    for frequency in FREQUENCIES:
        for dt_ms in STEPS:
            repeat_arrays = []
            for repetition in range(REPETITIONS):
                key = f"frequency{frequency:g}-dt{dt_ms:g}-repeat{repetition}"
                if key not in payload["runs"]:
                    arrays = simulate(
                        baseline=baseline, frequency_hz=frequency, dt_ms=dt_ms,
                    )
                    summary = summarize(arrays)
                    filename = f"{key}.npz"
                    digest = save_trace(raw_dir / filename, arrays)
                    payload["runs"][key] = {
                        "frequency_hz": frequency,
                        "dt_ms": dt_ms,
                        "repetition": repetition,
                        "filename": filename,
                        "sha256": digest,
                        "summary": summary,
                    }
                    checkpoint(output, payload)
                repeat_arrays.append(recover(payload["runs"][key]))
            if not exact_trace_repeat(*repeat_arrays):
                raise ValueError(f"exact repeat failed at {frequency:g} Hz, dt={dt_ms:g} ms")
        coarse = payload["runs"][f"frequency{frequency:g}-dt0.01-repeat0"]["summary"]
        fine = payload["runs"][f"frequency{frequency:g}-dt0.005-repeat0"]["summary"]
        payload["numerical_gates"][str(frequency)] = numerical_gate(coarse, fine)
        checkpoint(output, payload)
    payload["status"] = "completed-isolated-pulvinar-relay"
    checkpoint(output, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--seal", type=Path,
        default=Path("docs/validation-results/post2008-pulvinar-relay-isolated-seal-963.yaml"),
    )
    args = parser.parse_args()
    lock_path = args.output.with_suffix(args.output.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("an identical isolated relay runner is already active") from error
        execute(args.output, args.seal)


if __name__ == "__main__":
    main()
