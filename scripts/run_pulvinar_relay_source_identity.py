"""Execute the sealed rest-only V2 relay source-identity diagnostic."""

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
from smart_robustness.validation.pulvinar_relay_identity import (
    simulate_rest,
    source_faithful_numerical_gate,
    summarize_rest,
)

STEPS = (0.01, 0.005)


def checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".relay-identity-", dir=path.parent)
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
    if seal["status"] != "sealed-before-source-identity-outcomes":
        raise ValueError("a source-identity execution seal is required")
    for filename, digest in seal["files"].items():
        if file_sha256(Path(filename)) != digest:
            raise ValueError(f"sealed file changed: {filename}")
    return seal


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
        "status": "running-source-identity-rest",
        "identity": identity,
        "runs": {},
        "numerical_gate": None,
        "network_execution_authorized": False,
    }
    if output.exists():
        payload = yaml.safe_load(output.read_text())
        if payload["identity"] != identity:
            raise ValueError("checkpoint identity mismatch")
    raw_dir = output.with_suffix("")
    raw_dir.mkdir(parents=True, exist_ok=True)

    def recover(entry):
        path = raw_dir / entry["filename"]
        if path.parent.resolve() != raw_dir.resolve():
            raise ValueError("trace path escapes result directory")
        return load_trace(path, entry["sha256"])

    expected = {f"dt{dt:g}-repeat{rep}" for dt in STEPS for rep in range(2)}
    if set(payload["runs"]) - expected:
        raise ValueError("checkpoint contains unregistered run")
    for key, entry in payload["runs"].items():
        if key != f"dt{entry['dt_ms']:g}-repeat{entry['repetition']}":
            raise ValueError("checkpoint run identity mismatch")
        if summarize_rest(recover(entry)) != entry["summary"]:
            raise ValueError("checkpoint summary differs from raw trace")
    if payload["status"] == "completed-source-identity-rest":
        if set(payload["runs"]) != expected:
            raise ValueError("incomplete checkpoint marked complete")
        return
    for dt_ms in STEPS:
        repeats = []
        for repetition in range(2):
            key = f"dt{dt_ms:g}-repeat{repetition}"
            if key not in payload["runs"]:
                arrays = simulate_rest(baseline=baseline, dt_ms=dt_ms)
                summary = summarize_rest(arrays)
                filename = f"{key}.npz"
                payload["runs"][key] = {
                    "dt_ms": dt_ms,
                    "repetition": repetition,
                    "filename": filename,
                    "sha256": save_trace(raw_dir / filename, arrays),
                    "summary": summary,
                }
                checkpoint(output, payload)
            repeats.append(recover(payload["runs"][key]))
        if not exact_trace_repeat(*repeats):
            raise ValueError(f"exact repeat failed at dt={dt_ms:g} ms")
    payload["numerical_gate"] = source_faithful_numerical_gate(
        payload["runs"]["dt0.01-repeat0"]["summary"],
        payload["runs"]["dt0.005-repeat0"]["summary"],
    )
    payload["status"] = "completed-source-identity-rest"
    checkpoint(output, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--seal", type=Path,
        default=Path("docs/validation-results/post2008-pulvinar-relay-source-identity-seal-967.yaml"),
    )
    args = parser.parse_args()
    lock_path = args.output.with_suffix(args.output.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("an identical source-identity runner is active") from error
        execute(args.output, args.seal)


if __name__ == "__main__":
    main()
