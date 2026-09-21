"""Run the sealed isolated inhibitory target-location assay (registration 975)."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import os
import platform
import tempfile
from pathlib import Path

import numpy as np
import yaml

from smart_robustness.baseline import load_frozen_classic_baseline
from smart_robustness.validation.inhibitory_target_location import (
    ARMS,
    DT_MS,
    PROTOCOLS,
    cross_arm_gate,
    exact_trace_repeat,
    numerical_gate,
    simulate,
    summarize,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEAL = (
    ROOT
    / "docs/validation-results/post2008-inhibitory-target-location-isolated-seal-976.yaml"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".inhibitory-target-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            yaml.safe_dump(payload, stream, sort_keys=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def save_trace(path: Path, arrays: dict[str, np.ndarray]) -> str:
    if any(np.asarray(value).dtype.hasobject for value in arrays.values()):
        raise ValueError("object arrays are forbidden")
    descriptor, temporary = tempfile.mkstemp(prefix=".inhibitory-trace-", dir=path.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return file_sha256(path)


def load_trace(path: Path, expected_sha256: str) -> dict[str, np.ndarray]:
    if file_sha256(path) != expected_sha256:
        raise ValueError(f"raw trace fingerprint mismatch: {path}")
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def verify_seal(path: Path) -> dict:
    seal = yaml.safe_load(path.read_text())
    if seal["status"] != "sealed-before-isolated-outcomes":
        raise ValueError("an isolated execution seal is required")
    for relative, expected in seal["files"].items():
        if file_sha256(ROOT / relative) != expected:
            raise ValueError(f"sealed file changed: {relative}")
    runner = "scripts/run_inhibitory_target_location_isolated.py"
    if runner not in seal["files"]:
        raise ValueError("runner is missing from execution seal")
    return seal


def run_key(protocol: str, arm: str, dt_ms: float, repetition: int) -> str:
    return f"{protocol}-{arm}-dt{dt_ms:g}-repeat{repetition}"


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
        "status": "running-inhibitory-target-location-isolated",
        "identity": identity,
        "runs": {},
        "assessment": None,
        "network_execution": False,
        "frozen_baseline_modified": False,
    }
    if output.exists():
        payload = yaml.safe_load(output.read_text())
        if payload["identity"] != identity:
            raise ValueError("checkpoint identity differs from sealed implementation")

    raw_dir = output.with_suffix("")
    raw_dir.mkdir(parents=True, exist_ok=True)
    expected = {
        run_key(protocol, arm, dt_ms, repetition)
        for protocol in PROTOCOLS
        for arm in ARMS
        for dt_ms in DT_MS
        for repetition in range(2)
    }
    if set(payload["runs"]) - expected:
        raise ValueError("checkpoint contains an unregistered condition")

    def recover(entry: dict) -> dict[str, np.ndarray]:
        path = raw_dir / entry["filename"]
        if path.parent.resolve() != raw_dir.resolve():
            raise ValueError("trace path escapes result directory")
        arrays = load_trace(path, entry["sha256"])
        if summarize(arrays) != entry["summary"]:
            raise ValueError("checkpoint summary differs from raw trace")
        return arrays

    for key, entry in payload["runs"].items():
        expected_key = run_key(
            entry["protocol"], entry["arm"], entry["dt_ms"], entry["repetition"]
        )
        if key != expected_key:
            raise ValueError("checkpoint run identity mismatch")
        recover(entry)
    if payload["status"] == "completed-inhibitory-target-location-isolated":
        if set(payload["runs"]) != expected or payload["assessment"] is None:
            raise ValueError("incomplete checkpoint marked complete")
        return
    if payload["status"].startswith("stopped-"):
        return

    for protocol in PROTOCOLS:
        for arm in ARMS:
            for dt_ms in DT_MS:
                for repetition in range(2):
                    key = run_key(protocol, arm, dt_ms, repetition)
                    if key in payload["runs"]:
                        continue
                    path = raw_dir / f"{key}.npz"
                    if path.exists():
                        raise ValueError(f"uncheckpointed raw trace requires review: {path}")
                    print(f"Running {key}", flush=True)
                    arrays = simulate(
                        baseline=baseline, arm=arm, protocol=protocol, dt_ms=dt_ms
                    )
                    summary = summarize(arrays)
                    digest = save_trace(path, arrays)
                    payload["runs"][key] = {
                        "filename": path.name,
                        "sha256": digest,
                        "protocol": protocol,
                        "arm": arm,
                        "dt_ms": dt_ms,
                        "repetition": repetition,
                        "summary": summary,
                    }
                    if not summary["finite"]:
                        payload["status"] = "stopped-nonfinite-recording"
                        checkpoint(output, payload)
                        return
                    checkpoint(output, payload)
                    del arrays

    exact = {}
    numerical = {}
    cross_arm = {}
    for protocol in PROTOCOLS:
        for arm in ARMS:
            for dt_ms in DT_MS:
                prefix = f"{protocol}-{arm}-dt{dt_ms:g}"
                exact[prefix] = exact_trace_repeat(
                    recover(payload["runs"][prefix + "-repeat0"]),
                    recover(payload["runs"][prefix + "-repeat1"]),
                )
            numerical[f"{protocol}-{arm}"] = numerical_gate(
                payload["runs"][f"{protocol}-{arm}-dt0.01-repeat0"]["summary"],
                payload["runs"][f"{protocol}-{arm}-dt0.005-repeat0"]["summary"],
            )
        for dt_ms in DT_MS:
            cross_arm[f"{protocol}-dt{dt_ms:g}"] = cross_arm_gate(
                {
                    arm: recover(
                        payload["runs"][f"{protocol}-{arm}-dt{dt_ms:g}-repeat0"]
                    )
                    for arm in ARMS
                },
                protocol=protocol,
            )
    payload["assessment"] = {
        "exact_raw_repeats": exact,
        "numerical_gates": numerical,
        "cross_arm_gates": cross_arm,
        "all_exact_repeats": bool(all(exact.values())),
        "all_numerical_gates_pass": bool(
            all(result["pass"] for result in numerical.values())
        ),
        "all_cross_arm_gates_pass": bool(
            all(result["pass"] for result in cross_arm.values())
        ),
    }
    payload["assessment"]["isolated_promotion_gates_pass"] = bool(
        payload["assessment"]["all_exact_repeats"]
        and payload["assessment"]["all_numerical_gates_pass"]
        and payload["assessment"]["all_cross_arm_gates_pass"]
    )
    payload["status"] = "completed-inhibitory-target-location-isolated"
    checkpoint(output, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seal", type=Path, default=DEFAULT_SEAL)
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(output.suffix + ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another runner owns this result") from error
        execute(output, args.seal.resolve())


if __name__ == "__main__":
    main()
