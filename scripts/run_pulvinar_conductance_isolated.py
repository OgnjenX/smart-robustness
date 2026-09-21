"""Run the sealed isolated conductance-factorization assay (registration 959)."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import yaml

from smart_robustness.models.pulvinar_conductance import conductance_controls
from smart_robustness.models.pulvinar_stp import STPParameters

ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = (
    ROOT / "docs/validation-results/post2008-pulvinar-conductance-isolated-registration-959a.yaml"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_condition(*, frequency_hz: float, dt_ms: float, events: int, tail_ms: float):
    emissions = np.arange(events, dtype=float) * (1000 / frequency_hz)
    duration = emissions[-1] + 0.1 + tail_ms
    times = np.arange(round(duration / dt_ms) + 1, dtype=float) * dt_ms
    traces = conductance_controls(
        times,
        emissions,
        tau_ms=2.0,
        delay_ms=0.1,
        depletion_fraction=0.5,
        recovery_ms=100.0,
        type2_parameters=STPParameters(0.8, 2.0, 3.33),
    )
    return {"time_ms": times, "emissions_ms": emissions, **traces}


def summarize(arrays: dict[str, np.ndarray]) -> dict:
    time = arrays["time_ms"]
    return {
        name: {
            "peak": float(np.max(values)),
            "area_ms": float(np.trapz(values, time)),
            "finite": bool(np.all(np.isfinite(values))),
            "nonnegative": bool(np.all(values >= 0)),
        }
        for name, values in arrays.items()
        if name not in {"time_ms", "emissions_ms"}
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    registration_bytes = REGISTRATION.read_bytes()
    spec = yaml.safe_load(registration_bytes)
    for relative, expected in spec["implementation_sha256"].items():
        if _sha256(ROOT / relative) != expected:
            raise ValueError(f"sealed implementation mismatch: {relative}")
    grid = spec["scope"]

    raw_dir = args.output.with_suffix("")
    raw_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    try:
        for dt_ms in grid["dt_ms"]:
            for frequency_hz in grid["frequencies_hz"]:
                first = build_condition(
                    frequency_hz=frequency_hz, dt_ms=dt_ms,
                    events=grid["events_per_train"], tail_ms=grid["tail_ms"],
                )
                second = build_condition(
                    frequency_hz=frequency_hz, dt_ms=dt_ms,
                    events=grid["events_per_train"], tail_ms=grid["tail_ms"],
                )
                if first.keys() != second.keys() or any(
                    not np.array_equal(first[key], second[key]) for key in first
                ):
                    raise ValueError("exact raw-array repetition failed")
                summary = summarize(first)
                if not all(v["finite"] and v["nonnegative"] for v in summary.values()):
                    raise ValueError("trace integrity failed")
                filename = f"dt-{dt_ms:g}ms_frequency-{frequency_hz:g}hz.npz"
                raw_path = raw_dir / filename
                with raw_path.open("xb") as stream:
                    np.savez_compressed(stream, **first)
                rows.append({
                    "dt_ms": dt_ms, "frequency_hz": frequency_hz,
                    "exact_raw_repeat": True, "raw_file": filename,
                    "raw_sha256": _sha256(raw_path), "summary": summary,
                })
        artifact = {
            "schema_version": 1,
            "status": "complete-isolated-conductance-factorization",
            "registration_sha256": hashlib.sha256(registration_bytes).hexdigest(),
            "implementation_sha256": spec["implementation_sha256"],
            "network_execution": False,
            "software": {
                "python": platform.python_version(), "numpy": np.__version__,
                "pyyaml": yaml.__version__,
            },
            "rows": rows,
        }
        with args.output.open("x") as stream:
            json.dump(artifact, stream, indent=2, allow_nan=False)
            stream.write("\n")
    except BaseException:
        # Keep any completed exclusive raw files for diagnosis, but mark the
        # directory incomplete so it can never be mistaken for a result.
        marker = raw_dir / "INCOMPLETE"
        marker.write_text("runner did not complete\n")
        raise
    print(f"Completed {len(rows)} conditions with exact raw repeats: {args.output}")


if __name__ == "__main__":
    main()
