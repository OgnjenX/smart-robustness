"""Sealed isolated release assay; no neurons, fitting, or network attachment."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import yaml

from smart_robustness.models.pulvinar_stp import STPParameters, release_history

ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = ROOT / "docs/validation-results/post2008-pulvinar-stp-release-registration-956.yaml"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    registration_bytes = REGISTRATION.read_bytes()
    spec = yaml.safe_load(registration_bytes)
    for relative, expected in spec["implementation_sha256"].items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"sealed implementation mismatch: {relative}")
    rows = []
    for name, params in spec["parameters"].items():
        parameters = STPParameters(**params)
        for frequency in spec["frequencies_hz"]:
            arrivals = np.arange(spec["events_per_train"]) / frequency
            repetitions = []
            for _ in range(spec["repetitions"]):
                result = release_history(arrivals, parameters)
                repetitions.append({k: v.tolist() for k, v in result.items()})
            if repetitions[0] != repetitions[1]:
                raise ValueError("exact repetition failed")
            r = result["released"]
            finite = all(np.all(np.isfinite(v)) for v in result.values())
            bounded = all(
                np.all((v >= 0) & (v <= 1))
                for k, v in result.items() if k != "time_s"
            )
            conserved = bool(np.allclose(
                result["x_pre"], result["x_post"] + r, rtol=0, atol=1e-14,
            ))
            if not (finite and bounded and conserved):
                raise ValueError("state-integrity check failed")
            rows.append({
                "terminal": name, "frequency_hz": frequency,
                "repetitions": repetitions, "exact_repeat": True,
                "finite_bounded_conserved": True,
                "second_over_first_release": float(r[1] / r[0]),
                "last_over_first_release": float(r[-1] / r[0]),
            })
    artifact = {
        "schema_version": 1,
        "status": "complete-isolated-release-only",
        "registration_sha256": hashlib.sha256(registration_bytes).hexdigest(),
        "implementation_sha256": spec["implementation_sha256"],
        "network_execution": False,
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "pyyaml": yaml.__version__},
        "rows": rows,
    }
    # Exclusive creation: no accidental replacement of a prior scientific result.
    with args.output.open("x") as stream:
        json.dump(artifact, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Completed {len(rows)} conditions, each repeated exactly twice: {args.output}")


if __name__ == "__main__":
    main()
