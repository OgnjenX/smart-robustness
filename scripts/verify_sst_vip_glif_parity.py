"""Reconstruct synthetic parity gates from arrays without importing either simulator."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

FIELDS = (
    "voltage",
    "threshold",
    "AScurrents",
    "grid_spike_times",
    "spike_time_steps",
    "interpolated_spike_times",
    "interpolated_spike_voltage",
    "interpolated_spike_threshold",
)
ROOT = Path("results/sst-vip-glif-native-parity-1062")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reconstruct(arrays):
    checks = {}
    for field in FIELDS:
        a, b = arrays["native0_" + field], arrays["candidate0_" + field]
        shape = a.shape == b.shape
        mask = shape and np.array_equal(np.isnan(a), np.isnan(b))
        finite = shape and not np.isinf(a).any() and not np.isinf(b).any()
        values = False
        if mask and finite:
            valid = ~np.isnan(a)
            if field in ("spike_time_steps", "grid_spike_times"):
                values = np.array_equal(a[valid], b[valid])
            else:
                values = np.allclose(a[valid], b[valid], rtol=1e-10, atol=1e-12)
        checks[field] = {
            "shape": bool(shape),
            "nan_mask": bool(mask),
            "no_infinite_values": bool(finite),
            "values": bool(values),
        }
    comparison = {"passed": all(all(v.values()) for v in checks.values()), "fields": checks}
    repeats = {
        kind: all(
            np.array_equal(arrays[kind + "0_" + field], arrays[kind + "1_" + field], equal_nan=True)
            for field in FIELDS
        )
        for kind in ("native", "candidate")
    }
    return comparison, repeats


def main():
    registration = Path(
        "docs/validation-results/post2008-sst-vip-glif-parity-registration-1061.yaml"
    )
    seal_path = Path(
        "docs/validation-results/post2008-sst-vip-glif-parity-execution-seal-1063.yaml"
    )
    reg = yaml.safe_load(registration.read_text())
    seal = yaml.safe_load(seal_path.read_text())
    context = {"registration_sha256": digest(registration), "seal_sha256": digest(seal_path)}
    require(seal["registration_sha256"] == digest(registration), "registration changed")
    require(seal["native_reference_execution_authorized"] is True, "not authorized")
    for path, expected in seal["implementation_sha256"].items():
        require(digest(path) == expected, "implementation changed")
    for key in ("parent", "parameter_manifest"):
        require(digest(reg[key]) == reg[key + "_sha256"], "lineage changed")
    for key in ("native_neuron", "native_methods"):
        require(digest(reg[key + "_source"]) == reg[key + "_sha256"], "native source changed")
    parameters = {}
    source = yaml.safe_load(Path(reg["parameter_manifest"]).read_text())
    for record in source["records"]:
        if record["source"]["template"]["name"].startswith("Biophysical"):
            continue
        require(digest(record["raw_path"]) == record["raw_sha256"], "parameter bytes changed")
        parameters[str(record["source"]["model_id"])] = json.loads(
            Path(record["raw_path"]).read_text()
        )
    require(len(parameters) == 227, "source coverage changed")
    manifest = yaml.safe_load((ROOT / "manifest.yaml").read_text())
    require(manifest["context"] == context, "context changed")
    require(len(manifest["records"]) == 948, "case count changed")
    coverage, counts = Counter(), Counter()
    for number, saved in enumerate(manifest["records"]):
        require(
            yaml.safe_load((ROOT / f"case-{number:04d}.yaml").read_text()) == saved,
            "sidecar mismatch",
        )
        require(saved["context"] == context, "case context changed")
        raw = ROOT / f"case-{number:04d}.npz"
        require(
            str(raw) == saved["raw_path"] and digest(raw) == saved["raw_sha256"],
            "raw artifact changed",
        )
        identity = saved["identity"]
        case, label = identity["case"], identity["stimulus"]
        coverage[(case["identity"], label)] += 1
        p = case["parameters"]
        if case["identity"] in parameters:
            require(p == parameters[case["identity"]], "source parameters changed")
        else:
            require(case["identity"].startswith("fixture-"), "unknown case")
        c = p.get("coeffs", {})
        scale = c.get("G", 1) / p["R_input"] * (c.get("th_inf", 1) * p["th_inf"] - p["El"])
        time = np.arange(2000)
        recipes = {
            "zero": np.zeros(2000),
            "subthreshold": np.full(2000, 0.5 * scale),
            "step": np.where(time < 200, 0.0, 2 * scale),
            "pulses": np.where((time // 100) % 2 == 0, 2 * scale, 0.0),
        }
        report = saved["report"]
        with np.load(raw, allow_pickle=False) as arrays:
            require(json.loads(str(arrays["report_json"].item())) == report, "report mismatch")
            require(np.array_equal(arrays["stimulus"], recipes[label]), "stimulus changed")
            require(
                hashlib.sha256(arrays["stimulus"].tobytes()).hexdigest()
                == identity["stimulus_sha256"],
                "stimulus hash changed",
            )
            if report["errors"]:
                require(report["passed"] is False, "exception promoted")
                counts["exception_cases"] += 1
            else:
                comparison, repeats = reconstruct(arrays)
                require(comparison == report["comparison"], "array gate differs")
                for kind, exact in repeats.items():
                    require(exact == report[kind + "_exact_repeat"], "repeat gate differs")
                bad = any(s["bad_reset_stop"] for s in report["states"].values())
                require(bad == report["bad_reset_stop"], "stop flag differs")
                passed = comparison["passed"] and all(repeats.values()) and not bad
                require(passed == report["passed"], "pass gate differs")
                counts["numerical_mismatch_cases"] += not comparison["passed"]
                counts["bad_reset_cases"] += bad
                counts["repeat_failure_cases"] += not all(repeats.values())
            counts["passed_cases"] += report["passed"]
    fixture_ids = {
        f"fixture-{family}-cut-{cut}"
        for family in ("LIF", "LIF-R", "LIF-ASC", "LIF-R-ASC", "LIF-R-ASC-A")
        for cut in (0, 3)
    }
    expected = {
        (identity, label)
        for identity in set(parameters) | fixture_ids
        for label in ("zero", "subthreshold", "step", "pulses")
    }
    require(set(coverage) == expected and set(coverage.values()) == {1}, "coverage differs")
    require(counts["passed_cases"] == manifest["passed_cases"], "total differs")
    print(
        json.dumps(
            {
                "manifest_sha256": digest(ROOT / "manifest.yaml"),
                "cases": 948,
                "counts": dict(counts),
                "stop_flags_and_exceptions_are_recorded_not_independently_reexecuted": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
