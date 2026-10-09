"""Run all registered synthetic precision and corruption cases, never network data."""

from __future__ import annotations

import argparse
import itertools
import platform
import subprocess
from copy import deepcopy
from pathlib import Path

import numpy as np
import verify_l5_sst_like_stage4d as original
import yaml

from smart_robustness.analysis.cross_correlation import figure16_cross_correlations
from smart_robustness.validation.active_apical_recording import file_sha256
from smart_robustness.validation.stage4d_precision import BANDS, compare_band

REGISTRATION = Path(
    "docs/validation-results/post2008-l5-sst-like-stage4d-numerical-audit-registration-1019.yaml"
)
RESULT = Path("results/l5-sst-like-stage4d-numerical-audit-1020.yaml")
REGISTRATION_SHA256 = "c609fb498ccb36c53481edec360307081dd537dcd328b1721b4eb1577720a835"
FAMILIES = ("white_noise", "band_boundaries", "mixed_phases", "silent")
SEEDS = (19, 23)
AMPLITUDES = (1e-6, 1e-3, 1.0, 1e3, 1e6)
SHIFTS = (0, 7, 499)


def signals(family, seed, amplitude, shift):
    rng = np.random.default_rng(seed)
    t = np.arange(1000) / 1000
    if family == "white_noise":
        x = amplitude * rng.normal(size=1000)
        y = np.roll(x, shift) + 0.2 * amplitude * rng.normal(size=1000)
    elif family == "band_boundaries":
        x = amplitude * sum(np.sin(2 * np.pi * f * t) for f in (2, 4, 8, 12, 20, 99, 100))
        y = np.roll(x, shift)
    elif family == "mixed_phases":
        phases = rng.uniform(0, 2 * np.pi, 5)
        frequencies = (3, 6, 10, 16, 40)
        x = amplitude * sum(
            np.sin(2 * np.pi * f * t + p) for f, p in zip(frequencies, phases, strict=True)
        )
        y = amplitude * sum(
            np.cos(2 * np.pi * f * (t - shift / 1000) + p)
            for f, p in zip(frequencies, phases, strict=True)
        )
    elif family == "silent":
        x = y = np.zeros(1000)
    else:
        raise ValueError("Unregistered synthetic family")
    return x, y


def band_records(x, y):
    stored = [
        {
            "band_hz": list(c.band_hz),
            "lag_ms": c.lag_ms.tolist(),
            "raw": c.raw.tolist(),
            "normalized": c.normalized.tolist(),
            "peak_absolute_normalized": c.peak_absolute_normalized,
        }
        for c in figure16_cross_correlations(x, y, 1000, bands_hz=BANDS)
    ]
    # Only synthetic arrays are passed to the original independent algorithm.
    reference = original.reconstruct(
        {
            "v2": {"inferior_300um_potential_uV": [x.tolist()]},
            "v1": {"superior_300um_potential_uV": [y.tolist()]},
        }
    )
    energy = float(np.linalg.norm(x - x.mean()) * np.linalg.norm(y - y.mean()))
    return stored, reference, energy


def corruption_checks(stored, reference, energy, band):
    controls = {}
    mutations = {
        "raw": lambda d: d["raw"].__setitem__(
            500, d["raw"][500] + (1e-4 * energy if energy > 0 else 1e-30)
        ),
        "normalized": lambda d: d["normalized"].__setitem__(500, d["normalized"][500] + 1e-4),
        "peak": lambda d: d.__setitem__(
            "peak_absolute_normalized", d["peak_absolute_normalized"] + 1e-4
        ),
        "lag_order": lambda d: d["lag_ms"].reverse(),
        "wrong_length": lambda d: d["raw"].pop(),
        "wrong_band": lambda d: d.__setitem__("band_hz", [0, 1]),
    }
    for key in ("raw", "normalized", "peak_absolute_normalized"):
        for label, invalid in (("nan", float("nan")), ("inf", float("inf"))):
            if key == "peak_absolute_normalized":
                mutations[f"{key}_{label}"] = lambda d, v=invalid: d.__setitem__(
                    "peak_absolute_normalized", v
                )
            else:
                mutations[f"{key}_{label}"] = lambda d, k=key, v=invalid: d[k].__setitem__(500, v)
    for label, mutate in mutations.items():
        corrupted = deepcopy(stored)
        mutate(corrupted)
        try:
            compare_band(corrupted, reference, energy=energy, band=band)
        except ValueError:
            controls[label] = True
        else:
            controls[label] = False
    for label, invalid in (
        ("energy_nan", float("nan")),
        ("energy_inf", float("inf")),
        ("energy_negative", -1.0),
    ):
        try:
            compare_band(stored, reference, energy=invalid, band=band)
        except ValueError:
            controls[label] = True
        else:
            controls[label] = False
    return controls


def run_audit():
    if file_sha256(REGISTRATION) != REGISTRATION_SHA256:
        raise ValueError("Prospective numerical-audit registration changed")
    reg = yaml.safe_load(REGISTRATION.read_text())
    design = reg["synthetic_design"]
    if (
        design["seeds"] != list(SEEDS)
        or design["amplitudes"] != list(AMPLITUDES)
        or design["circular_shifts_samples"] != list(SHIFTS)
        or tuple(design["families"]) != FAMILIES
        or reg["network_execution_authorized"] is not False
        or reg["network_artifact_replay_authorized"] is not False
    ):
        raise ValueError("Synthetic audit inventory or authorization changed")
    records = []
    for family, seed, amplitude, shift in itertools.product(FAMILIES, SEEDS, AMPLITUDES, SHIFTS):
        stored, reference, energy = band_records(*signals(family, seed, amplitude, shift))
        bands = []
        for band, a, b in zip(BANDS, stored, reference, strict=True):
            try:
                evidence = compare_band(a, b, energy=energy, band=band)
            except ValueError as exc:
                evidence = {"comparison_pass": False, "failure": str(exc)}
            else:
                evidence["comparison_pass"] = True
            evidence["original_dimensional_raw_pass"] = bool(
                np.allclose(a["raw"], b["raw"], rtol=1e-12, atol=1e-12)
            )
            controls = corruption_checks(a, b, energy, band)
            bands.append({"band_hz": list(band), **evidence, "corruption_rejections": controls})
        records.append(
            {
                "family": family,
                "seed": seed,
                "amplitude": amplitude,
                "shift": shift,
                "energy": energy,
                "bands": bands,
            }
        )
    files = [
        str(REGISTRATION),
        "scripts/run_l5_sst_like_stage4d_numerical_audit.py",
        "src/smart_robustness/validation/stage4d_precision.py",
        "scripts/verify_l5_sst_like_stage4d.py",
        "src/smart_robustness/analysis/cross_correlation.py",
    ]
    passed = all(
        b["comparison_pass"] and all(b["corruption_rejections"].values())
        for r in records
        for b in r["bands"]
    )
    return {
        "schema_version": 1,
        "status": "completed-synthetic-audit",
        "contract_pass": passed,
        "implementation_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "files": {f: file_sha256(Path(f)) for f in files},
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "network_execution": False,
        "network_artifact_read": False,
        "original_failure_retained": True,
        "cases": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULT)
    args = parser.parse_args()
    if args.output.resolve() != RESULT.resolve() or args.output.exists():
        raise ValueError("Audit output must be a fresh designated artifact")
    payload = run_audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(payload, sort_keys=False))
    print(
        {
            "cases": len(payload["cases"]),
            "bands": sum(len(r["bands"]) for r in payload["cases"]),
            "contract_pass": payload["contract_pass"],
        }
    )


if __name__ == "__main__":
    main()
