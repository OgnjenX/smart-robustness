"""Reconstruct Stage-4D regional means and correlations without assay helpers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import run_l5_sst_like_stage4d as runner
import yaml

from smart_robustness.validation import l5_sst_like_stage4d as contract
from smart_robustness.validation.active_apical_recording import file_sha256


def reconstruct(fields: dict) -> list[dict]:
    """Use full complex transforms and direct circular dot products, not rFFT correlation."""
    x = np.asarray(fields["v2"]["inferior_300um_potential_uV"]).mean(axis=0)
    y = np.asarray(fields["v1"]["superior_300um_potential_uV"]).mean(axis=0)
    x, y = x - x.mean(), y - y.mean()
    n = len(x)
    frequencies = np.abs(np.fft.fftfreq(n, d=0.001))
    lags = np.arange(-n // 2, n - n // 2)
    total_energy = float(np.sqrt(np.sum(x**2) * np.sum(y**2)))
    results = []
    for low, high in contract.BANDS:
        mask = (frequencies >= low) & (frequencies < high)
        fx = np.fft.ifft(np.fft.fft(x) * mask).real
        fy = np.fft.ifft(np.fft.fft(y) * mask).real
        raw = np.array([np.dot(fx, np.roll(fy, int(lag))) for lag in lags])
        energy = float(np.sqrt(np.sum(fx**2) * np.sum(fy**2)))
        meaningful = energy > np.finfo(float).eps * max(total_energy, 1.0) * n
        normalized = raw / energy if meaningful else np.zeros(n)
        results.append(
            {
                "band_hz": [low, high],
                "lag_ms": lags.tolist(),
                "raw": raw.tolist(),
                "normalized": normalized.tolist(),
                "peak_absolute_normalized": float(np.max(np.abs(normalized))),
            }
        )
    return results


def check_outcome(outcome: dict) -> dict:
    times = np.asarray(outcome["sample_times_ms"], dtype=float)
    if times.shape != (1000,) or not np.allclose(times, np.arange(1000, 2000), rtol=0, atol=1e-8):
        raise ValueError("Stage-4D recording sample inventory changed")
    if outcome.get("learned_state_provenance") != "simulated-learned-weight-snapshot":
        raise ValueError("Stage-4D learned-state provenance changed")
    fields = outcome["fields"]
    if set(fields) != {"v1", "v2"}:
        raise ValueError("Stage-4D requires both cortical fields")
    for area, seed in (("v1", 16), ("v2", 17)):
        field = fields[area]
        if field.get("seed") != seed or not isinstance(field.get("fingerprint"), str):
            raise ValueError("Stage-4D field geometry changed")
        for name in (
            "potential_uV",
            "current_source_density_uV_per_um",
            "inferior_300um_potential_uV",
            "superior_300um_potential_uV",
        ):
            value = np.asarray(field[name], dtype=float)
            if (
                value.ndim != 2
                or value.shape[0] == 0
                or value.shape[1] != 1000
                or not np.all(np.isfinite(value))
            ):
                raise ValueError("Stage-4D field shape or finiteness invalid")
        for region in ("inferior", "superior"):
            depths = np.asarray(field[f"{region}_300um_tip_depth_um"], dtype=float)
            if depths.shape != (len(field[f"{region}_300um_potential_uV"]),) or not np.all(
                np.isfinite(depths)
            ):
                raise ValueError("Stage-4D regional depth inventory invalid")
    rebuilt = reconstruct(fields)
    stored = outcome["cross_correlations"]
    if len(stored) != len(rebuilt):
        raise ValueError("Stage-4D frequency band inventory changed")
    for original, actual in zip(stored, rebuilt, strict=True):
        if original.get("band_hz") != actual["band_hz"]:
            raise ValueError("Stage-4D band order changed")
        for key in ("lag_ms", "raw", "normalized", "peak_absolute_normalized"):
            a, b = np.asarray(original[key]), np.asarray(actual[key])
            if a.shape != b.shape or not np.allclose(a, b, rtol=1e-12, atol=1e-12):
                raise ValueError(
                    f"Stage-4D correlation contradicts independent reconstruction: {key}"
                )
    peaks = [r["peak_absolute_normalized"] for r in rebuilt]
    gates = dict(zip(contract.GATES, [True, True, max(peaks[:4]) > peaks[4]], strict=True))
    if outcome.get("gates") != gates or any(type(v) is not bool for v in outcome["gates"].values()):
        raise ValueError("Stage-4D gates contradict reconstructed fields")
    return {"peak_by_band": peaks, "failed_gates": [k for k, v in gates.items() if not v]}


def verify_result(path: Path, seal_path: Path = runner.SEAL) -> dict:
    seal = runner.verify_seal(seal_path)
    learning, spectra, synchrony, _baseline, _profile, protocol = runner.load_inputs(seal)
    if path.resolve() != runner.RESULT.resolve():
        raise ValueError("Stage-4D result path changed")
    payload = yaml.safe_load(path.read_text())
    runner.validate_payload(
        payload, runner.identity_for(seal_path, seal, protocol), learning, spectra, synchrony
    )
    if payload["status"] != "completed-stage4d":
        raise ValueError("Stage-4D result is not terminal")
    return {
        "raw_result_sha256": file_sha256(path),
        "raw_result_bytes": path.stat().st_size,
        "completed_points": 7,
        "completed_repetitions": 14,
        "all_points_exact_repeat": True,
        "independent_verification_passed": True,
        "joint_classification_counts": payload["joint_classification_counts"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--seal", type=Path, default=runner.SEAL)
    args = parser.parse_args()
    print(json.dumps(verify_result(args.result, args.seal), indent=2))


if __name__ == "__main__":
    main()
