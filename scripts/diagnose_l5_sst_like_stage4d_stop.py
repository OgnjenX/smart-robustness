"""Describe the preserved Stage-4D engineering stop; never authorize or rerun it."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import verify_l5_sst_like_stage4d as verifier
import yaml

from smart_robustness.validation.active_apical_recording import file_sha256

RESULT = Path("results/l5-sst-like-stage4d-figure16.yaml")
EXPECTED_SHA256 = "d949391b2ef505df892ea50fe438fdb87e1a4547953b6de06e3567a6c242e7f2"


def diagnose() -> dict:
    if file_sha256(RESULT) != EXPECTED_SHA256:
        raise ValueError("Preserved Stage-4D stop artifact changed")
    payload = yaml.safe_load(RESULT.read_text())
    if len(payload["points"]) != 1 or len(payload["points"][0]["outcomes"]) != 1:
        raise ValueError("Expected exactly one preserved first-repetition outcome")
    outcome = payload["points"][0]["outcomes"][0]
    rebuilt = verifier.reconstruct(outcome["fields"])
    evidence = []
    for stored, independent in zip(outcome["cross_correlations"], rebuilt, strict=True):
        row = {"band_hz": stored["band_hz"]}
        for key in ("raw", "normalized"):
            a, b = np.asarray(stored[key]), np.asarray(independent[key])
            failed = np.flatnonzero(~np.isclose(a, b, rtol=1e-12, atol=1e-12))
            row[key] = {
                "max_absolute_error": float(np.max(np.abs(a - b))),
                "stored_max_absolute_value": float(np.max(np.abs(a))),
                "failed_samples": [
                    {
                        "index": int(i),
                        "lag_ms": stored["lag_ms"][i],
                        "stored_value": float(a[i]),
                        "reconstructed_value": float(b[i]),
                        "absolute_error": float(abs(a[i] - b[i])),
                        "allowed_error": float(1e-12 + 1e-12 * abs(b[i])),
                    }
                    for i in failed
                ],
            }
        row["stored_peak"] = stored["peak_absolute_normalized"]
        row["reconstructed_peak"] = independent["peak_absolute_normalized"]
        evidence.append(row)
    return {
        "raw_result_sha256": EXPECTED_SHA256,
        "raw_result_bytes": RESULT.stat().st_size,
        "engineering_stop_retained": True,
        "network_rerun": False,
        "gate_or_tolerance_changed": False,
        "point_id": payload["points"][0]["point_id"],
        "stored_gates": outcome["gates"],
        "reconstructed_lower_frequency_gate": max(
            r["peak_absolute_normalized"] for r in rebuilt[:4]
        )
        > rebuilt[4]["peak_absolute_normalized"],
        "band_evidence": evidence,
    }


if __name__ == "__main__":
    print(json.dumps(diagnose(), indent=2))
