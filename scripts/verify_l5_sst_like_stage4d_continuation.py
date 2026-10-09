"""Amended raw numerical check; unchanged fields, normalized metrics and gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import verify_l5_sst_like_stage4d as original
import yaml

from smart_robustness.validation import l5_sst_like_stage4d as contract
from smart_robustness.validation.active_apical_recording import file_sha256
from smart_robustness.validation.stage4d_precision import BANDS, compare_band


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
    x = np.asarray(fields["v2"]["inferior_300um_potential_uV"]).mean(axis=0)
    y = np.asarray(fields["v1"]["superior_300um_potential_uV"]).mean(axis=0)
    energy = float(np.linalg.norm(x - x.mean()) * np.linalg.norm(y - y.mean()))
    rebuilt = original.reconstruct(fields)
    stored = outcome["cross_correlations"]
    if len(stored) != 5:
        raise ValueError("Stage-4D frequency band inventory changed")
    diagnostics = [
        {"band_hz": list(band), **compare_band(a, b, energy=energy, band=band)}
        for band, a, b in zip(BANDS, stored, rebuilt, strict=True)
    ]
    peaks = [r["peak_absolute_normalized"] for r in rebuilt]
    gates = dict(zip(contract.GATES, [True, True, max(peaks[:4]) > peaks[4]], strict=True))
    if outcome.get("gates") != gates or any(type(v) is not bool for v in outcome["gates"].values()):
        raise ValueError("Stage-4D gates contradict reconstructed fields")
    return {
        "peak_by_band": peaks,
        "failed_gates": [k for k, v in gates.items() if not v],
        "signal_energy": energy,
        "numerical_diagnostics": diagnostics,
    }


def verify_result(path: Path, seal_path: Path | None = None) -> dict:
    import run_l5_sst_like_stage4d_continuation as runner

    seal_path = seal_path or runner.SEAL
    seal = runner.verify_seal(seal_path)
    inputs = runner.load_inputs(seal)
    if path.resolve() != runner.RESULT.resolve():
        raise ValueError("Continuation result path changed")
    payload = yaml.safe_load(path.read_text())
    runner.validate_payload(payload, runner.identity_for(seal_path, seal, inputs), inputs)
    if payload["status"] != "completed-stage4d-continuation":
        raise ValueError("Continuation result is not terminal")
    return {
        "raw_result_sha256": file_sha256(path),
        "raw_result_bytes": path.stat().st_size,
        "completed_points": 7,
        "completed_repetitions": 14,
        "original_failure_retained": True,
        "joint_classification_counts": payload["joint_classification_counts"],
        "numerical_diagnostics": payload["numerical_diagnostics"],
    }


def main():
    import run_l5_sst_like_stage4d_continuation as runner

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--seal", type=Path, default=runner.SEAL)
    args = parser.parse_args()
    print(json.dumps(verify_result(args.result, args.seal), indent=2))


if __name__ == "__main__":
    main()
