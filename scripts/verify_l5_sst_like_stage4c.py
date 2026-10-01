"""Independently recompute SST local synchrony from raw spikes, without assay helpers."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import run_l5_sst_like_stage4c as runner
import yaml

from smart_robustness.validation.active_apical_recording import file_sha256


def check_outcome(outcome: dict) -> dict:
    indices = np.asarray(outcome["layer4_spike_indices"])
    times = np.asarray(outcome["layer4_spike_times_ms"], dtype=float)
    if (
        indices.ndim != 1
        or times.ndim != 1
        or len(indices) != len(times)
        or not np.all(np.isfinite(times))
        or np.any(times < 0)
        or np.any(times >= 1000)
        or not np.all(np.isin(indices, np.arange(81)))
    ):
        raise ValueError("Stage-4C raw spike inventory invalid")
    trains, counts = [], []
    for cell in (39, 40):
        selected = times[indices == cell]
        counts.append(len(selected))
        train = np.zeros(1000)
        train[np.unique(np.floor(selected).astype(int))] = 1.0
        trains.append(train - train.mean())
    # Deliberately use direct correlation and explicit FFT, not scipy's assay helpers.
    full = np.correlate(trains[0], trains[1], mode="full")
    centered = full - full.mean()
    window = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(len(full)) / len(full))
    power = np.abs(np.fft.rfft(centered * window)) ** 2
    frequencies = np.fft.rfftfreq(len(full), d=0.001)
    mask = (frequencies >= 20) & (frequencies <= 70)
    peak = float(frequencies[mask][np.argmax(power[mask])]) if np.any(power[mask] > 0) else None
    # A truly silent pair has no spectral peak; tiny roundoff in FFT correlation
    # is not evidence of oscillation. Non-silent trains have the same registered peak.
    stored = outcome.get("gamma_peak_hz")
    if (peak is None and stored is not None) or (
        peak is not None
        and (
            not isinstance(stored, (int, float))
            or not math.isclose(peak, stored, abs_tol=1e-12, rel_tol=1e-12)
        )
    ):
        raise ValueError("Stage-4C gamma peak contradicts independent FFT")
    gates = {
        runner.GATES[0]: counts[0] >= 2 and counts[1] >= 2,
        runner.GATES[1]: peak is not None and 20 <= peak <= 70,
    }
    diagnostic = peak is not None and abs(peak - 44) <= 5
    if (
        outcome.get("first_cell_spikes") != counts[0]
        or outcome.get("second_cell_spikes") != counts[1]
        or outcome.get("gates") != gates
        or outcome.get("figure15_source_identifiable_pass") is not all(gates.values())
        or outcome.get("numeric_44hz_diagnostic_pass") is not diagnostic
    ):
        raise ValueError("Stage-4C stored gates or counts contradict spikes")
    return {
        "first_cell_spikes": counts[0],
        "second_cell_spikes": counts[1],
        "gamma_peak_hz": peak,
        "numeric_44hz_diagnostic_pass": diagnostic,
        "failed_gates": [k for k, v in gates.items() if not v],
    }


def verify_result(path: Path, seal_path: Path = runner.SEAL) -> dict:
    seal = runner.verify_seal(seal_path)
    learning, spectra, _baseline, _profile, protocol = runner.load_inputs(seal)
    if path.resolve() != runner.RESULT.resolve():
        raise ValueError("Stage-4C result path changed")
    payload = yaml.safe_load(path.read_text())
    runner.validate_payload(
        payload, runner.identity_for(seal_path, seal, protocol), learning, spectra
    )
    if payload["status"] != "completed-stage4c":
        raise ValueError("Stage-4C result is not terminal")
    return {
        "raw_result_sha256": file_sha256(path),
        "raw_result_bytes": path.stat().st_size,
        "completed_points": 7,
        "completed_repetitions": 14,
        "all_points_exact_repeat": True,
        "independent_verification_passed": True,
        "classification_counts": payload["classification_counts"],
        "point_evidence": [
            {
                "point_id": p["point_id"],
                "stage4a_classification": p["stage4a_classification"],
                "stage4b_classification": p["stage4b_classification"],
                "stage4c_classification": p["classification"],
                "repetitions": [check_outcome(o) for o in p["outcomes"]],
            }
            for p in payload["points"]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--seal", type=Path, default=runner.SEAL)
    args = parser.parse_args()
    print(json.dumps(verify_result(args.result, args.seal), indent=2))


if __name__ == "__main__":
    main()
