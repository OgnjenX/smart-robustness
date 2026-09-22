from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from smart_robustness.analysis.figure14 import (
    assess_figure14_spectra,
    figure14_spectrum_from_spikes,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_l5_sst_like_stage4b.py"
PROTOCOL = {
    "duration_ms": 1000.0,
    "dt_ms": 0.01,
    "histogram_bin_ms": 1.0,
    "hamming_window_ms": 200.0,
}


def load_verifier():
    spec = importlib.util.spec_from_file_location("l5_sst_stage4b_verifier", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SCRIPT.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def condition(spikes: list[float]) -> tuple[dict, object]:
    spectrum = figure14_spectrum_from_spikes(
        spikes,
        duration_ms=PROTOCOL["duration_ms"],
        histogram_bin_ms=PROTOCOL["histogram_bin_ms"],
        hamming_window_ms=PROTOCOL["hamming_window_ms"],
    )
    return (
        {
            "cortical_spike_times_ms": spikes,
            "cortical_spike_count": len(spikes),
            "relay_events": 1,
            "trn_events": 0,
            "nonspecific_events": 0,
            "dominant_frequency_hz": spectrum.dominant_frequency_hz,
            "low_power_2_8_hz": spectrum.low_power,
            "middle_caption_power_8_20_hz": spectrum.middle_caption_power,
            "middle_methods_power_8_10_hz": spectrum.middle_methods_power,
            "gamma_power_20_70_hz": spectrum.gamma_power,
        },
        spectrum,
    )


def test_verifier_recomputes_spectra_and_rejects_tampering() -> None:
    verifier = load_verifier()
    match, match_spectrum = condition([float(x) for x in range(10, 1000, 20)])
    mismatch, mismatch_spectrum = condition([float(x) for x in range(10, 1000, 100)])
    assessment = assess_figure14_spectra(match_spectrum, mismatch_spectrum)
    gates = {
        verifier.runner.GATE_NAMES[0]: assessment.match_gamma_dominant,
        verifier.runner.GATE_NAMES[1]: assessment.mismatch_lower_frequency_dominant,
        verifier.runner.GATE_NAMES[2]: assessment.mismatch_gamma_reduced,
    }
    outcome = {
        "repetition": 0,
        "match": match,
        "mismatch": mismatch,
        "gates": gates,
        "figure14_pass": all(gates.values()),
    }
    result = verifier._check_outcome(outcome, PROTOCOL)
    assert result["match_dominant_frequency_hz"] == match_spectrum.dominant_frequency_hz
    match["gamma_power_20_70_hz"] += 1.0
    with pytest.raises(ValueError, match="spectrum summary mismatch"):
        verifier._check_outcome(outcome, PROTOCOL)
    match["gamma_power_20_70_hz"] -= 1.0
    outcome["gates"][verifier.runner.GATE_NAMES[0]] = not gates[verifier.runner.GATE_NAMES[0]]
    with pytest.raises(ValueError, match="gates contradict"):
        verifier._check_outcome(outcome, PROTOCOL)


def test_verifier_rejects_spike_inventory_mismatch() -> None:
    verifier = load_verifier()
    data, _ = condition([1.0, 2.0])
    data["cortical_spike_count"] = 3
    with pytest.raises(ValueError, match="spike inventory mismatch"):
        verifier._check_condition(data, PROTOCOL)
