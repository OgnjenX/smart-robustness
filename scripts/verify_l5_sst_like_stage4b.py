"""Independently verify a completed sealed SST-like Stage-4B Figure-14 result."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import run_l5_sst_like_stage4b as runner
import yaml

from smart_robustness.analysis.figure14 import (
    assess_figure14_spectra,
    figure14_spectrum_from_spikes,
)
from smart_robustness.validation.active_apical_recording import file_sha256


def _check_condition(data: dict[str, Any], protocol: dict[str, float]) -> Any:
    spikes = data.get("cortical_spike_times_ms")
    if not isinstance(spikes, list) or len(spikes) != data.get("cortical_spike_count"):
        raise ValueError("Figure-14 cortical spike inventory mismatch")
    spectrum = figure14_spectrum_from_spikes(
        spikes,
        duration_ms=protocol["duration_ms"],
        histogram_bin_ms=protocol["histogram_bin_ms"],
        hamming_window_ms=protocol["hamming_window_ms"],
    )
    diagnostics = {
        "dominant_frequency_hz": spectrum.dominant_frequency_hz,
        "low_power_2_8_hz": spectrum.low_power,
        "middle_caption_power_8_20_hz": spectrum.middle_caption_power,
        "middle_methods_power_8_10_hz": spectrum.middle_methods_power,
        "gamma_power_20_70_hz": spectrum.gamma_power,
    }
    for name, computed in diagnostics.items():
        stored = data.get(name)
        if not isinstance(stored, (float, int)) or not math.isclose(
            float(stored), computed, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError(f"Figure-14 spectrum summary mismatch: {name}")
    for name in ("relay_events", "trn_events", "nonspecific_events"):
        if not isinstance(data.get(name), int) or data[name] < 0:
            raise ValueError(f"invalid Figure-14 event count: {name}")
    return spectrum


def _check_outcome(outcome: dict[str, Any], protocol: dict[str, float]) -> dict[str, Any]:
    match = _check_condition(outcome["match"], protocol)
    mismatch = _check_condition(outcome["mismatch"], protocol)
    assessment = assess_figure14_spectra(match, mismatch)
    gates = {
        runner.GATE_NAMES[0]: assessment.match_gamma_dominant,
        runner.GATE_NAMES[1]: assessment.mismatch_lower_frequency_dominant,
        runner.GATE_NAMES[2]: assessment.mismatch_gamma_reduced,
    }
    if outcome.get("gates") != gates or outcome.get("figure14_pass") is not all(gates.values()):
        raise ValueError("stored Figure-14 gates contradict recalculated spectra")
    return {
        "match_dominant_frequency_hz": match.dominant_frequency_hz,
        "mismatch_dominant_frequency_hz": mismatch.dominant_frequency_hz,
        "match_gamma_power": match.gamma_power,
        "mismatch_gamma_power": mismatch.gamma_power,
        "failed_gates": [name for name, passed in gates.items() if not passed],
    }


def verify_result(result_path: Path, seal_path: Path) -> dict[str, Any]:
    """Recompute identities, spectra, gates, exact repeats and classifications."""

    seal = runner.verify_seal(seal_path)
    if result_path.resolve() != Path(seal["result"]).resolve():
        raise ValueError("Stage-4B result path differs from seal")
    raw = result_path.read_bytes()
    result = yaml.safe_load(raw)
    if not isinstance(result, dict) or result.get("status") != "completed-l5-sst-like-stage4b":
        raise ValueError("Stage-4B result is not terminal")
    if result.get("all_points_reported") is not True or result.get("network_execution") is not True:
        raise ValueError("Stage-4B result lacks complete network execution")
    if result.get("frozen_baseline_modified") is not False:
        raise ValueError("Stage-4B result does not preserve frozen baseline")
    identity = result.get("identity")
    if not isinstance(identity, dict):
        raise TypeError("Stage-4B identity must be a mapping")
    expected_identity = {
        "seal_sha256": file_sha256(seal_path),
        "stage4_registration_sha256": seal["stage4_registration_sha256"],
        "parent_figure14_registration_sha256": seal["parent_figure14_registration_sha256"],
        "stage4a_assessment_sha256": seal["stage4a_assessment_sha256"],
        "stage4a_result_sha256": seal["stage4a_result_sha256"],
        "baseline_manifest_fingerprint": seal["baseline_manifest_fingerprint"],
        "runtime_fingerprint": seal["runtime_fingerprint"],
        "registered_points": runner.stage4a.registered_points(),
    }
    for name, value in expected_identity.items():
        if identity.get(name) != value:
            raise ValueError(f"Stage-4B result identity mismatch: {name}")
    registration = yaml.safe_load(Path(seal["stage4_registration"]).read_text())
    parent = yaml.safe_load(Path(seal["parent_figure14_registration"]).read_text())
    protocol = runner.validate_protocol(registration, parent)
    if identity.get("protocol") != protocol:
        raise ValueError("Stage-4B result protocol mismatch")
    stage4a_result = runner.stage4a_input(seal)

    points = result.get("points")
    if not isinstance(points, list) or len(points) != len(runner.stage4a.registered_points()):
        raise ValueError("Stage-4B result point inventory is incomplete")
    runner.validate_checkpoint_points(points)
    point_evidence = []
    for index, point in enumerate(points):
        if point.get("stage4a_classification") != stage4a_result["points"][index]["classification"]:
            raise ValueError("Stage-4B point loses Stage-4A classification")
        outcomes = point["outcomes"]
        if len(outcomes) != runner.REPETITIONS:
            raise ValueError("Stage-4B point lacks exact two-repetition inventory")
        summaries = [_check_outcome(outcome, protocol) for outcome in outcomes]
        normalized = []
        for outcome in outcomes:
            copy = deepcopy(outcome)
            copy.pop("repetition", None)
            normalized.append(copy)
        exact_repeat = normalized[0] == normalized[1]
        both_pass = all(outcome["figure14_pass"] is True for outcome in outcomes)
        classification = (
            "engineering_stop"
            if not exact_repeat
            else "figure14_survival"
            if both_pass
            else "figure14_failure"
        )
        if (
            point.get("exact_repeat") is not exact_repeat
            or point.get("both_figure14_gate_sets_pass") is not both_pass
            or point.get("classification") != classification
        ):
            raise ValueError("Stage-4B point classification contradicts raw outcomes")
        if classification == "engineering_stop":
            raise ValueError("Stage-4B exact repeat failed")
        point_evidence.append(
            {
                "point_id": point["point_id"],
                "delay_ms": point["delay_ms"],
                "resource_fraction": point["resource_fraction"],
                "total_conductance_nS": point["total_conductance_nS"],
                "stage4a_classification": point["stage4a_classification"],
                "stage4b_classification": classification,
                "exact_repeat": exact_repeat,
                "repetitions": summaries,
            }
        )

    counts = dict(sorted(Counter(point["classification"] for point in points).items()))
    if result.get("classification_counts") != counts:
        raise ValueError("Stage-4B classification counts mismatch")
    return {
        "raw_result": str(result_path),
        "raw_result_sha256": file_sha256(result_path),
        "raw_result_bytes": len(raw),
        "seal": str(seal_path),
        "seal_sha256": file_sha256(seal_path),
        "completed_points": len(points),
        "completed_repetitions": sum(len(point["outcomes"]) for point in points),
        "classification_counts": counts,
        "all_points_exact_repeat": all(point["exact_repeat"] for point in points),
        "point_evidence": point_evidence,
        "independent_verification_passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--seal", type=Path, default=runner.DEFAULT_SEAL)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    evidence = verify_result(args.result.resolve(), args.seal.resolve())
    rendered = yaml.safe_dump(evidence, sort_keys=False)
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.write_text(rendered)


if __name__ == "__main__":
    main()
