"""Run sealed SST-like Stage 4B Figure-14 spectra from Stage-4A weights."""

from __future__ import annotations

import argparse
import fcntl
import os
import platform
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import run_l5_sst_like_stage4a as stage4a
import yaml
from verify_l5_sst_like_stage4a import verify_result as verify_stage4a_result

from smart_robustness import classic_sector
from smart_robustness.analysis.figure14 import assess_figure14_spectra
from smart_robustness.baseline import load_frozen_classic_baseline
from smart_robustness.protocols import MatchCondition
from smart_robustness.validation.active_apical_recording import file_sha256
from smart_robustness.validation.figure14 import run_figure14_condition

REPETITIONS = 2
DEFAULT_SEAL = Path("docs/validation-results/post2008-l5-sst-like-stage4b-seal-1007.yaml")
LEARNED_IDS = {
    "bottom_up_weights": "modeldb112923.projection.035",
    "top_down_wide_weights": "modeldb112923.projection.005",
    "top_down_narrow_weights": "modeldb112923.projection.007",
}
GATE_NAMES = (
    "match_dominant_frequency_in_20_70_hz",
    "mismatch_dominant_frequency_in_2_20_hz",
    "mismatch_gamma_power_strictly_below_match",
)


def checkpoint(path: Path, payload: dict[str, Any]) -> None:
    """Atomically preserve each completed repetition."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".l5-sst-stage4b-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            yaml.safe_dump(payload, stream, sort_keys=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def verify_seal(path: Path) -> dict[str, Any]:
    """Reject changed source or an unauthorized Stage-4B execution."""

    seal = yaml.safe_load(path.read_text())
    if seal.get("status") != "sealed-before-stage4b-network-outcomes":
        raise ValueError("a Stage-4B execution seal is required")
    files = seal.get("files")
    if not isinstance(files, dict):
        raise TypeError("Stage-4B seal lacks its file inventory")
    for filename, digest in files.items():
        if file_sha256(Path(filename)) != digest:
            raise ValueError(f"sealed file changed: {filename}")
    if "scripts/run_l5_sst_like_stage4b.py" not in files:
        raise ValueError("Stage-4B runner is missing from execution seal")
    return seal


def validate_protocol(registration: dict[str, Any], parent: dict[str, Any]) -> dict[str, float]:
    """Ensure the preregistered Figure-14 assay is unchanged."""

    stage = registration["stage4b_figure14"]
    protocol = parent["figure14_protocol"]
    expected = {
        "duration_ms": 1000.0,
        "dt_ms": 0.01,
        "histogram_bin_ms": 1.0,
        "hamming_window_ms": 200.0,
    }
    for key, value in expected.items():
        if stage.get(key) != value or protocol.get(key) != value:
            raise ValueError(f"Stage-4B protocol mismatch: {key}")
    if stage.get("conditions") != ["match", "mismatch"]:
        raise ValueError("Stage-4B conditions changed")
    if stage.get("input_learning_state") != "corresponding-stage4a-repetition-and-point":
        raise ValueError("Stage-4B learned-state source changed")
    if tuple(stage.get("gates", ())) != GATE_NAMES:
        raise ValueError("Stage-4B gates changed")
    if protocol.get("hamming_overlap_ms") != 0.0:
        raise ValueError("Figure-14 window overlap changed")
    if protocol.get("cortical_signal") != "cumulative_spikes_all_v1_cortical_populations":
        raise ValueError("Figure-14 cortical signal changed")
    expected_bands = {
        "low_band_hz": [2.0, 8.0],
        "middle_caption_band_hz": [8.0, 20.0],
        "middle_methods_band_hz": [8.0, 10.0],
        "gamma_band_hz": [20.0, 70.0],
    }
    for name, band in expected_bands.items():
        if protocol.get(name) != band:
            raise ValueError(f"Figure-14 analysis band changed: {name}")
    return expected


def stage4a_input(seal: dict[str, Any]) -> dict[str, Any]:
    """Check assessed, complete Stage-4A evidence before reading learned weights."""

    assessment_path = Path(seal["stage4a_assessment"])
    result_path = Path(seal["stage4a_result"])
    if file_sha256(assessment_path) != seal["stage4a_assessment_sha256"]:
        raise ValueError("Stage-4A assessment differs from Stage-4B seal")
    if file_sha256(result_path) != seal["stage4a_result_sha256"]:
        raise ValueError("Stage-4A raw result differs from Stage-4B seal")
    verified = verify_stage4a_result(result_path, Path(seal["stage4a_seal"]))
    if verified["raw_result_sha256"] != seal["stage4a_result_sha256"]:
        raise ValueError("independent Stage-4A verification differs from seal")
    assessment = yaml.safe_load(assessment_path.read_text())
    if (
        assessment.get("status") != "completed-mixed-stage4a-six-survivors-one-failure"
        or assessment.get("raw_result_sha256") != seal["stage4a_result_sha256"]
        or assessment.get("stage4b_authorization", {}).get("authorized") is not True
    ):
        raise ValueError("Stage-4A assessment does not authorize Stage 4B")
    result = yaml.safe_load(result_path.read_text())
    if (
        result.get("status") != "completed-l5-sst-like-stage4a"
        or result.get("all_points_reported") is not True
        or result.get("identity", {}).get("registered_points") != stage4a.registered_points()
        or len(result.get("points", [])) != len(stage4a.registered_points())
    ):
        raise ValueError("Stage-4A result is incomplete or changed")
    stage4a.validate_checkpoint_points(result["points"])
    if any(len(point["outcomes"]) != REPETITIONS for point in result["points"]):
        raise ValueError("Stage-4A repetitions are incomplete")
    return result


def learned_weights_for_repetition(
    stage4a_result: dict[str, Any], point_index: int, repetition: int
) -> dict[str, Any]:
    """Use precisely the corresponding Stage-4A point and repetition."""

    point = stage4a_result["points"][point_index]
    if point["point_id"] != stage4a.registered_points()[point_index]["point_id"]:
        raise ValueError("Stage-4A point identity mismatch")
    outcome = point["outcomes"][repetition]
    if outcome["repetition"] != repetition:
        raise ValueError("Stage-4A repetition identity mismatch")
    figure6 = outcome["figure6"]
    return {projection_id: figure6[key] for key, projection_id in LEARNED_IDS.items()}


def condition_summary(result: Any) -> dict[str, Any]:
    network = result.network_result
    spectrum = result.spectrum
    return {
        "cortical_spike_times_ms": [float(value) for value in network.v1_cortical_spike_times_ms],
        "cortical_spike_count": len(network.v1_cortical_spike_times_ms),
        "relay_events": len(network.relay_spike_times_ms),
        "trn_events": len(network.trn_spike_times_ms),
        "nonspecific_events": len(network.nonspecific_spike_times_ms),
        "dominant_frequency_hz": spectrum.dominant_frequency_hz,
        "low_power_2_8_hz": spectrum.low_power,
        "middle_caption_power_8_20_hz": spectrum.middle_caption_power,
        "middle_methods_power_8_10_hz": spectrum.middle_methods_power,
        "gamma_power_20_70_hz": spectrum.gamma_power,
    }


def run_registered_repetition(
    *,
    point: dict[str, Any],
    repetition: int,
    learned_weights: dict[str, Any],
    conventions: Any,
    scales: dict[str, float],
    profile: dict[str, Any],
    protocol: dict[str, float],
    brian: Any,
) -> dict[str, Any]:
    """Run fixed match and mismatch spectra with the corresponding learned state."""

    figure7_protocol = profile["figure7_protocol"]
    comparator = profile["comparator"]
    common = {
        "top_down_current_pA": float(figure7_protocol["top_down_current_pA"]),
        "learned_weights": learned_weights,
        "conventions": conventions,
        "persistent_projection_weight_scales": scales,
        "comparator_top_k_targets": int(comparator["target_count"]),
        "comparator_source_index": int(comparator["source_index"]),
        "top_down_current_mode": figure7_protocol["top_down_current_mode"],
        "top_down_cue_lead_ms": float(figure7_protocol["top_down_cue_lead_ms"]),
        "equilibration_ms": float(figure7_protocol["equilibration_ms"]),
        **protocol,
        "brian": brian,
    }
    original_builder = classic_sector.build_first_order_connected_sector
    try:
        classic_sector.build_first_order_connected_sector = stage4a._point_builder(point)
        match = run_figure14_condition(condition=MatchCondition.MATCH, **common)
        mismatch = run_figure14_condition(condition=MatchCondition.MISMATCH, **common)
    finally:
        classic_sector.build_first_order_connected_sector = original_builder
    assessment = assess_figure14_spectra(match.spectrum, mismatch.spectrum)
    gates = {
        GATE_NAMES[0]: assessment.match_gamma_dominant,
        GATE_NAMES[1]: assessment.mismatch_lower_frequency_dominant,
        GATE_NAMES[2]: assessment.mismatch_gamma_reduced,
    }
    return {
        "repetition": repetition,
        "match": condition_summary(match),
        "mismatch": condition_summary(mismatch),
        "gates": gates,
        "figure14_pass": all(gates.values()),
    }


def classify_point(outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    if len(outcomes) != REPETITIONS:
        raise ValueError("Stage-4B classification requires exactly two repetitions")
    normalized = []
    for outcome in outcomes:
        copy = deepcopy(outcome)
        copy.pop("repetition", None)
        normalized.append(copy)
    exact_repeat = normalized[0] == normalized[1]
    both_pass = all(outcome.get("figure14_pass") is True for outcome in outcomes)
    return {
        "exact_repeat": exact_repeat,
        "both_figure14_gate_sets_pass": both_pass,
        "classification": (
            "engineering_stop"
            if not exact_repeat
            else "figure14_survival"
            if both_pass
            else "figure14_failure"
        ),
    }


def validate_checkpoint_points(points: list[dict[str, Any]]) -> None:
    expected = stage4a.registered_points()
    if len(points) > len(expected):
        raise ValueError("checkpoint contains unregistered Stage-4B points")
    identity_keys = (
        "stage4a_index",
        "point_index",
        "point_id",
        "delay_ms",
        "resource_fraction",
        "total_conductance_nS",
    )
    for index, point in enumerate(points):
        if any(point.get(key) != expected[index][key] for key in identity_keys):
            raise ValueError("checkpoint violates Stage-4B point order or identity")
        outcomes = point.get("outcomes")
        if not isinstance(outcomes, list) or len(outcomes) > REPETITIONS:
            raise ValueError("checkpoint has an invalid repetition inventory")
        for repetition, outcome in enumerate(outcomes):
            if outcome.get("repetition") != repetition:
                raise ValueError("checkpoint violates repetition order")
        if point.get("classification") is not None and len(outcomes) != REPETITIONS:
            raise ValueError("checkpoint classifies an incomplete point")
        if index < len(points) - 1 and point.get("classification") is None:
            raise ValueError("checkpoint advances beyond an incomplete point")


def execute(output: Path, seal_path: Path) -> None:
    import brian2 as brian

    seal = verify_seal(seal_path)
    if file_sha256(Path(seal["stage4_registration"])) != seal["stage4_registration_sha256"]:
        raise ValueError("Stage-4 registration changed")
    if (
        file_sha256(Path(seal["parent_figure14_registration"]))
        != seal["parent_figure14_registration_sha256"]
    ):
        raise ValueError("parent Figure-14 registration changed")
    registration = yaml.safe_load(Path(seal["stage4_registration"]).read_text())
    parent = yaml.safe_load(Path(seal["parent_figure14_registration"]).read_text())
    if (
        registration["stage4b_figure14"]["parent_protocol_sha256"]
        != seal["parent_figure14_registration_sha256"]
    ):
        raise ValueError("Stage-4B parent protocol identity changed")
    protocol = validate_protocol(registration, parent)
    stage4a_result = stage4a_input(seal)
    baseline = load_frozen_classic_baseline(seal["baseline_manifest"])
    if (
        baseline.manifest_fingerprint != seal["baseline_manifest_fingerprint"]
        or baseline.runtime_fingerprint != seal["runtime_fingerprint"]
    ):
        raise ValueError("frozen baseline identity changed")
    manifest = yaml.safe_load(Path(seal["baseline_manifest"]).read_text())
    profile = yaml.safe_load(
        (baseline.repository_root / manifest["implementation"]["profile"]["path"]).read_text()
    )
    figure7_protocol = profile["figure7_protocol"]
    comparator = profile["comparator"]
    parent_protocol = parent["figure14_protocol"]
    if (
        figure7_protocol["top_down_current_pA"] != parent_protocol["top_down_current_pA"]
        or figure7_protocol["top_down_current_mode"] != parent_protocol["top_down_current_mode"]
        or comparator != {"transform": "top_k_binary", "source_index": 40, "target_count": 5}
        or parent_protocol["comparator"] != "top_k_binary_5_targets_source_40"
    ):
        raise ValueError("Figure-14 stimulus or comparator differs from parent")
    identity = {
        "seal_sha256": file_sha256(seal_path),
        "stage4_registration_sha256": seal["stage4_registration_sha256"],
        "parent_figure14_registration_sha256": seal["parent_figure14_registration_sha256"],
        "stage4a_assessment_sha256": seal["stage4a_assessment_sha256"],
        "stage4a_result_sha256": seal["stage4a_result_sha256"],
        "baseline_manifest_fingerprint": baseline.manifest_fingerprint,
        "runtime_fingerprint": baseline.runtime_fingerprint,
        "registered_points": stage4a.registered_points(),
        "protocol": protocol,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "brian2": brian.__version__,
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "running-l5-sst-like-stage4b",
        "identity": identity,
        "points": [],
        "all_points_reported": False,
        "classification_counts": {},
        "network_execution": True,
        "frozen_baseline_modified": False,
    }
    if output.exists():
        payload = yaml.safe_load(output.read_text())
        if payload.get("identity") != identity:
            raise ValueError("checkpoint identity mismatch")
    points = payload.get("points")
    if not isinstance(points, list):
        raise TypeError("checkpoint point inventory is invalid")
    validate_checkpoint_points(points)
    if any(point.get("classification") == "engineering_stop" for point in points):
        raise RuntimeError("Stage-4B checkpoint contains an exact-repeat failure")
    if payload.get("status") == "completed-l5-sst-like-stage4b":
        if len(points) != len(stage4a.registered_points()):
            raise ValueError("incomplete checkpoint marked complete")
        return

    brian.prefs.codegen.target = "numpy"
    conventions = baseline.runtime_conventions()
    scales = dict(baseline.projection_weight_scales)
    expected_points = stage4a.registered_points()
    for point_index, registered in enumerate(expected_points):
        if point_index == len(points):
            point: dict[str, Any] = {
                **registered,
                "stage4a_classification": stage4a_result["points"][point_index]["classification"],
                "outcomes": [],
                "exact_repeat": None,
                "both_figure14_gate_sets_pass": None,
                "classification": None,
            }
            points.append(point)
        else:
            point = points[point_index]
            if (
                point.get("stage4a_classification")
                != stage4a_result["points"][point_index]["classification"]
            ):
                raise ValueError("checkpoint Stage-4A classification mismatch")
        for repetition in range(len(point["outcomes"]), REPETITIONS):
            weights = learned_weights_for_repetition(stage4a_result, point_index, repetition)
            point["outcomes"].append(
                run_registered_repetition(
                    point=point,
                    repetition=repetition,
                    learned_weights=weights,
                    conventions=conventions,
                    scales=scales,
                    profile=profile,
                    protocol=protocol,
                    brian=brian,
                )
            )
            checkpoint(output, payload)
        point.update(classify_point(point["outcomes"]))
        checkpoint(output, payload)
        if point["classification"] == "engineering_stop":
            raise RuntimeError("Stage-4B exact-repeat requirement failed")

    counts: dict[str, int] = {}
    for point in points:
        name = str(point["classification"])
        counts[name] = counts.get(name, 0) + 1
    payload["all_points_reported"] = len(points) == len(expected_points)
    payload["classification_counts"] = dict(sorted(counts.items()))
    payload["status"] = "completed-l5-sst-like-stage4b"
    checkpoint(output, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seal", type=Path, default=DEFAULT_SEAL)
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(output.suffix + ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("an identical L5 SST-like Stage-4B runner is active") from error
        execute(output, args.seal.resolve())


if __name__ == "__main__":
    main()
