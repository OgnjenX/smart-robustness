"""Run sealed SST Stage 4C with every registered point and its learned weights."""

from __future__ import annotations

import argparse
import fcntl
import math
import platform
from collections import Counter
from copy import deepcopy
from pathlib import Path

import brian2 as brian
import run_l5_sst_like_stage4b as parent
import yaml
from verify_l5_sst_like_stage4b_continuation import verify_result as verify_stage4b

from smart_robustness import classic_sector
from smart_robustness.baseline import load_frozen_classic_baseline
from smart_robustness.validation.active_apical_recording import file_sha256
from smart_robustness.validation.figure15 import run_figure15_condition

SEAL = Path("docs/validation-results/post2008-l5-sst-like-stage4c-seal-1015.yaml")
ASSESSMENT = Path("docs/validation-results/post2008-l5-sst-like-stage4b-assessment-1014.yaml")
REGISTRATION = Path(
    "docs/validation-results/post2008-l5-sst-like-stage4-progression-registration-1002.yaml"
)
PARENT_PROTOCOL = Path("docs/validation-results/calibrated-figure15-holdout-registration-810.yaml")
RESULT = Path("results/l5-sst-like-stage4c-figure15.yaml")
GATES = ("both_cells_have_at_least_two_spikes", "peak_in_published_gamma_band")


def validate_protocol(registration: dict, parent_registration: dict) -> dict:
    stage = registration["stage4c_figure15"]
    protocol = parent_registration["figure15_protocol"]
    expected = {
        "duration_ms": 1000.0,
        "dt_ms": 0.01,
        "histogram_bin_ms": 1.0,
        "display_max_lag_ms": 180.0,
    }
    if any(stage.get(k) != v or protocol.get(k) != v for k, v in expected.items()):
        raise ValueError("Stage-4C protocol changed")
    if (
        stage.get("pair") != [39, 40]
        or protocol.get("first_cell_index") != 39
        or protocol.get("second_cell_index") != 40
        or stage.get("condition") != "learned_horizontal_match"
        or protocol.get("condition") != "learned_horizontal_match"
        or protocol.get("spectrum_input") != "complete_linear_cross_correlogram"
        or stage.get("input_learning_state") != "corresponding-stage4a-repetition-and-point"
        or stage.get("source_identifiable_gate")
        != {
            "both_cells_have_at_least_two_spikes": True,
            "peak_in_published_gamma_band_hz": [20.0, 70.0],
        }
    ):
        raise ValueError("Stage-4C pair, analysis, gates or weights changed")
    diagnostic = stage["retained_noncompensatory_numeric_diagnostic"]
    if (
        diagnostic["graphical_target_hz"] != 44.0
        or diagnostic["repository_tolerance_hz"] != 5.0
        or protocol["target_hz"] != 44.0
        or protocol["tolerance_hz"] != 5.0
        or stage["result_target"] != str(RESULT)
    ):
        raise ValueError("Stage-4C diagnostic or result changed")
    design = registration["execution_design"]
    if (
        any(
            design.get(k) is not True
            for k in (
                "exact_repeat_required",
                "execute_all_seven_points_in_every_substage",
                "no_point_dropped_after_a_failed_substage",
                "failed_earlier_substage_cannot_be_compensated_by_later_pass",
            )
        )
        or design["repetitions_per_point_per_substage"] != 2
    ):
        raise ValueError("Stage-4C inventory or progression changed")
    return {
        "duration_ms": 1000.0,
        "dt_ms": 0.01,
        "histogram_bin_ms": 1.0,
        "max_lag_ms": 180.0,
        "first_cell_index": 39,
        "second_cell_index": 40,
        "target_hz": 44.0,
        "tolerance_hz": 5.0,
    }


def verify_seal(path: Path) -> dict:
    seal = yaml.safe_load(path.read_text())
    required = {
        "scripts/run_l5_sst_like_stage4c.py",
        "scripts/verify_l5_sst_like_stage4c.py",
        "tests/test_l5_sst_like_stage4c.py",
        "src/smart_robustness/validation/figure15.py",
        "src/smart_robustness/analysis/figure15.py",
        str(REGISTRATION),
        str(PARENT_PROTOCOL),
        str(ASSESSMENT),
    }
    if (
        seal.get("status") != "sealed-before-stage4c-network-outcomes"
        or seal.get("execution_authorized") is not True
        or seal.get("network_outcome_observed_before_seal") is not False
        or not required.issubset(seal.get("files", {}))
        or seal.get("result") != str(RESULT)
    ):
        raise ValueError("Stage-4C execution seal is incomplete or unauthorized")
    for filename, digest in seal["files"].items():
        if file_sha256(Path(filename)) != digest:
            raise ValueError(f"Stage-4C sealed file changed: {filename}")
    environment = {
        "python": platform.python_version(),
        "brian2": brian.__version__,
        "platform": platform.platform(),
    }
    if seal["environment"] != environment:
        raise ValueError("Stage-4C environment changed")
    return seal


def load_inputs(seal: dict):
    assessment = yaml.safe_load(ASSESSMENT.read_text())
    authorization = assessment["stage4c_authorization"]
    if (
        authorization.get("authorized") is not True
        or authorization.get("all_seven_points_required") is not True
    ):
        raise ValueError("Stage-4B assessment does not authorize Stage 4C")
    result_path = Path(assessment["raw_result"])
    if file_sha256(result_path) != assessment["raw_result_sha256"]:
        raise ValueError("Stage-4B result changed")
    verify_stage4b(result_path)
    spectra = yaml.safe_load(result_path.read_text())
    old_seal = parent.verify_seal(parent.DEFAULT_SEAL)
    learning = parent.stage4a_input(old_seal)
    baseline = load_frozen_classic_baseline(old_seal["baseline_manifest"])
    if (
        baseline.manifest_fingerprint != seal["baseline_manifest_fingerprint"]
        or baseline.runtime_fingerprint != seal["runtime_fingerprint"]
    ):
        raise ValueError("Stage-4C frozen baseline changed")
    registration = yaml.safe_load(REGISTRATION.read_text())
    parent_registration = yaml.safe_load(PARENT_PROTOCOL.read_text())
    if file_sha256(PARENT_PROTOCOL) != registration["stage4c_figure15"]["parent_protocol_sha256"]:
        raise ValueError("Stage-4C parent protocol identity changed")
    protocol = validate_protocol(registration, parent_registration)
    manifest = yaml.safe_load(Path(old_seal["baseline_manifest"]).read_text())
    profile = yaml.safe_load(
        (baseline.repository_root / manifest["implementation"]["profile"]["path"]).read_text()
    )
    if (
        profile["figure7_protocol"]["top_down_current_pA"] != 800.0
        or profile["figure7_protocol"]["top_down_current_mode"] != "until_cued_cell_first_event"
        or profile["comparator"]
        != {"transform": "top_k_binary", "source_index": 40, "target_count": 5}
    ):
        raise ValueError("Stage-4C stimulus or comparator changed")
    return learning, spectra, baseline, profile, protocol


def run_repetition(*, point, repetition, learned_weights, conventions, scales, profile, protocol):
    stimulus = profile["figure7_protocol"]
    original = classic_sector.build_first_order_connected_sector
    try:
        classic_sector.build_first_order_connected_sector = parent.stage4a._point_builder(point)
        result = run_figure15_condition(
            learned_weights=learned_weights,
            conventions=conventions,
            persistent_projection_weight_scales=scales,
            top_down_current_pA=float(stimulus["top_down_current_pA"]),
            top_down_current_mode=stimulus["top_down_current_mode"],
            top_down_cue_lead_ms=float(stimulus["top_down_cue_lead_ms"]),
            equilibration_ms=float(stimulus["equilibration_ms"]),
            comparator_top_k_targets=5,
            comparator_source_index=40,
            **protocol,
            brian=brian,
        )
    finally:
        classic_sector.build_first_order_connected_sector = original
    network, synchrony = result.network_result, result.synchrony
    peak = float(synchrony.gamma_peak_hz)
    gates = {
        GATES[0]: synchrony.first_spike_count >= 2 and synchrony.second_spike_count >= 2,
        GATES[1]: math.isfinite(peak) and 20.0 <= peak <= 70.0,
    }
    return {
        "repetition": repetition,
        "layer4_spike_indices": [int(i) for i in network.layer4_spike_indices],
        "layer4_spike_times_ms": [float(t) for t in network.layer4_spike_times_ms],
        "first_cell_spikes": int(synchrony.first_spike_count),
        "second_cell_spikes": int(synchrony.second_spike_count),
        "gamma_peak_hz": peak if math.isfinite(peak) else None,
        "numeric_44hz_diagnostic_pass": math.isfinite(peak) and abs(peak - 44.0) <= 5.0,
        "gates": gates,
        "figure15_source_identifiable_pass": all(gates.values()),
    }


def classify_point(outcomes: list[dict]) -> dict:
    if len(outcomes) != 2:
        raise ValueError("Stage-4C requires two repetitions")
    normalized = []
    for outcome in outcomes:
        value = deepcopy(outcome)
        value.pop("repetition", None)
        normalized.append(value)
    exact = normalized[0] == normalized[1]
    passed = all(o["figure15_source_identifiable_pass"] is True for o in outcomes)
    return {
        "exact_repeat": exact,
        "both_figure15_gate_sets_pass": passed,
        "classification": "engineering_stop"
        if not exact
        else "figure15_survival"
        if passed
        else "figure15_failure",
    }


def validate_payload(payload: dict, identity: dict, learning: dict, spectra: dict) -> None:
    if (
        payload.get("identity") != identity
        or payload.get("schema_version") != 1
        or payload.get("status") not in {"running-stage4c", "completed-stage4c"}
        or payload.get("frozen_baseline_modified") is not False
        or payload.get("network_execution") is not True
    ):
        raise ValueError("Stage-4C checkpoint identity or status changed")
    parent.validate_checkpoint_points(payload["points"])
    for index, point in enumerate(payload["points"]):
        if (
            point.get("stage4a_classification") != learning["points"][index]["classification"]
            or point.get("stage4b_classification") != spectra["points"][index]["classification"]
        ):
            raise ValueError("Stage-4C earlier classification changed")
        for outcome in point["outcomes"]:
            from verify_l5_sst_like_stage4c import check_outcome

            check_outcome(outcome)
        if point.get("classification") is not None:
            if any(point.get(k) != v for k, v in classify_point(point["outcomes"]).items()):
                raise ValueError("Stage-4C checkpoint classification changed")
            if point["classification"] == "engineering_stop":
                raise RuntimeError("Stage-4C exact repeat failed")
    complete = len(payload["points"]) == 7 and all(
        p.get("classification") for p in payload["points"]
    )
    if payload["status"] == "completed-stage4c" and (
        not complete
        or payload.get("all_points_reported") is not True
        or payload.get("classification_counts")
        != dict(sorted(Counter(p["classification"] for p in payload["points"]).items()))
    ):
        raise ValueError("Stage-4C terminal inventory changed")


def identity_for(seal_path: Path, seal: dict, protocol: dict) -> dict:
    return {
        "seal_sha256": file_sha256(seal_path),
        "files": seal["files"],
        "protocol": protocol,
        "environment": seal["environment"],
        "registered_points": parent.stage4a.registered_points(),
        "baseline_manifest_fingerprint": seal["baseline_manifest_fingerprint"],
        "runtime_fingerprint": seal["runtime_fingerprint"],
    }


def execute(output: Path, seal_path: Path) -> None:
    seal = verify_seal(seal_path)
    if output.resolve() != RESULT.resolve():
        raise ValueError("Stage-4C output differs from registration")
    learning, spectra, baseline, profile, protocol = load_inputs(seal)
    identity = identity_for(seal_path, seal, protocol)
    payload = (
        yaml.safe_load(output.read_text())
        if output.exists()
        else {
            "schema_version": 1,
            "status": "running-stage4c",
            "identity": identity,
            "points": [],
            "all_points_reported": False,
            "classification_counts": {},
            "network_execution": True,
            "frozen_baseline_modified": False,
        }
    )
    validate_payload(payload, identity, learning, spectra)
    if payload["status"] == "completed-stage4c":
        return
    brian.prefs.codegen.target = "numpy"
    for index, registered in enumerate(parent.stage4a.registered_points()):
        if index == len(payload["points"]):
            payload["points"].append(
                {
                    **registered,
                    "outcomes": [],
                    "classification": None,
                    "stage4a_classification": learning["points"][index]["classification"],
                    "stage4b_classification": spectra["points"][index]["classification"],
                }
            )
        point = payload["points"][index]
        for repetition in range(len(point["outcomes"]), 2):
            outcome = run_repetition(
                point=point,
                repetition=repetition,
                learned_weights=parent.learned_weights_for_repetition(learning, index, repetition),
                conventions=baseline.runtime_conventions(),
                scales=dict(baseline.projection_weight_scales),
                profile=profile,
                protocol=protocol,
            )
            from verify_l5_sst_like_stage4c import check_outcome

            check_outcome(outcome)
            point["outcomes"].append(outcome)
            parent.checkpoint(output, payload)
        point.update(classify_point(point["outcomes"]))
        parent.checkpoint(output, payload)
        if point["classification"] == "engineering_stop":
            raise RuntimeError("Stage-4C exact repeat failed")
    payload.update(
        status="completed-stage4c",
        all_points_reported=True,
        classification_counts=dict(
            sorted(Counter(p["classification"] for p in payload["points"]).items())
        ),
    )
    validate_payload(payload, identity, learning, spectra)
    parent.checkpoint(output, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seal", type=Path, default=SEAL)
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(output.suffix + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(output, args.seal.resolve())


if __name__ == "__main__":
    main()
