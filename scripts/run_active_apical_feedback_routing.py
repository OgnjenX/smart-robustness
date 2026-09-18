"""Run the sealed conductance-only active-apical feedback routing sweep."""

from __future__ import annotations

import argparse
import hashlib
import tempfile
from pathlib import Path

import numpy as np
import yaml
from run_joint_calibration_projection038_figure10_screen import _figure6_gates

from smart_robustness import classic_sector
from smart_robustness.baseline import load_frozen_classic_baseline
from smart_robustness.classic_sector import build_full_smart_network
from smart_robustness.models.active_apical_feedback import (
    ACTIVE_APICAL_FEEDBACK_TEMPLATE_ID,
    active_apical_feedback_topology,
    make_active_apical_feedback_full_builder,
)
from smart_robustness.validation.figure6 import (
    Figure6LearningProtocol,
    run_figure6_learning,
)
from smart_robustness.validation.figure10_search_cycle_spread import (
    build_projection036_variance_sector,
)
from smart_robustness.validation.higher_order import (
    Figure16Protocol,
    assess_figure16_candidate,
    run_figure16_candidate,
)

ARM_FRACTIONS = {
    "offset_0": 0.0,
    "offset_0p25": 0.25,
    "offset_0p5": 0.5,
    "offset_0p75": 0.75,
    "offset_1": 1.0,
}
ARM_ORDER = tuple(ARM_FRACTIONS)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _without_repetition(value: dict[str, object]) -> dict[str, object]:
    return {key: item for key, item in value.items() if key != "repetition"}


def _topology_summary(fraction: float) -> dict[str, object]:
    pre, post, factor = active_apical_feedback_topology(fraction)
    matrix = np.zeros((81, 81), dtype=float)
    matrix[pre, post] = factor
    control_pre, control_post, control_factor = active_apical_feedback_topology(0.0)
    control = np.zeros((81, 81), dtype=float)
    control[control_pre, control_post] = control_factor
    return {
        "edge_count": int(np.count_nonzero(matrix)),
        "source_row_sum_min": float(np.min(np.sum(matrix, axis=1))),
        "source_row_sum_max": float(np.max(np.sum(matrix, axis=1))),
        "source_row_sums_equal_control": bool(
            np.allclose(
                np.sum(matrix, axis=1),
                np.sum(control, axis=1),
                rtol=1e-12,
                atol=1e-12,
            )
        ),
        "center_factor_at_source40": float(matrix[40, 40]),
        "nearest_surround_factor_at_source40": float(matrix[40, 39]),
    }


def _run_repetition(
    *,
    fraction: float,
    repetition: int,
    learned_weights,
    scales,
    conventions,
    protocol: Figure16Protocol,
    geometry_seed: int,
    brian,
) -> dict[str, object]:
    builder = make_active_apical_feedback_full_builder(
        offset_route_fraction=fraction,
        base_builder=build_full_smart_network,
    )
    with tempfile.TemporaryDirectory(
        prefix=f"smart-apical-{fraction:g}-{repetition}-",
        dir="/private/tmp",
    ) as temporary:
        candidate = run_figure16_candidate(
            learned_weights=learned_weights,
            persistent_projection_weight_scales=scales,
            projection036_variance_topology=True,
            protocol=protocol,
            geometry_seed=geometry_seed,
            conventions=conventions,
            cpp_standalone_directory=Path(temporary) / "standalone",
            network_builder=builder,
            brian=brian,
        )
    assessment = assess_figure16_candidate(candidate)
    correlations = [
        {
            "band_hz": list(item.band_hz),
            "peak_absolute_normalized": item.peak_absolute_normalized,
            "peak_lag_ms": item.peak_lag_ms,
        }
        for item in assessment.cross_correlations
    ]
    gamma = next(
        item["peak_absolute_normalized"]
        for item in correlations
        if item["band_hz"] == [20.0, 100.0]
    )
    lower = max(
        item["peak_absolute_normalized"]
        for item in correlations
        if item["band_hz"][1] <= 20.0
    )
    frequency = assessment.frequency_assessment
    v1_finite = bool(np.all(np.isfinite(candidate.v1_field.potential_uV)))
    v2_finite = bool(np.all(np.isfinite(candidate.v2_field.potential_uV)))
    gates = {
        "exactly_1000_recording_samples": len(candidate.sample_times_ms) == 1000,
        "v1_and_v2_fields_finite": v1_finite and v2_finite,
        "lower_frequency_stronger_than_gamma": (
            frequency.lower_frequency_stronger_than_gamma
        ),
        "strongest_band_remains_2_4_hz": tuple(frequency.strongest_band_hz)
        == (2.0, 4.0),
    }
    return {
        "repetition": repetition,
        "sample_count": len(candidate.sample_times_ms),
        "v1_all_finite": v1_finite,
        "v2_all_finite": v2_finite,
        "cross_correlations": correlations,
        "strongest_band_hz": list(frequency.strongest_band_hz),
        "lower_frequency_to_gamma_peak_ratio": float(lower / gamma),
        "gates": gates,
        "pass": all(gates.values()),
    }


def _arm_summary(
    fraction: float,
    outcomes: list[dict[str, object]],
    exact_reruns: int,
) -> dict[str, object]:
    exact_repeat = len(outcomes) < 2 or all(
        _without_repetition(item) == _without_repetition(outcomes[0])
        for item in outcomes[1:]
    )
    return {
        "offset_route_fraction": fraction,
        "topology": _topology_summary(fraction),
        "outcomes": outcomes,
        "exact_repeat": exact_repeat,
        "all_repetitions_pass": bool(
            len(outcomes) == exact_reruns
            and exact_repeat
            and all(item["pass"] for item in outcomes)
        ),
    }


def _identity(args, baseline, registration_sha256: str, study_sha256: str):
    return {
        "baseline_manifest": args.baseline,
        "baseline_manifest_fingerprint": baseline.manifest_fingerprint,
        "runtime_fingerprint": baseline.runtime_fingerprint,
        "study": args.study,
        "study_sha256": study_sha256,
        "registration": args.registration,
        "registration_sha256": registration_sha256,
        "network_outcome_used_for_parameter_selection": False,
    }


def _load_arms(output: Path, *, identity: dict[str, object]) -> dict[str, object]:
    if not output.exists():
        return {}
    raw = yaml.safe_load(output.read_text())
    if any(raw.get(key) != value for key, value in identity.items()):
        raise ValueError("existing active-apical result does not match this sealed run")
    arms = raw.get("arms", {})
    if not isinstance(arms, dict) or set(arms) - set(ARM_ORDER):
        raise ValueError("existing active-apical result contains unknown arms")
    return arms


def _write(
    output: Path,
    *,
    status: str,
    identity: dict[str, object],
    figure6_gates: dict[str, bool],
    arms: dict[str, object],
    complete: bool,
) -> None:
    payload = {
        "schema_version": 1,
        "status": status,
        **identity,
        "first_order_figure6_gates": figure6_gates,
        "intervention_projection_id": ACTIVE_APICAL_FEEDBACK_TEMPLATE_ID,
        "arms": arms,
    }
    if complete:
        payload.update(
            {
                "all_exact_repeats": all(
                    arms[name]["exact_repeat"] for name in ARM_ORDER
                ),
                "figure16_passing_offset_fractions": [
                    ARM_FRACTIONS[name]
                    for name in ARM_ORDER
                    if arms[name]["all_repetitions_pass"]
                ],
            }
        )
    output.write_text(yaml.safe_dump(payload, sort_keys=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        default="configs/baselines/classic_smart_calibrated_v1.yaml",
    )
    parser.add_argument(
        "--study",
        default="configs/robustness/active_apical_feedback_routing_v1.yaml",
    )
    parser.add_argument(
        "--registration",
        default=(
            "docs/validation-results/"
            "post2008-active-apical-feedback-routing-registration-945.yaml"
        ),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    baseline = load_frozen_classic_baseline(args.baseline)
    study_path = Path(args.study)
    registration_path = Path(args.registration)
    study = yaml.safe_load(study_path.read_text())
    registration = yaml.safe_load(registration_path.read_text())
    if _sha256(study_path) != registration["study_sha256"]:
        raise ValueError("active-apical study differs from sealed registration")
    sealed = registration["sealed_implementation"]
    sealed_paths = {
        "module": "module_sha256",
        "full_network_extension_hook": "full_network_extension_hook_sha256",
        "higher_order_harness": "higher_order_harness_sha256",
        "runner": "runner_sha256",
        "focused_test": "focused_test_sha256",
    }
    for path_key, hash_key in sealed_paths.items():
        sealed_path = Path(__file__) if path_key == "runner" else Path(sealed[path_key])
        if _sha256(sealed_path) != sealed[hash_key]:
            raise ValueError(f"sealed active-apical {path_key} differs from registration")
    source_audit = registration["source_audit"]
    if _sha256(Path(source_audit["path"])) != source_audit["sha256"]:
        raise ValueError("active-apical source audit differs from registration")
    control = registration["control"]
    if _sha256(Path(control["manifest"])) != control["manifest_sha256"]:
        raise ValueError("frozen control manifest differs from registration")
    if tuple(study["execution"]["arm_order"]) != ARM_ORDER:
        raise ValueError("runner arm order differs from sealed study")
    if tuple(float(v) for v in study["intervention"]["route_fractions"]) != tuple(
        ARM_FRACTIONS.values()
    ):
        raise ValueError("runner route fractions differ from sealed study")
    exact_reruns = int(study["execution"]["exact_reruns_per_arm"])
    if exact_reruns != 2:
        raise ValueError("the sealed study requires two exact repetitions")

    raw_manifest = yaml.safe_load(Path(args.baseline).read_text())
    profile = yaml.safe_load(
        (
            baseline.repository_root / raw_manifest["implementation"]["profile"]["path"]
        ).read_text()
    )
    training_profile = yaml.safe_load(
        (baseline.repository_root / profile["training_profile"]).read_text()
    )

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    original_builder = classic_sector.build_first_order_connected_sector
    try:
        classic_sector.build_first_order_connected_sector = (
            build_projection036_variance_sector
        )
        training = run_figure6_learning(
            conventions=baseline.runtime_conventions(),
            protocol=Figure6LearningProtocol(
                monitored_populations=tuple(training_profile["monitored_populations"])
            ),
            projection_weight_scales=dict(baseline.projection_weight_scales),
            brian=brian,
        )
    finally:
        classic_sector.build_first_order_connected_sector = original_builder
    figure6_gates = _figure6_gates(training, profile)
    if not all(figure6_gates.values()):
        raise RuntimeError(f"fresh Figure 6 prerequisite failed: {figure6_gates}")

    protocol_values = registration["figure16_protocol"]
    protocol = Figure16Protocol(
        prestimulus_ms=float(protocol_values["prestimulus_ms"]),
        recording_ms=float(protocol_values["recording_ms"]),
        inter_area_delay_ms=float(protocol_values["inter_area_delay_ms"]),
        integration_dt_ms=float(protocol_values["integration_dt_ms"]),
        recording_sample_ms=float(protocol_values["recording_sample_ms"]),
        frequency_bands_hz=tuple(
            tuple(float(value) for value in band)
            for band in protocol_values["frequency_bands_hz"]
        ),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    identity = _identity(
        args,
        baseline,
        _sha256(registration_path),
        _sha256(study_path),
    )
    arms = _load_arms(output, identity=identity)
    scales = dict(baseline.projection_weight_scales)
    for arm_index, arm_name in enumerate(ARM_ORDER):
        if any(name in arms for name in ARM_ORDER[arm_index + 1 :]) and arm_name not in arms:
            raise ValueError("resume result violates the sealed arm order")
        outcomes = list(arms.get(arm_name, {}).get("outcomes", []))
        if len(outcomes) > exact_reruns:
            raise ValueError(f"invalid saved outcome count for {arm_name}")
        for repetition in range(len(outcomes), exact_reruns):
            outcomes.append(
                _run_repetition(
                    fraction=ARM_FRACTIONS[arm_name],
                    repetition=repetition,
                    learned_weights=training.learned_weights,
                    scales=scales,
                    conventions=baseline.runtime_conventions(),
                    protocol=protocol,
                    geometry_seed=int(protocol_values["geometry_seed_v1"]),
                    brian=brian,
                )
            )
            arms[arm_name] = _arm_summary(
                ARM_FRACTIONS[arm_name],
                outcomes,
                exact_reruns,
            )
            _write(
                output,
                status="running-active-apical-feedback-routing",
                identity=identity,
                figure6_gates=figure6_gates,
                arms=arms,
                complete=False,
            )
        if arm_name == "offset_0" and not arms[arm_name]["all_repetitions_pass"]:
            raise RuntimeError("contemporaneous offset-0 control failed")
    _write(
        output,
        status="completed-active-apical-feedback-routing",
        identity=identity,
        figure6_gates=figure6_gates,
        arms=arms,
        complete=True,
    )


if __name__ == "__main__":
    main()
