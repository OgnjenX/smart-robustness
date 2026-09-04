"""Derive and probe current-balance fixed points of the recovered cell."""

from __future__ import annotations

import argparse
import itertools
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.optimize import root

from smart_robustness.classic_sector import first_order_population_parameters
from smart_robustness.models.compartmental_hh import create_compartmental_hh_population
from smart_robustness.models.currents import (
    NaKRateConvention,
    TTypeGateConvention,
    t_type_h_inf,
    t_type_m_inf,
    traub_miles_rates,
)
from smart_robustness.models.modeldb112923 import first_order_population_facts
from smart_robustness.validation.calibration import runtime_conventions_for_candidate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        default=(
            "configs/calibration/"
            "isolated_nonspecific_fixed_point_derivation_v1.yaml"
        ),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    brian.start_scope()
    profile_path = Path(args.profile)
    profile = yaml.safe_load(profile_path.read_text())
    runtime_profile = yaml.safe_load(Path(profile["runtime_profile"]).read_text())
    base_profile = yaml.safe_load(Path(runtime_profile["base_profile"]).read_text())
    base = runtime_conventions_for_candidate(base_profile["candidate"])
    detector = runtime_profile["detector"]
    conventions = replace(
        base,
        trn_spike_event_coordinate="absolute_physical",
        trn_spike_event_threshold_mV=float(detector["arm_mV"]),
        trn_spike_event_release_mV=float(detector["release_mV"]),
        trn_spike_event_proximal_blend_fraction=None,
        **runtime_profile["runtime_overrides"],
    )
    facts = next(
        item
        for item in first_order_population_facts()
        if item.canonical_name == profile["population"]
    )
    params = first_order_population_parameters(facts, conventions=conventions)
    if params["voltage_coordinate"] != "relative_to_table3_leak":
        raise ValueError("fixed-point derivation requires the registered leak-relative rates")
    if params["calcium_voltage_coordinate"] != "integrated_voltage":
        raise ValueError("fixed-point derivation requires integrated-voltage calcium rates")
    population = create_compartmental_hh_population(
        name="isolated_nonspecific_fixed_point",
        size=1,
        params=params,
        brian=brian,
    )
    group = population.group
    compartments = population.cell_spec.compartments
    expected_order = [item.name for item in compartments]
    if profile["root_search"]["compartment_order"] != expected_order:
        raise ValueError("registered compartment order differs from the recovered cell")
    nak = NaKRateConvention(params["nak_rate_convention"])
    calcium = TTypeGateConvention(params["calcium_gate_convention"])

    def set_steady_state(voltages_mV: np.ndarray) -> None:
        for compartment, voltage_mV in zip(
            compartments, voltages_mV, strict=True
        ):
            setattr(group, f"v_{compartment.name}", voltage_mV * brian.mV)
            if compartment.g_na_mS_cm2 is not None:
                paper_voltage_mV = voltage_mV - compartment.e_leak_mV
                rates = traub_miles_rates(paper_voltage_mV, nak)
                setattr(
                    group,
                    f"m_{compartment.name}",
                    rates.alpha_m / (rates.alpha_m + rates.beta_m),
                )
                setattr(
                    group,
                    f"h_{compartment.name}",
                    rates.alpha_h / (rates.alpha_h + rates.beta_h),
                )
                setattr(
                    group,
                    f"n_{compartment.name}",
                    rates.alpha_n / (rates.alpha_n + rates.beta_n),
                )
            if compartment.g_ca_mS_cm2 is not None:
                setattr(
                    group,
                    f"m_ca_{compartment.name}",
                    t_type_m_inf(voltage_mV, calcium),
                )
                setattr(
                    group,
                    f"h_ca_{compartment.name}",
                    t_type_h_inf(voltage_mV, calcium),
                )

    def residual_pA(voltages_mV: np.ndarray) -> np.ndarray:
        set_steady_state(voltages_mV)
        return np.asarray(
            [
                float(
                    (
                        getattr(
                            group,
                            f"i_membrane_inward_{compartment.name}",
                        )[0]
                        + getattr(group, f"i_axial_inward_{compartment.name}")[0]
                    )
                    / brian.pA
                )
                for compartment in compartments
            ]
        )

    search = profile["root_search"]
    maximum_residual_pA = float(search["maximum_absolute_residual_pA"])
    uniqueness_tolerance_mV = float(search["uniqueness_tolerance_mV"])
    if "initial_guess_vectors_mV" in search:
        guesses = tuple(
            tuple(float(value) for value in item)
            for item in search["initial_guess_vectors_mV"]
        )
        if any(len(item) != len(compartments) for item in guesses):
            raise ValueError("each registered initial-guess vector needs one voltage per compartment")
    else:
        guesses = tuple(
            itertools.product(
                tuple(float(item) for item in search["initial_guess_axis_mV"]),
                repeat=len(compartments),
            )
        )
    unique_roots: list[np.ndarray] = []
    converged_attempts = 0
    for guess in guesses:
        candidate = root(residual_pA, guess)
        residual = residual_pA(candidate.x)
        if not candidate.success or np.max(np.abs(residual)) > maximum_residual_pA:
            continue
        converged_attempts += 1
        if not any(
            np.max(np.abs(candidate.x - known)) < uniqueness_tolerance_mV
            for known in unique_roots
        ):
            unique_roots.append(candidate.x.copy())

    for index, fixed_point in enumerate(unique_roots):
        refined = root(residual_pA, fixed_point)
        if refined.success and np.max(np.abs(residual_pA(refined.x))) <= maximum_residual_pA:
            unique_roots[index] = refined.x.copy()

    expected_fixed_point_mV = search.get("expected_fixed_point_mV")
    expected_fixed_point_pass: bool | None = None
    if expected_fixed_point_mV is not None:
        expected = np.asarray(expected_fixed_point_mV, dtype=float)
        if expected.shape != (len(compartments),):
            raise ValueError("expected fixed point needs one voltage per compartment")
        expected_fixed_point_pass = bool(
            len(unique_roots) == 1
            and np.max(np.abs(unique_roots[0] - expected))
            <= float(search["expected_fixed_point_tolerance_mV"])
        )

    stationarity = profile["stationarity_probe"]
    stationarity_result: dict[str, Any] | None = None
    if len(unique_roots) == 1:
        fixed_point = unique_roots[0]
        fixed_point_residual_pA = residual_pA(fixed_point)
        perturbation_mV = np.asarray(
            stationarity.get(
                "initial_voltage_perturbation_mV",
                [0.0] * len(compartments),
            ),
            dtype=float,
        )
        if perturbation_mV.shape != (len(compartments),):
            raise ValueError("voltage perturbation needs one value per compartment")
        for compartment, perturbation in zip(
            compartments, perturbation_mV, strict=True
        ):
            variable = f"v_{compartment.name}"
            setattr(group, variable, getattr(group, variable) + perturbation * brian.mV)
        brian.defaultclock.dt = float(stationarity["dt_ms"]) * brian.ms
        voltage_names = tuple(f"v_{item.name}" for item in compartments)
        state = brian.StateMonitor(
            group,
            voltage_names,
            record=True,
            dt=float(stationarity["recording_dt_ms"]) * brian.ms,
        )
        spikes = brian.SpikeMonitor(group)
        network = brian.Network(
            group,
            state,
            spikes,
            *group.contained_objects,
        )
        network.run(float(stationarity["duration_ms"]) * brian.ms)
        trajectories: dict[str, dict[str, float]] = {}
        maximum_peak_to_peak_mV = 0.0
        for name in voltage_names:
            values_mV = np.asarray(getattr(state, name)[0] / brian.mV, dtype=float)
            peak_to_peak_mV = float(np.ptp(values_mV))
            maximum_peak_to_peak_mV = max(
                maximum_peak_to_peak_mV, peak_to_peak_mV
            )
            trajectories[name.removeprefix("v_")] = {
                "initial_mV": float(values_mV[0]),
                "terminal_mV": float(values_mV[-1]),
                "minimum_mV": float(np.min(values_mV)),
                "maximum_mV": float(np.max(values_mV)),
                "peak_to_peak_mV": peak_to_peak_mV,
            }
        stationary = bool(
            maximum_peak_to_peak_mV
            <= float(stationarity["maximum_peak_to_peak_mV"])
        )
        stationarity_result = {
            "initial_residual_pA": fixed_point_residual_pA.tolist(),
            "initial_voltage_perturbation_mV": perturbation_mV.tolist(),
            "detector_event_count": int(spikes.count[0]),
            "detector_event_times_ms": np.asarray(
                spikes.t / brian.ms, dtype=float
            ).tolist(),
            "maximum_peak_to_peak_mV": maximum_peak_to_peak_mV,
            "trajectories": trajectories,
            "stationary_by_registered_probe": stationary,
        }

    has_perturbation = bool(
        stationarity_result
        and any(stationarity_result["initial_voltage_perturbation_mV"])
    )
    stationary = bool(
        stationarity_result
        and stationarity_result["stationary_by_registered_probe"]
    )
    if has_perturbation:
        status = (
            "fixed-point-stability-probe-pass"
            if stationary
            else "fixed-point-stability-probe-failed"
        )
    elif expected_fixed_point_mV is not None:
        status = "fixed-point-verification-complete"
    else:
        status = "exploratory-fixed-point-derivation-complete"
    artifact = {
        "schema_version": 1,
        "id": Path(args.output).stem,
        "date": datetime.now(tz=UTC).date().isoformat(),
        "status": status,
        "classification": "source-equation-initialization-audit",
        "profile": str(profile_path),
        "predecessor_artifact": profile["predecessor_artifact"],
        "runtime_fingerprint": conventions.fingerprint,
        "root_search": search,
        "attempt_count": len(guesses),
        "converged_attempt_count": converged_attempts,
        "unique_root_count": len(unique_roots),
        "unique_fixed_points_mV": [item.tolist() for item in unique_roots],
        "expected_fixed_point_pass": expected_fixed_point_pass,
        "stationarity_probe": stationarity,
        "stationarity_result": stationarity_result,
        "interpretation_limits": profile["interpretation_limits"],
        "next_gate": profile["next_gate"],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(artifact, sort_keys=False))
    print(
        f"attempts={len(guesses)} converged={converged_attempts} "
        f"unique_roots={len(unique_roots)} stationary={stationary}",
        flush=True,
    )


if __name__ == "__main__":
    main()
