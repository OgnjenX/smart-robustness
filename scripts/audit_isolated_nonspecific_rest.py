"""Audit whether the recovered nonspecific cell relaxes to quiescent rest."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from smart_robustness.classic_sector import first_order_population_parameters
from smart_robustness.models.compartmental_hh import create_compartmental_hh_population
from smart_robustness.models.modeldb112923 import first_order_population_facts
from smart_robustness.validation.calibration import runtime_conventions_for_candidate


def _float_mV(values: Any, brian: Any) -> np.ndarray:
    return np.asarray(values / brian.mV, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        default="configs/calibration/isolated_nonspecific_rest_audit_v1.yaml",
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
    population = create_compartmental_hh_population(
        name="isolated_nonspecific_rest_audit",
        size=1,
        params=params,
        brian=brian,
    )

    protocol = profile["protocol"]
    duration_ms = float(protocol["duration_ms"])
    dt_ms = float(protocol["dt_ms"])
    recording_dt_ms = float(protocol["recording_dt_ms"])
    terminal_window_ms = float(protocol["terminal_window_ms"])
    brian.defaultclock.dt = dt_ms * brian.ms
    voltage_names = tuple(
        f"v_{compartment.name}" for compartment in population.cell_spec.compartments
    )
    state = brian.StateMonitor(
        population.group,
        voltage_names,
        record=True,
        dt=recording_dt_ms * brian.ms,
    )
    spikes = brian.SpikeMonitor(population.group)
    network = brian.Network(
        population.group,
        state,
        spikes,
        *population.group.contained_objects,
    )
    network.run(duration_ms * brian.ms)

    terminal_samples = round(terminal_window_ms / recording_dt_ms)
    compartments: dict[str, dict[str, float | bool]] = {}
    all_finite = True
    max_terminal_peak_to_peak_mV = 0.0
    for variable in voltage_names:
        values_mV = _float_mV(getattr(state, variable)[0], brian)
        terminal_mV = values_mV[-terminal_samples:]
        finite = bool(np.all(np.isfinite(values_mV)))
        all_finite = all_finite and finite
        peak_to_peak_mV = float(np.ptp(terminal_mV)) if finite else float("nan")
        if finite:
            max_terminal_peak_to_peak_mV = max(
                max_terminal_peak_to_peak_mV, peak_to_peak_mV
            )
        compartments[variable.removeprefix("v_")] = {
            "finite": finite,
            "initial_mV": float(values_mV[0]),
            "terminal_mV": float(values_mV[-1]),
            "terminal_min_mV": float(np.min(terminal_mV)),
            "terminal_max_mV": float(np.max(terminal_mV)),
            "terminal_peak_to_peak_mV": peak_to_peak_mV,
        }

    spike_times_ms = np.asarray(spikes.t / brian.ms, dtype=float)
    terminal_start_ms = duration_ms - terminal_window_ms
    terminal_events = int(np.count_nonzero(spike_times_ms >= terminal_start_ms))
    gate = profile["operational_gate"]
    quiescent = bool(
        all_finite
        and max_terminal_peak_to_peak_mV
        <= float(gate["terminal_peak_to_peak_at_most_mV"])
        and terminal_events == int(gate["terminal_detector_events"])
    )
    artifact = {
        "schema_version": 1,
        "id": Path(args.output).stem,
        "date": datetime.now(tz=UTC).date().isoformat(),
        "status": (
            "quiescent-rest-supported"
            if quiescent
            else "quiescent-rest-not-supported"
        ),
        "classification": "source-interpretation-audit",
        "profile": str(profile_path),
        "registration_artifact": profile["registration_artifact"],
        "runtime_fingerprint": conventions.fingerprint,
        "population": profile["population"],
        "protocol": protocol,
        "detector_event_count": int(spikes.count[0]),
        "detector_event_times_ms": spike_times_ms.tolist(),
        "terminal_detector_event_count": terminal_events,
        "all_state_samples_finite": all_finite,
        "maximum_terminal_peak_to_peak_mV": max_terminal_peak_to_peak_mV,
        "compartments": compartments,
        "quiescent_rest_supported": quiescent,
        "next_gate": profile["next_gate"],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(artifact, sort_keys=False))
    print(
        f"events={int(spikes.count[0])} terminal_events={terminal_events} "
        f"terminal_peak_to_peak_mV={max_terminal_peak_to_peak_mV:.6f} "
        f"quiescent={quiescent}",
        flush=True,
    )


if __name__ == "__main__":
    main()
