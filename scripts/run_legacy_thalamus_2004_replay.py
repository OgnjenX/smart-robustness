"""Replay the preregistered 2004 KInNeSS relay tonic/burst benchmark."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import yaml

from smart_robustness.models.compartmental_hh import create_compartmental_hh_population
from smart_robustness.models.ports import ExternalInputPortSpec
from smart_robustness.models.table3 import CellSpec, CompartmentSpec
from smart_robustness.validation.isolated_cells import (
    IsolatedCellTrace,
    figure8_voltage_peak_times_ms,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _legacy_relay_cell() -> CellSpec:
    """Return the two-compartment relay cell printed in the 2004 benchmark."""

    return CellSpec(
        "kinness_2004_relay",
        (
            CompartmentSpec(
                "soma", 0.01, 0.05, 20.0, -70.0, 0.1, 50.0, 30.0, 250.0
            ),
            CompartmentSpec(
                "proximal_dendrite", 0.01, 0.05, 20.0, -70.0, 0.1, None, None, 250.0
            ),
        ),
    )


def _legacy_input_ports() -> tuple[ExternalInputPortSpec, ...]:
    return (
        ExternalInputPortSpec(
            name="legacy_hyperpolarization",
            record_id="kinness2004.hyperpolarization",
            compartment="soma",
            conductance_density_mS_cm2=0.4,
            reversal_mV=0.0,
            sensitivities_mV=(-250.0, 0.0, 0.0, 0.0),
        ),
        ExternalInputPortSpec(
            name="legacy_depolarization",
            record_id="kinness2004.depolarization",
            compartment="proximal_dendrite",
            conductance_density_mS_cm2=0.9,
            reversal_mV=0.0,
            sensitivities_mV=(0.0, 250.0, 0.0, 0.0),
        ),
    )


def _model_params(method: str) -> dict[str, object]:
    return {
        "cell_spec": _legacy_relay_cell(),
        "cell_class": "thalamic_relay",
        "axial_convention": "kinness_serialized_edge",
        "leak_convention": "table3_reversal",
        "voltage_coordinate": "relative_to_table3_leak",
        "nak_rate_convention": "standard_traub_miles",
        "calcium_gate_convention": "modeldb_112923",
        "calcium_voltage_coordinate": "integrated_voltage",
        "gate_initialization_convention": "steady_state_at_initial_voltage",
        "membrane_initialization_convention": "physical_leak_voltage",
        "spike_event_coordinate": "absolute_physical",
        "spike_event_threshold_mV": -20.0,
        "spike_event_rule": "falling_threshold_crossing",
        "calcium_density_convention": "table3",
        "ahp_convention": "modeldb_112923",
        "specific_capacitance_uF_cm2": 1.0,
        "enable_ahp_ach": False,
        "external_input_ports": _legacy_input_ports(),
        "method": method,
        "e_na_mV": 50.0,
        "e_k_mV": -90.0,
        "e_ca_mV": 180.0,
    }


def _set_port(group, name: str, channel: str, value: float) -> None:
    for other in ("red", "green", "blue", "alpha"):
        setattr(group, f"{name}_input_{other}", value if other == channel else 0.0)
    setattr(group, f"{name}_input_source_count", 1.0)


def _trace(group, voltage, spikes, brian) -> IsolatedCellTrace:
    return IsolatedCellTrace(
        condition="legacy_replay",
        time_ms=np.asarray(voltage.t / brian.ms, dtype=float),
        soma_voltage_mV=np.asarray(voltage.v_soma[0] / brian.mV, dtype=float),
        spike_times_ms=np.asarray(spikes.t / brian.ms, dtype=float),
    )


def _run_tonic(*, input_value: float, method: str, duration_ms: float, dt_ms: float, brian):
    brian.start_scope()
    brian.defaultclock.dt = dt_ms * brian.ms
    population = create_compartmental_hh_population(
        name="legacy_2004_tonic", size=1, params=_model_params(method), brian=brian
    )
    _set_port(population.group, "legacy_depolarization", "green", input_value)
    voltage = brian.StateMonitor(population.group, "v_soma", record=True)
    spikes = brian.SpikeMonitor(population.group)
    brian.Network(population.group, voltage, spikes).run(duration_ms * brian.ms)
    return _trace(population.group, voltage, spikes, brian)


def _run_burst(
    *,
    input_value: float,
    method: str,
    duration_ms: float,
    hyperpolarization_on_ms: float,
    hyperpolarization_off_ms: float,
    dt_ms: float,
    brian,
):
    brian.start_scope()
    brian.defaultclock.dt = dt_ms * brian.ms
    population = create_compartmental_hh_population(
        name="legacy_2004_burst", size=1, params=_model_params(method), brian=brian
    )
    voltage = brian.StateMonitor(population.group, "v_soma", record=True)
    spikes = brian.SpikeMonitor(population.group)
    network = brian.Network(population.group, voltage, spikes)
    network.run(hyperpolarization_on_ms * brian.ms)
    _set_port(population.group, "legacy_hyperpolarization", "red", input_value)
    network.run((hyperpolarization_off_ms - hyperpolarization_on_ms) * brian.ms)
    _set_port(population.group, "legacy_hyperpolarization", "red", 0.0)
    network.run((duration_ms - hyperpolarization_off_ms) * brian.ms)
    return _trace(population.group, voltage, spikes, brian)


def _summary(trace: IsolatedCellTrace) -> dict[str, object]:
    events = np.asarray(trace.spike_times_ms, dtype=float)
    peaks = figure8_voltage_peak_times_ms(
        trace, soma_leak_mV=-70.0, relative_threshold_mV=30.0
    )
    intervals = np.diff(events)
    return {
        "event_count": int(events.size),
        "event_times_ms": [float(value) for value in events],
        "voltage_peak_count": int(peaks.size),
        "voltage_peak_times_ms": [float(value) for value in peaks],
        "median_interevent_interval_ms": (
            float(np.median(intervals)) if intervals.size else None
        ),
        "minimum_soma_voltage_mV": float(np.min(trace.soma_voltage_mV)),
        "maximum_soma_voltage_mV": float(np.max(trace.soma_voltage_mV)),
        "finite": bool(np.all(np.isfinite(trace.soma_voltage_mV))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True)
    args = parser.parse_args()
    registration_path = Path(args.registration)
    registration = yaml.safe_load(registration_path.read_text())

    for source in registration["sources"].values():
        path = Path(source["path"])
        if _sha256(path) != source["sha256"]:
            raise ValueError(f"source differs from registration: {path}")
    script_path = Path(registration["script"])
    if _sha256(script_path) != registration["script_sha256"]:
        raise ValueError("runner differs from registration")

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    protocol = registration["protocol"]
    outcomes = []
    for method in protocol["integration_methods"]:
        for input_mode, input_value in protocol["input_encodings"].items():
            tonic = _run_tonic(
                input_value=float(input_value),
                method=method,
                duration_ms=float(protocol["duration_ms"]),
                dt_ms=float(protocol["dt_ms"]),
                brian=brian,
            )
            burst = _run_burst(
                input_value=float(input_value),
                method=method,
                duration_ms=float(protocol["duration_ms"]),
                hyperpolarization_on_ms=float(protocol["hyperpolarization_on_ms"]),
                hyperpolarization_off_ms=float(protocol["hyperpolarization_off_ms"]),
                dt_ms=float(protocol["dt_ms"]),
                brian=brian,
            )
            outcomes.append(
                {
                    "integration_method": method,
                    "input_encoding": input_mode,
                    "input_value": float(input_value),
                    "tonic": _summary(tonic),
                    "burst": _summary(burst),
                }
            )

    result = {
        "schema_version": 1,
        "id": registration["result_id"],
        "date": registration["date"],
        "status": "completed-legacy-thalamus-2004-replay",
        "classification": registration["classification"],
        "registration": str(registration_path),
        "archived_targets": registration["archived_targets"],
        "outcomes": outcomes,
        "selection_performed": False,
        "original_figure8_reproduced": False,
        "baseline_frozen": False,
    }
    Path(registration["result"]).write_text(yaml.safe_dump(result, sort_keys=False))


if __name__ == "__main__":
    main()
