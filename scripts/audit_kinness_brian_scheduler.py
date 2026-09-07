"""Audit the classic SMART event schedule against KInNeSS documentation."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from smart_robustness.classic_sector import build_first_order_connected_sector
from smart_robustness.validation.calibration import runtime_conventions_for_candidate


def _slot(obj: Any) -> dict[str, int | str]:
    return {"when": str(obj.when), "order": int(obj.order)}


def _first(values: list[float]) -> float | None:
    return None if not values else float(values[0])


def _microprobe(*, brian: Any, dt_ms: float, delay_ms: float) -> dict[str, Any]:
    """Measure when a delayed event first becomes a summed postsynaptic gate."""

    brian.defaultclock.dt = dt_ms * brian.ms
    source = brian.SpikeGeneratorGroup(1, [0], [0] * brian.ms)
    target = brian.NeuronGroup(1, "gate : 1")
    projection = brian.Synapses(
        source,
        target,
        model="last_amplitude : 1\ngate_post=last_amplitude : 1 (summed)",
        on_pre="last_amplitude=1",
    )
    projection.connect()
    projection.last_amplitude = 0
    projection.delay = delay_ms * brian.ms
    monitor = brian.StateMonitor(target, "gate", record=0, when="end")
    network = brian.Network(source, target, projection, monitor)
    network.run((delay_ms + 5 * dt_ms) * brian.ms)
    times_ms = np.asarray(monitor.t / brian.ms)
    gate = np.asarray(monitor.gate[0])
    positive = np.flatnonzero(gate > 0)
    first_gate_time_ms = None if not positive.size else float(times_ms[positive[0]])
    return {
        "dt_ms": dt_ms,
        "serialized_delay_ms": delay_ms,
        "event_time_ms": 0.0,
        "gate_at_serialized_arrival": float(
            gate[int(np.argmin(np.abs(times_ms - delay_ms)))]
        ),
        "first_positive_gate_time_ms": first_gate_time_ms,
        "expected_next_step_time_ms": delay_ms + dt_ms,
        "next_step_delivery_pass": bool(
            first_gate_time_ms is not None
            and np.isclose(first_gate_time_ms, delay_ms + dt_ms)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True)
    args = parser.parse_args()

    registration = yaml.safe_load(Path(args.registration).read_text())
    match = yaml.safe_load(Path(registration["match_result"]).read_text())[
        "match_result"
    ]
    profile = yaml.safe_load(Path(registration["profile"]).read_text())
    training_profile = yaml.safe_load(Path(profile["training_profile"]).read_text())
    base_profile = yaml.safe_load(Path(training_profile["base_profile"]).read_text())
    conventions = replace(
        runtime_conventions_for_candidate(base_profile["candidate"]),
        **training_profile["runtime_overrides"],
    )
    if conventions.fingerprint != registration["runtime_fingerprint"]:
        raise ValueError("registration runtime differs from executable profile")

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    brian.BrianLogger.suppress_name("unused_brian_object")
    dt_ms = float(profile["protocol"]["dt_ms"])
    brian.start_scope()
    brian.defaultclock.dt = dt_ms * brian.ms
    sector = build_first_order_connected_sector(conventions=conventions, brian=brian)
    projection = sector.projections["modeldb112923.projection.047"]
    nonspecific = sector.populations["thalamic_nonspecific"].group
    previous_voltage_runner = next(
        obj
        for obj in nonspecific.contained_objects
        if obj.when == "end" and obj.order == 1
    )
    summed_updater = projection.summed_updaters["port_000_gate_post"]

    first_upstream_event_ms = min(
        value
        for value in (
            _first(match["trn_spike_times_ms"]),
            _first(match["relay_spike_times_ms"]),
            _first(match["category_spike_times_ms"]),
        )
        if value is not None
    )
    nonspecific_events = [float(value) for value in match["nonspecific_spike_times_ms"]]
    pre_upstream_events = [
        value for value in nonspecific_events if value < first_upstream_event_ms
    ]
    serialized_delay_ms = float(projection.delay[0] / brian.ms)
    microprobe = _microprobe(
        brian=brian,
        dt_ms=dt_ms,
        delay_ms=serialized_delay_ms,
    )

    artifact = {
        "schema_version": 1,
        "id": registration["result_id"],
        "date": registration["date"],
        "status": "scheduler-contract-verified",
        "classification": "source-runtime-semantics-audit",
        "registration": args.registration,
        "runtime_fingerprint": conventions.fingerprint,
        "runtime": {
            "brian_version": brian.__version__,
            "dt_ms": dt_ms,
            "network_schedule": list(sector.network.schedule),
            "representative_trn_to_nonspecific_projection": {
                "projection_id": "modeldb112923.projection.047",
                "serialized_delay_ms": serialized_delay_ms,
                "summed_gate_updater": _slot(summed_updater),
                "presynaptic_event_pathway": _slot(projection.pre),
            },
            "nonspecific_cell": {
                "state_updater": _slot(nonspecific.state_updater),
                "spike_thresholder": _slot(nonspecific.thresholder["spike"]),
                "spike_resetter": _slot(nonspecific.resetter["spike"]),
                "previous_voltage_capture": _slot(previous_voltage_runner),
            },
        },
        "delivery_microprobe": microprobe,
        "figure7_event_localization": {
            "match_result": registration["match_result"],
            "nonspecific_event_count": len(nonspecific_events),
            "first_trn_event_ms": _first(match["trn_spike_times_ms"]),
            "first_relay_event_ms": _first(match["relay_spike_times_ms"]),
            "first_category_event_ms": _first(match["category_spike_times_ms"]),
            "first_upstream_circuit_event_ms": first_upstream_event_ms,
            "nonspecific_events_before_first_upstream_event_ms": pre_upstream_events,
            "pre_upstream_nonspecific_event_count": len(pre_upstream_events),
            "post_upstream_nonspecific_event_count": (
                len(nonspecific_events) - len(pre_upstream_events)
            ),
        },
        "gates": {
            "registered_runtime_reconstructed": True,
            "default_schedule_order": list(sector.network.schedule)
            == ["start", "groups", "thresholds", "synapses", "resets", "end"],
            "summed_gate_precedes_cell_integration": (
                summed_updater.when == "groups"
                and summed_updater.order < nonspecific.state_updater.order
            ),
            "event_pathway_follows_threshold_detection": (
                projection.pre.when == "synapses"
                and nonspecific.thresholder["spike"].when == "thresholds"
            ),
            "previous_voltage_captured_after_event_processing": (
                previous_voltage_runner.when == "end"
            ),
            "delayed_gate_visible_on_next_integration_step": microprobe[
                "next_step_delivery_pass"
            ],
            "early_nonspecific_events_precede_network_feedback": (
                len(pre_upstream_events)
                == int(registration["fixed_checks"]["expected_early_events"])
            ),
        },
        "interpretation_boundary": registration["interpretation_boundary"],
    }
    print(yaml.safe_dump(artifact, sort_keys=False), end="")


if __name__ == "__main__":
    main()
