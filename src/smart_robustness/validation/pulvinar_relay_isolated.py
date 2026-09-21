"""Construction-only isolated SMART V2 relay assay; no run on import."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..classic_sector import first_order_population_parameters
from ..modeldb_projections import MODELDB_FULL
from ..models.compartmental_hh import create_compartmental_hh_population
from ..models.modeldb112923 import second_order_population_facts
from ..models.pulvinar_conductance import CONTROL_NAMES, conductance_controls
from ..models.pulvinar_stp import STPParameters

PROJECTION_ID = "modeldb112923.projection.065"
SETTLING_MS = 1000.0
RESPONSE_TAIL_MS = 100.0
SERIALIZED_PEAK_WEIGHT = 6.0
TYPE2 = STPParameters(0.8, 2.0, 3.33)


def response_input_arrays(frequency_hz: float, dt_ms: float) -> dict[str, np.ndarray]:
    """Five fixed dimensionless projection gates from emission through final tail."""
    if frequency_hz not in (0.5, 2, 5, 10, 20):
        raise ValueError("unregistered frequency")
    if dt_ms not in (0.01, 0.005):
        raise ValueError("unregistered integration step")
    emissions = np.arange(10, dtype=float) * (1000 / frequency_hz)
    duration = emissions[-1] + 0.1 + RESPONSE_TAIL_MS
    time = np.arange(round(duration / dt_ms) + 1, dtype=float) * dt_ms
    controls = conductance_controls(
        time,
        emissions,
        tau_ms=2.0,
        delay_ms=0.1,
        depletion_fraction=0.5,
        recovery_ms=100.0,
        type2_parameters=TYPE2,
    )
    return {
        "time_ms": time,
        "emissions_ms": emissions,
        "gate": np.column_stack([controls[name] for name in CONTROL_NAMES])
        * SERIALIZED_PEAK_WEIGHT,
    }


def merged_recording_intervals(frequency_hz: float) -> tuple[tuple[float, float], ...]:
    """Physical response intervals relative to first emission, merged exactly."""
    emissions = np.arange(10, dtype=float) * (1000 / frequency_hz)
    intervals: list[list[float]] = []
    for start in emissions:
        end = start + 0.1 + RESPONSE_TAIL_MS
        if intervals and start <= intervals[-1][1]:
            intervals[-1][1] = max(intervals[-1][1], end)
        else:
            intervals.append([float(start), float(end)])
    return tuple((start, end) for start, end in intervals)


@dataclass
class IsolatedPulvinarRelayAssay:
    network: Any
    population: Any
    state_monitor: Any
    spike_monitor: Any
    inputs: dict[str, np.ndarray]
    port_name: str


def build_isolated_pulvinar_relay_assay(
    *, baseline, frequency_hz: float, dt_ms: float, brian,
) -> IsolatedPulvinarRelayAssay:
    """Build five independent frozen relay cells driven through record 065 only."""
    inputs = response_input_arrays(frequency_hz, dt_ms)
    facts = next(
        fact for fact in second_order_population_facts()
        if fact.canonical_name == "thalamic_relay_v2"
    )
    parameters = first_order_population_parameters(
        facts, conventions=baseline.runtime_conventions(), catalog=MODELDB_FULL,
    )
    port = next(p for p in parameters["synaptic_ports"] if p.record_id == PROJECTION_ID)
    record = MODELDB_FULL.by_id(PROJECTION_ID)
    if (
        port.compartment != "proximal_dendrite"
        or port.name != "port_003"
        or record.weight != SERIALIZED_PEAK_WEIGHT
    ):
        raise ValueError("projection 065 source mapping changed")
    brian.defaultclock.dt = dt_ms * brian.ms
    population = create_compartmental_hh_population(
        name="isolated_pulvinar_relay", size=len(CONTROL_NAMES),
        params=parameters, brian=brian,
    )
    drive = brian.TimedArray(inputs["gate"], dt=dt_ms * brian.ms)
    population.group.namespace["pulvinar_assay_drive"] = drive
    population.group.run_regularly(
        f"{port.name}_gate = int(t >= {SETTLING_MS}*ms)"
        f"*pulvinar_assay_drive(t-{SETTLING_MS}*ms, i)",
        when="start", order=-1,
    )
    state = brian.StateMonitor(
        population.group,
        [
            "v_soma", "v_proximal_dendrite", "v_distal_dendrite",
            f"{port.name}_gate", f"i_{port.name}",
        ],
        record=True,
        when="end",
    )
    spikes = brian.SpikeMonitor(population.group)
    state.active = False
    network = brian.Network(population.group, state, spikes)
    return IsolatedPulvinarRelayAssay(network, population, state, spikes, inputs, port.name)
