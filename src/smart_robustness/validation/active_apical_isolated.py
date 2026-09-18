"""Construction-only isolated L5 assay for preregistered design 949.

No simulation runs on import or construction. Execution requires a separately
sealed runner. Each cell is an independent condition, with no synaptic network.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..classic_sector import first_order_population_facts, first_order_population_parameters
from ..modeldb_projections import MODELDB_FULL
from ..models.active_apical_integration import (
    DRIVE_FACTORS,
    arrival_gate,
    isolated_receptor_ports,
)
from ..models.compartmental_hh import create_compartmental_hh_population

INPUTS = ("none", "distal_only", "proximal_only", "distal_and_proximal")
SETTLING_MS = 1000.0
DURATION_MS = 1400.0
ARRIVALS_MS = np.array([1050.1, 1083.43, 1116.77])


@dataclass(frozen=True)
class AssayCase:
    drive_factor: float
    input_kind: str


def assay_cases() -> tuple[AssayCase, ...]:
    return tuple(AssayCase(drive, kind) for drive in DRIVE_FACTORS for kind in INPUTS)


def frozen_l5_parameters(baseline) -> dict:
    fact = next(
        f for f in first_order_population_facts() if f.canonical_name == "layer5_excitatory_v1"
    )
    return first_order_population_parameters(
        fact,
        conventions=baseline.runtime_conventions(),
        catalog=MODELDB_FULL,
    )


def input_gate_arrays(dt_ms: float) -> dict[str, np.ndarray]:
    """Shared unit-resource event history, with the registered 0.1-ms delay.

    Input times are sampled on the integration grid. The 0.005-ms refinement
    samples every event at the same physical arrival time as the 0.01-ms grid.
    """
    if dt_ms not in (0.01, 0.005):
        raise ValueError("unregistered integration step")
    time = np.arange(round(DURATION_MS / dt_ms) + 1) * dt_ms
    return {
        "time_ms": time,
        "fast": arrival_gate(time, ARRIVALS_MS, rise_ms=2.0, fall_ms=2.0),
        "slow": arrival_gate(time, ARRIVALS_MS, rise_ms=0.7, fall_ms=80.0),
    }


@dataclass
class IsolatedApicalAssay:
    network: Any
    population: Any
    state_monitor: Any
    spike_monitor: Any
    cases: tuple[AssayCase, ...]
    inputs: dict[str, np.ndarray]


def build_isolated_apical_assay(
    *,
    baseline,
    arm: str,
    resting_distal_mV: float,
    dt_ms: float,
    brian,
) -> IsolatedApicalAssay:
    """Construct all 24 isolated conditions; leave monitors initially inactive.

    The runner must record the last 100 ms of settling for rest checks and the
    full 400 ms response window. No cell shares a current, weight or state with
    another. Baseline intrinsic parameters and existing ports are unmodified.
    """
    inputs = input_gate_arrays(dt_ms)
    params = frozen_l5_parameters(baseline)
    cell = params["cell_spec"]
    ports = isolated_receptor_ports(
        arm,
        resting_distal_mV=resting_distal_mV,
        distal_area_cm2=cell.compartment("distal_dendrite").lateral_area_cm2,
        proximal_area_cm2=cell.compartment("proximal_dendrite").lateral_area_cm2,
    )
    params = {**params, "synaptic_ports": params["synaptic_ports"] + ports}
    brian.defaultclock.dt = dt_ms * brian.ms
    cases = assay_cases()
    pop = create_compartmental_hh_population(
        name="apical_isolated",
        size=len(cases),
        params=params,
        brian=brian,
    )
    # TimedArrays encode frozen drive factors, not fitted conductances. Fast and
    # slow kernels share exactly the same distal case mask and arrival history.
    distal = np.array(
        [
            c.drive_factor if c.input_kind in ("distal_only", "distal_and_proximal") else 0.0
            for c in cases
        ]
    )
    proximal = np.array(
        [
            c.drive_factor if c.input_kind in ("proximal_only", "distal_and_proximal") else 0.0
            for c in cases
        ]
    )
    namespace = {}
    code = []
    for port in ports:
        slow = port.name == "assay_slow"
        mask = proximal if port.name == "assay_proximal" else distal
        values = inputs["slow" if slow else "fast"][:, None] * mask[None, :]
        key = f"drive_{port.name}"
        namespace[key] = brian.TimedArray(values, dt=dt_ms * brian.ms)
        code.append(f"{port.name}_gate = {key}(t, i)")
    pop.group.namespace.update(namespace)
    pop.group.run_regularly("\n".join(code), when="start", order=-1)
    variables = ["v_soma", "v_proximal_dendrite", "v_distal_dendrite"]
    for port in ports:
        variables.extend([f"{port.name}_gate", f"i_{port.name}"])
    state = brian.StateMonitor(pop.group, variables, record=True, when="end")
    spikes = brian.SpikeMonitor(pop.group)
    state.active = False
    network = brian.Network(pop.group, state, spikes)
    return IsolatedApicalAssay(network, pop, state, spikes, cases, inputs)
