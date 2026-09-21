"""Sealed isolated target-location assay for cortical GABA-A routing.

The registered assay changes only the compartment receiving one analytically
replayed projection-036 conductance.  Network execution is intentionally absent.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import replace
from itertools import combinations
from typing import Any

import numpy as np

from ..classic_sector import FirstOrderRuntimeConventions, first_order_population_parameters
from ..modeldb_projections import MODELDB_FIRST_ORDER
from ..models.compartmental_hh import create_compartmental_hh_population
from ..models.currents import biexponential_normalization
from ..models.modeldb112923 import first_order_population_facts
from ..models.ports import SynapticPortSpec
from ..projections import Receptor

ARMS = {
    "legacy_proximal": "proximal_dendrite",
    "pv_like_somatic": "soma",
    "sst_like_distal": "distal_dendrite",
}
PROTOCOLS = ("all_compartments_clamped_minus55", "free_membrane")
DT_MS = (0.01, 0.005)
EMISSIONS_MS = np.array([100.0, 120.0, 140.0, 160.0, 180.0])
ARRIVALS_MS = EMISSIONS_MS + 0.1
DEPLETION_EPSILON = 0.5
DEPLETION_RECOVERY_MS = 100.0
DURATION_MS = 250.0
REFERENCE_TARGET_AREA_CM2 = 0.00007853981633974484
REFERENCE_PORT_CONDUCTANCE_NS = 193.28648801211204
SERIALIZED_WEIGHT = 2.0
PORT_NAME = "port_000"
PORT_RECORD_ID = "isolated.projection036.target_location"


def projection036():
    """Return the immutable archived receptor template."""

    return next(
        record
        for record in MODELDB_FIRST_ORDER.projections
        if record.id == "modeldb112923.projection.036"
    )


def replay_resource_amplitudes() -> np.ndarray:
    """Pre-depletion transmitter resource at the registered emissions."""

    resource = 1.0
    previous_ms = float(EMISSIONS_MS[0])
    values = []
    for index, emission_ms in enumerate(EMISSIONS_MS):
        if index:
            elapsed_ms = float(emission_ms - previous_ms)
            resource = 1.0 - (1.0 - resource) * math.exp(
                -elapsed_ms / DEPLETION_RECOVERY_MS
            )
        values.append(resource)
        resource *= 1.0 - DEPLETION_EPSILON
        previous_ms = float(emission_ms)
    return np.array(values)


def _wave(elapsed_ms: float, amplitude: float, *, rise_ms: float, fall_ms: float) -> float:
    if elapsed_ms < 0:
        return 0.0
    normalization = biexponential_normalization(rise_ms, fall_ms)
    return float(
        amplitude
        * normalization
        * (
            math.exp(-min(elapsed_ms / fall_ms, 100.0))
            - math.exp(-min(elapsed_ms / rise_ms, 100.0))
        )
    )


def replay_gate(time_ms: float) -> float:
    """Archived last-two-arrival union gate, including serialized weight."""

    record = projection036()
    amplitudes = replay_resource_amplitudes()
    reached = int(np.searchsorted(ARRIVALS_MS, time_ms, side="right"))
    waves = []
    for index in range(max(0, reached - 2), reached):
        waves.append(
            _wave(
                time_ms - float(ARRIVALS_MS[index]),
                float(amplitudes[index]),
                rise_ms=float(record.rise_ms),
                fall_ms=float(record.fall_ms),
            )
        )
    if not waves:
        signal = 0.0
    elif len(waves) == 1:
        signal = waves[0]
    else:
        signal = waves[-1] + waves[-2] - waves[-1] * waves[-2]
    return SERIALIZED_WEIGHT * signal


def target_port(arm: str) -> SynapticPortSpec:
    """Clone projection 036 while conserving total port conductance."""

    if arm not in ARMS:
        raise ValueError(f"unknown target-location arm: {arm}")
    target = ARMS[arm]
    facts = next(
        fact
        for fact in first_order_population_facts()
        if fact.canonical_name == "layer5_excitatory_v1"
    )
    compartment = facts.cell.compartment(target)
    record = projection036()
    density = REFERENCE_PORT_CONDUCTANCE_NS / (compartment.lateral_area_cm2 * 1e6)
    return SynapticPortSpec(
        name=PORT_NAME,
        record_id=PORT_RECORD_ID,
        compartment=target,
        receptor=Receptor.GABA,
        reversal_mV=float(record.reversal_mV),
        conductance_density_mS_cm2=density,
        rise_ms=float(record.rise_ms),
        fall_ms=float(record.fall_ms),
        normalization=bieponential_normalization_checked(record),
        voltage_block=False,
    )


def bieponential_normalization_checked(record: Any) -> float:
    if record.rise_ms is None or record.fall_ms is None:
        raise ValueError("projection 036 lacks registered kinetics")
    return biexponential_normalization(float(record.rise_ms), float(record.fall_ms))


def isolated_parameters(
    arm: str,
    protocol: str,
    *,
    conventions: FirstOrderRuntimeConventions | None = None,
) -> dict[str, Any]:
    """Build one-cell parameters with only the registered GABA-A port."""

    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown target-location protocol: {protocol}")
    fact = next(
        item
        for item in first_order_population_facts()
        if item.canonical_name == "layer5_excitatory_v1"
    )
    params = first_order_population_parameters(
        fact, conventions=conventions or FirstOrderRuntimeConventions()
    )
    params["synaptic_ports"] = (target_port(arm),)
    params["gap_junction_ports"] = ()
    params["external_input_ports"] = ()
    params["injection_ports"] = ()
    params["depletion_epsilon"] = None
    params["depletion_recovery_ms"] = None
    params["voltage_clamps_mV"] = (
        {name: -55.0 for name in ("soma", "proximal_dendrite", "distal_dendrite")}
        if protocol == "all_compartments_clamped_minus55"
        else {}
    )
    return params


def simulate(
    *, baseline: Any, arm: str, protocol: str, dt_ms: float
) -> dict[str, np.ndarray]:
    """Run one fresh deterministic Brian2 runtime condition."""

    import brian2 as brian

    if dt_ms not in DT_MS:
        raise ValueError(f"unregistered time step: {dt_ms}")
    brian.start_scope()
    brian.defaultclock.dt = dt_ms * brian.ms
    population = create_compartmental_hh_population(
        name=f"isolated_inhibitory_{arm}_{protocol}",
        size=1,
        params=isolated_parameters(
            arm, protocol, conventions=baseline.runtime_conventions()
        ),
        brian=brian,
    )
    group = population.group

    @brian.network_operation(dt=dt_ms * brian.ms, when="start")
    def replay_operation():
        group.port_000_gate = replay_gate(float(brian.defaultclock.t / brian.ms))

    state = brian.StateMonitor(
        group,
        (
            "v_soma",
            "v_proximal_dendrite",
            "v_distal_dendrite",
            "port_000_gate",
            "i_port_000",
        ),
        record=True,
        when="end",
    )
    spikes = brian.SpikeMonitor(group)
    network = brian.Network(group, replay_operation, state, spikes)
    network.run(DURATION_MS * brian.ms)
    gate = np.asarray(state.port_000_gate[0], dtype=float)
    arrays = {
        "time_ms": np.asarray(state.t / brian.ms, dtype=float),
        "v_soma_mV": np.asarray(state.v_soma[0] / brian.mV, dtype=float),
        "v_proximal_dendrite_mV": np.asarray(
            state.v_proximal_dendrite[0] / brian.mV, dtype=float
        ),
        "v_distal_dendrite_mV": np.asarray(
            state.v_distal_dendrite[0] / brian.mV, dtype=float
        ),
        "target_gate": gate,
        "target_effective_conductance_nS": REFERENCE_PORT_CONDUCTANCE_NS * gate,
        "target_synaptic_current_pA": np.asarray(
            state.i_port_000[0] / brian.pA, dtype=float
        ),
        "spike_times_ms": np.asarray(spikes.t / brian.ms, dtype=float),
        "spike_indices": np.asarray(spikes.i, dtype=np.int64),
        "arrivals_ms": ARRIVALS_MS.copy(),
        "resource_amplitudes": replay_resource_amplitudes(),
        "dt_ms": np.array(dt_ms),
        "arm": np.array(arm),
        "protocol": np.array(protocol),
    }
    return arrays


def trace_sha256(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(arrays):
        value = np.asarray(arrays[key])
        digest.update(key.encode())
        digest.update(value.dtype.str.encode())
        digest.update(repr(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def exact_trace_repeat(first: dict[str, np.ndarray], second: dict[str, np.ndarray]) -> bool:
    return first.keys() == second.keys() and all(
        np.asarray(first[key]).dtype == np.asarray(second[key]).dtype
        and np.array_equal(first[key], second[key])
        for key in first
    )


def summarize(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    time = arrays["time_ms"]
    arrival = float(ARRIVALS_MS[0])
    pre = time < arrival
    post = time >= arrival
    current = arrays["target_synaptic_current_pA"]
    conductance = arrays["target_effective_conductance_nS"]
    arm = str(arrays["arm"])
    target_name = f"v_{ARMS[arm]}_mV"
    finite = all(
        bool(np.all(np.isfinite(value)))
        for value in arrays.values()
        if np.issubdtype(np.asarray(value).dtype, np.number)
    )
    voltage = {}
    for compartment in ("soma", "proximal_dendrite", "distal_dendrite"):
        values = arrays[f"v_{compartment}_mV"]
        voltage[compartment] = {
            "minimum_mV": float(np.min(values)),
            "maximum_mV": float(np.max(values)),
            "post_arrival_minimum_mV": float(np.min(values[post])),
            "post_arrival_maximum_mV": float(np.max(values[post])),
            "pre_arrival_peak_to_peak_mV": float(np.ptp(values[pre])),
        }
    return {
        "finite": finite,
        "trace_sha256": trace_sha256(arrays),
        "spike_count": int(arrays["spike_times_ms"].size),
        "peak_effective_conductance_nS": float(np.max(conductance)),
        "integral_effective_conductance_nS_ms": float(np.trapz(conductance, time)),
        "peak_absolute_current_pA": float(np.max(np.abs(current))),
        "integral_absolute_current_pA_ms": float(np.trapz(np.abs(current), time)),
        "target_voltage_post_arrival_range_mV": [
            float(np.min(arrays[target_name][post])),
            float(np.max(arrays[target_name][post])),
        ],
        "voltage": voltage,
    }


def _relative_difference(first: float, second: float) -> float:
    scale = max(abs(first), abs(second), np.finfo(float).tiny)
    return abs(first - second) / scale


def numerical_gate(coarse: dict[str, Any], fine: dict[str, Any]) -> dict[str, Any]:
    voltage_errors = {
        f"{compartment}_{bound}": abs(
            coarse["voltage"][compartment][bound]
            - fine["voltage"][compartment][bound]
        )
        for compartment in ("soma", "proximal_dendrite", "distal_dendrite")
        for bound in ("minimum_mV", "maximum_mV")
    }
    resource_errors = {
        name: _relative_difference(coarse[name], fine[name])
        for name in (
            "peak_effective_conductance_nS",
            "integral_effective_conductance_nS_ms",
        )
    }
    current_errors = {
        name: _relative_difference(coarse[name], fine[name])
        for name in ("peak_absolute_current_pA", "integral_absolute_current_pA_ms")
    }
    return {
        "voltage_absolute_errors_mV": voltage_errors,
        "resource_relative_errors": resource_errors,
        "current_relative_errors": current_errors,
        "pass": bool(
            max(voltage_errors.values()) <= 0.5
            and max(resource_errors.values()) <= 0.001
            and max(current_errors.values()) <= 0.01
        ),
    }


def cross_arm_gate(traces: dict[str, dict[str, np.ndarray]], *, protocol: str) -> dict[str, Any]:
    if set(traces) != set(ARMS):
        raise ValueError("cross-arm gate requires every registered arm")
    times = [traces[arm]["time_ms"] for arm in ARMS]
    if any(not np.array_equal(times[0], value) for value in times[1:]):
        raise ValueError("cross-arm time grids differ")
    if protocol == "all_compartments_clamped_minus55":
        conductance_equal = all(
            np.array_equal(
                traces[first]["target_effective_conductance_nS"],
                traces[second]["target_effective_conductance_nS"],
            )
            for first, second in combinations(ARMS, 2)
        )
        current_max_difference = max(
            float(
                np.max(
                    np.abs(
                        traces[first]["target_synaptic_current_pA"]
                        - traces[second]["target_synaptic_current_pA"]
                    )
                )
            )
            for first, second in combinations(ARMS, 2)
        )
        return {
            "effective_conductance_exact_across_arms": conductance_equal,
            "maximum_current_difference_pA": current_max_difference,
            "pass": conductance_equal and current_max_difference <= 1e-9,
        }
    if protocol != "free_membrane":
        raise ValueError(f"unknown protocol: {protocol}")
    pre = times[0] < float(ARRIVALS_MS[0])
    post = ~pre
    pre_equal = all(
        all(
            np.array_equal(
                traces[first][f"v_{compartment}_mV"][pre],
                traces[second][f"v_{compartment}_mV"][pre],
            )
            for compartment in ("soma", "proximal_dendrite", "distal_dendrite")
        )
        for first, second in combinations(ARMS, 2)
    )
    soma_difference = max(
        float(
            np.max(
                np.abs(
                    traces[first]["v_soma_mV"][post]
                    - traces[second]["v_soma_mV"][post]
                )
            )
        )
        for first, second in combinations(ARMS, 2)
    )
    target = {arm: traces[arm][f"v_{ARMS[arm]}_mV"] for arm in ARMS}
    target_difference = max(
        float(np.max(np.abs(target[first][post] - target[second][post])))
        for first, second in combinations(ARMS, 2)
    )
    return {
        "pre_arrival_voltage_traces_exact_across_arms": pre_equal,
        "maximum_post_arrival_soma_difference_mV": soma_difference,
        "maximum_post_arrival_target_difference_mV": target_difference,
        "soma_location_effect_detected": soma_difference >= 0.01,
        "target_location_effect_detected": target_difference >= 0.01,
        "pass": bool(pre_equal and soma_difference >= 0.01 and target_difference >= 0.01),
    }


def clone_with_compartment(port: SynapticPortSpec, compartment: str) -> SynapticPortSpec:
    """Test helper retaining immutable port fields except location."""

    return replace(port, compartment=compartment)
