"""Network-blind conductance-transfer and rebound assays for SMART thalamus."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from ..models.adapter import SmartPopulationAdapter


@dataclass(frozen=True, slots=True)
class ConductanceReboundProtocol:
    equilibration_ms: float = 100.0
    baseline_ms: float = 100.0
    inhibition_ms: float = 100.0
    release_ms: float = 100.0
    early_release_ms: float = 25.0
    dt_ms: float = 0.02
    record_dt_ms: float = 0.1

    def __post_init__(self) -> None:
        values = (
            self.equilibration_ms,
            self.baseline_ms,
            self.inhibition_ms,
            self.release_ms,
            self.early_release_ms,
            self.dt_ms,
            self.record_dt_ms,
        )
        if any(not np.isfinite(value) or value <= 0 for value in values):
            raise ValueError("conductance-rebound durations must be finite and positive")
        if self.early_release_ms > self.release_ms:
            raise ValueError("early release window cannot exceed the release epoch")


@dataclass(frozen=True, slots=True)
class ConductanceReboundCondition:
    inhibition_scale: float
    baseline_spike_count: int
    inhibition_spike_count: int
    release_spike_count: int
    early_release_spike_count: int
    first_release_spike_latency_ms: float | None
    first_release_isi_ms: float | None
    baseline_mean_voltage_mV: float
    inhibition_mean_voltage_mV: float
    release_t_current_inward_charge_pA_ms: float
    baseline_t_current_inward_charge_pA_ms: float
    finite: bool


@dataclass(frozen=True, slots=True)
class ConductanceReboundResult:
    conditions: tuple[ConductanceReboundCondition, ...]


def _port_names(
    population: SmartPopulationAdapter,
    records: Mapping[str, float],
) -> dict[str, str]:
    ports = {port.record_id: port.name for port in population.compiled.synaptic_ports}
    missing = set(records) - set(ports)
    if missing:
        raise ValueError(f"population has no registered synaptic ports: {sorted(missing)}")
    return {record_id: ports[record_id] for record_id in records}


def run_conductance_rebound_protocol(
    *,
    population_factory,
    population_params: dict,
    inhibition_scales: Sequence[float],
    inhibitory_gate_baselines: Mapping[str, float],
    tonic_synaptic_gate_baselines: Mapping[str, float] | None = None,
    tonic_external_input: tuple[str, str, float] | None = None,
    protocol: ConductanceReboundProtocol | None = None,
    dynamics_seed: int | None = None,
    brian=None,
) -> ConductanceReboundResult:
    """Measure suppression and post-inhibitory release through SMART ports."""

    if brian is None:
        import brian2 as brian
    protocol = protocol or ConductanceReboundProtocol()
    scales = np.asarray(inhibition_scales, dtype=float)
    if (
        scales.ndim != 1
        or scales.size == 0
        or np.any(~np.isfinite(scales))
        or np.any(scales < 0)
    ):
        raise ValueError("inhibition scales must be a non-empty non-negative vector")
    if any(not np.isfinite(value) or value < 0 for value in inhibitory_gate_baselines.values()):
        raise ValueError("inhibitory gate baselines must be finite and non-negative")
    tonic_gates = tonic_synaptic_gate_baselines or {}
    if any(not np.isfinite(value) or value < 0 for value in tonic_gates.values()):
        raise ValueError("tonic gate baselines must be finite and non-negative")

    brian.start_scope()
    brian.defaultclock.dt = protocol.dt_ms * brian.ms
    population = population_factory(
        name="isolated_conductance_rebound",
        size=int(scales.size),
        params=population_params,
        brian=brian,
    )
    inhibitory_ports = _port_names(population, inhibitory_gate_baselines)
    tonic_ports = _port_names(population, tonic_gates)
    for record_id, baseline in tonic_gates.items():
        setattr(population.group, f"{tonic_ports[record_id]}_gate", float(baseline))
    if tonic_external_input is not None:
        record_id, channel, value = tonic_external_input
        population.set_external_input(record_id, channel, float(value))

    if dynamics_seed is not None:
        brian.seed(int(dynamics_seed))
    calcium_variables = tuple(
        f"i_ca_{name}"
        for name in population.compartments
        if f"i_ca_{name}" in population.group.variables
    )
    state_variables = ("v_soma", *calcium_variables)
    spikes = brian.SpikeMonitor(population.group)
    state = brian.StateMonitor(
        population.group,
        state_variables,
        record=True,
        dt=protocol.record_dt_ms * brian.ms,
    )
    network = brian.Network(population.group, spikes, state)
    network.run(protocol.equilibration_ms * brian.ms)
    baseline_start = protocol.equilibration_ms
    network.run(protocol.baseline_ms * brian.ms)
    inhibition_start = baseline_start + protocol.baseline_ms
    for record_id, baseline in inhibitory_gate_baselines.items():
        setattr(
            population.group,
            f"{inhibitory_ports[record_id]}_gate",
            float(baseline) * scales,
        )
    network.run(protocol.inhibition_ms * brian.ms)
    release_start = inhibition_start + protocol.inhibition_ms
    for record_id in inhibitory_gate_baselines:
        setattr(population.group, f"{inhibitory_ports[record_id]}_gate", 0.0)
    network.run(protocol.release_ms * brian.ms)

    spike_times = np.asarray(spikes.t / brian.ms, dtype=float)
    spike_indices = np.asarray(spikes.i, dtype=int)
    sample_times = np.asarray(state.t / brian.ms, dtype=float)
    voltage = np.asarray(state.v_soma / brian.mV, dtype=float)
    baseline_mask = (sample_times >= baseline_start) & (sample_times < inhibition_start)
    inhibition_mask = (sample_times >= inhibition_start) & (sample_times < release_start)
    release_mask = (sample_times >= release_start) & (
        sample_times < release_start + protocol.release_ms
    )
    calcium_pA = np.zeros_like(voltage)
    for variable in calcium_variables:
        calcium_pA += np.asarray(getattr(state, variable) / brian.pA, dtype=float)

    conditions = []
    for index, scale in enumerate(scales):
        times = spike_times[spike_indices == index]
        baseline = times[(times >= baseline_start) & (times < inhibition_start)]
        inhibited = times[(times >= inhibition_start) & (times < release_start)]
        release = times[
            (times >= release_start) & (times < release_start + protocol.release_ms)
        ]
        early = release[release < release_start + protocol.early_release_ms]
        release_latencies = release - release_start
        release_isis = np.diff(release_latencies)
        baseline_inward = np.maximum(-calcium_pA[index, baseline_mask], 0.0)
        release_inward = np.maximum(-calcium_pA[index, release_mask], 0.0)
        finite = bool(
            np.all(np.isfinite(voltage[index]))
            and np.all(np.isfinite(calcium_pA[index]))
        )
        conditions.append(
            ConductanceReboundCondition(
                inhibition_scale=float(scale),
                baseline_spike_count=int(baseline.size),
                inhibition_spike_count=int(inhibited.size),
                release_spike_count=int(release.size),
                early_release_spike_count=int(early.size),
                first_release_spike_latency_ms=(
                    None if release.size == 0 else float(release_latencies[0])
                ),
                first_release_isi_ms=(
                    None if release_isis.size == 0 else float(release_isis[0])
                ),
                baseline_mean_voltage_mV=float(np.mean(voltage[index, baseline_mask])),
                inhibition_mean_voltage_mV=float(np.mean(voltage[index, inhibition_mask])),
                release_t_current_inward_charge_pA_ms=float(
                    np.sum(release_inward) * protocol.record_dt_ms
                ),
                baseline_t_current_inward_charge_pA_ms=float(
                    np.sum(baseline_inward) * protocol.record_dt_ms
                ),
                finite=finite,
            )
        )
    return ConductanceReboundResult(tuple(conditions))


def classic_rebound_valid(
    condition: ConductanceReboundCondition,
    *,
    protocol: ConductanceReboundProtocol,
) -> bool:
    """Operational gate fixed before the classic validity scan."""

    baseline_early_expectation = (
        condition.baseline_spike_count * protocol.early_release_ms / protocol.baseline_ms
    )
    return bool(
        condition.inhibition_scale > 0
        and condition.finite
        and condition.baseline_spike_count >= 2
        and condition.inhibition_spike_count <= condition.baseline_spike_count / 2
        and condition.early_release_spike_count >= max(2, baseline_early_expectation + 1)
        and condition.first_release_spike_latency_ms is not None
        and condition.first_release_spike_latency_ms <= protocol.early_release_ms
        and condition.first_release_isi_ms is not None
        and condition.first_release_isi_ms <= 10.0
        and condition.release_t_current_inward_charge_pA_ms
        > condition.baseline_t_current_inward_charge_pA_ms
    )
