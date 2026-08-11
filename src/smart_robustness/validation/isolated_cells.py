"""Predeclared isolated-cell protocols for SMART figure validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..models.compartmental_hh import create_compartmental_hh_population


@dataclass(frozen=True, slots=True)
class Figure8Protocol:
    """Source-backed parts of the Figure 8 relay-cell protocol.

    The paper specifies the 0.3 nA pulse and the presence/absence of a
    hyperpolarizing voltage clamp, but does not report the clamp voltage or
    exact epoch durations. Those values remain explicit calibration inputs.
    """

    pulse_pA: float = 300.0
    precondition_ms: float = 100.0
    pulse_ms: float = 300.0
    depolarized_hold_mV: float = -60.0
    hyperpolarized_hold_mV: float = -80.0
    hyperpolarizing_bias_pA: float = 0.0
    dt_ms: float = 0.01


@dataclass(frozen=True, slots=True)
class IsolatedCellTrace:
    condition: str
    time_ms: np.ndarray
    soma_voltage_mV: np.ndarray
    spike_times_ms: np.ndarray


@dataclass(frozen=True, slots=True)
class Figure8Assessment:
    tonic_pass: bool
    burst_pass: bool
    tonic_spike_count: int
    burst_spike_count: int
    notes: tuple[str, ...]

    @property
    def reproduced(self) -> bool:
        return self.tonic_pass and self.burst_pass


def run_figure8_condition(
    *,
    hyperpolarized: bool,
    model_params: dict[str, Any],
    protocol: Figure8Protocol | None = None,
    brian=None,
) -> IsolatedCellTrace:
    """Run one Figure 8 condition with a pre-pulse voltage clamp.

    The clamp is released when the published 0.3 nA pulse begins. This is a
    declared interpretation of the caption, not a recovered KInNeSS detail.
    """

    if brian is None:
        import brian2 as brian

    protocol = protocol or Figure8Protocol()
    brian.start_scope()
    brian.defaultclock.dt = protocol.dt_ms * brian.ms
    hold_mV = protocol.hyperpolarized_hold_mV if hyperpolarized else protocol.depolarized_hold_mV
    params = dict(model_params)
    params["cell_class"] = "thalamic_relay"
    params["v_init_mV"] = hold_mV
    population = create_compartmental_hh_population(
        name="figure8_relay", size=1, params=params, brian=brian
    )
    group = population.group

    def apply_voltage_clamp() -> None:
        for compartment in population.compartments:
            setattr(group, f"v_{compartment}", hold_mV * brian.mV)

    clamp = brian.NetworkOperation(apply_voltage_clamp, when="start")
    voltage = brian.StateMonitor(group, "v_soma", record=True)
    spikes = brian.SpikeMonitor(group)
    network = brian.Network(group, clamp, voltage, spikes)
    network.run(protocol.precondition_ms * brian.ms)
    pulse_start_ms = float(network.t / brian.ms)
    clamp.active = False
    pulse_pA = protocol.pulse_pA
    if hyperpolarized:
        pulse_pA += protocol.hyperpolarizing_bias_pA
    group.i_drive_soma = pulse_pA * brian.pA
    network.run(protocol.pulse_ms * brian.ms)

    times_ms = np.asarray(voltage.t / brian.ms)
    pulse_mask = times_ms >= pulse_start_ms
    spike_times_ms = np.asarray(spikes.t / brian.ms) - pulse_start_ms
    spike_times_ms = spike_times_ms[spike_times_ms >= 0]
    return IsolatedCellTrace(
        condition="hyperpolarized" if hyperpolarized else "depolarized",
        time_ms=times_ms[pulse_mask] - pulse_start_ms,
        soma_voltage_mV=np.asarray(voltage.v_soma[0] / brian.mV)[pulse_mask],
        spike_times_ms=spike_times_ms,
    )


def assess_figure8(
    tonic: IsolatedCellTrace,
    burst: IsolatedCellTrace,
    *,
    pulse_ms: float = 300.0,
) -> Figure8Assessment:
    """Score the qualitative Figure 8 signatures without fitting the traces."""

    tonic_spikes = tonic.spike_times_ms
    burst_spikes = burst.spike_times_ms
    tonic_sustained = len(tonic_spikes) >= 3 and tonic_spikes[-1] >= 0.6 * pulse_ms
    if len(tonic_spikes) >= 4:
        intervals = np.diff(tonic_spikes)
        tonic_regular = float(np.std(intervals) / np.mean(intervals)) < 0.5
    else:
        tonic_regular = False
    early_burst = int(np.sum(burst_spikes <= 80.0))
    late_burst = int(np.sum(burst_spikes > 120.0))
    burst_transient = 2 <= early_burst <= 10 and late_burst == 0
    notes: list[str] = []
    if not (tonic_sustained and tonic_regular):
        notes.append("depolarized condition is not a sustained regular tonic train")
    if not burst_transient:
        notes.append("hyperpolarized condition is not an early burst followed by silence")
    return Figure8Assessment(
        tonic_pass=tonic_sustained and tonic_regular,
        burst_pass=burst_transient,
        tonic_spike_count=len(tonic_spikes),
        burst_spike_count=len(burst_spikes),
        notes=tuple(notes),
    )
