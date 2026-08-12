"""Reproducible first-order SMART bar-protocol validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..classic_sector import (
    FirstOrderRuntimeConventions,
    build_first_order_connected_sector,
    first_order_population_parameters,
)
from ..models.compartmental_hh import create_compartmental_hh_population
from ..models.modeldb112923 import first_order_population_facts
from ..protocols import BarOrientation, ClassicBarStimulus, apply_bar_stimulus


@dataclass(frozen=True, slots=True)
class FirstOrderBarProtocol:
    orientation: BarOrientation = BarOrientation.HORIZONTAL
    warmup_ms: float = 0.0
    stimulus_ms: float = 100.0
    dt_ms: float = 0.01
    source_value: float = 120.0

    def __post_init__(self) -> None:
        if self.warmup_ms < 0:
            raise ValueError("warmup_ms cannot be negative")
        if self.stimulus_ms <= 0 or self.dt_ms <= 0:
            raise ValueError("stimulus_ms and dt_ms must be positive")


@dataclass(frozen=True, slots=True)
class FirstOrderBarResult:
    conventions: FirstOrderRuntimeConventions
    protocol: FirstOrderBarProtocol
    warmup_spikes: dict[str, int]
    stimulus_spikes: dict[str, int]
    active_relay_spikes: tuple[int, ...]
    inactive_relay_spikes: int

    @property
    def convention_fingerprint(self) -> str:
        return self.conventions.fingerprint

    @property
    def active_relay_rates_hz(self) -> tuple[float, ...]:
        scale = 1000.0 / self.protocol.stimulus_ms
        return tuple(count * scale for count in self.active_relay_spikes)

    def as_document(self) -> dict[str, object]:
        """Return a serialization-safe validation record with exact inputs."""

        return {
            "convention_fingerprint": self.convention_fingerprint,
            "conventions": asdict(self.conventions),
            "protocol": {
                **asdict(self.protocol),
                "orientation": self.protocol.orientation.value,
            },
            "warmup_spikes": self.warmup_spikes,
            "stimulus_spikes": self.stimulus_spikes,
            "active_relay_spikes": list(self.active_relay_spikes),
            "active_relay_rates_hz": list(self.active_relay_rates_hz),
            "inactive_relay_spikes": self.inactive_relay_spikes,
        }


@dataclass(frozen=True, slots=True)
class FirstOrderBarAssessment:
    relay_rate_pass: bool
    selectivity_pass: bool
    target_rate_hz: float
    tolerance_hz: float
    observed_rates_hz: tuple[float, ...]
    inactive_relay_spikes: int

    @property
    def reproduced_drive(self) -> bool:
        return self.relay_rate_pass and self.selectivity_pass


@dataclass(frozen=True, slots=True)
class IsolatedRelayInputResult:
    conventions: FirstOrderRuntimeConventions
    duration_ms: float
    source_value: float
    spike_times_ms: tuple[float, ...]
    maximum_soma_voltage_mV: float
    final_soma_voltage_mV: float
    voltage_clamp_mV: float | None = None

    @property
    def numerically_valid(self) -> bool:
        return bool(
            np.isfinite(self.maximum_soma_voltage_mV)
            and np.isfinite(self.final_soma_voltage_mV)
        )

    @property
    def rate_hz(self) -> float:
        return len(self.spike_times_ms) * 1000.0 / self.duration_ms


def assess_first_order_bar(
    result: FirstOrderBarResult,
    *,
    target_rate_hz: float = 40.0,
    tolerance_hz: float = 5.0,
    maximum_inactive_spikes: int = 0,
) -> FirstOrderBarAssessment:
    """Score the predeclared Methods 4.9 relay-drive target."""

    rates = result.active_relay_rates_hz
    return FirstOrderBarAssessment(
        relay_rate_pass=all(abs(rate - target_rate_hz) <= tolerance_hz for rate in rates),
        selectivity_pass=result.inactive_relay_spikes <= maximum_inactive_spikes,
        target_rate_hz=target_rate_hz,
        tolerance_hz=tolerance_hz,
        observed_rates_hz=rates,
        inactive_relay_spikes=result.inactive_relay_spikes,
    )


def run_isolated_relay_input(
    *,
    conventions: FirstOrderRuntimeConventions,
    duration_ms: float = 100.0,
    source_value: float = 120.0,
    dt_ms: float = 0.01,
    voltage_clamp_mV: float | None = None,
    brian=None,
) -> IsolatedRelayInputResult:
    """Fast discriminator using the exact relay cell and archived input port."""

    if duration_ms <= 0 or dt_ms <= 0:
        raise ValueError("duration_ms and dt_ms must be positive")
    if brian is None:
        import brian2 as brian

    brian.start_scope()
    brian.defaultclock.dt = dt_ms * brian.ms
    facts = next(
        fact for fact in first_order_population_facts() if fact.canonical_name == "thalamic_relay"
    )
    parameters = first_order_population_parameters(facts, conventions=conventions)
    if voltage_clamp_mV is not None:
        parameters["voltage_clamps_mV"] = {"proximal_dendrite": voltage_clamp_mV}
    population = create_compartmental_hh_population(
        name="isolated_relay_input",
        size=1,
        params=parameters,
        brian=brian,
    )
    population.set_external_input(
        "modeldb112923.external.002",
        "green",
        source_value,
    )
    voltage = brian.StateMonitor(population.group, "v_soma", record=True)
    spikes = brian.SpikeMonitor(population.group)
    brian.Network(population.group, voltage, spikes).run(duration_ms * brian.ms)
    soma_mV = np.asarray(voltage.v_soma[0] / brian.mV)
    return IsolatedRelayInputResult(
        conventions=conventions,
        duration_ms=duration_ms,
        source_value=source_value,
        spike_times_ms=tuple(float(value) for value in np.asarray(spikes.t / brian.ms)),
        maximum_soma_voltage_mV=float(np.max(soma_mV)),
        final_soma_voltage_mV=float(soma_mV[-1]),
        voltage_clamp_mV=voltage_clamp_mV,
    )


def run_first_order_bar(
    *,
    conventions: FirstOrderRuntimeConventions | None = None,
    protocol: FirstOrderBarProtocol | None = None,
    brian=None,
) -> FirstOrderBarResult:
    """Run one exact connected 9×9 candidate and return source-level metrics."""

    if brian is None:
        import brian2 as brian

    conventions = conventions or FirstOrderRuntimeConventions()
    protocol = protocol or FirstOrderBarProtocol()
    brian.start_scope()
    brian.defaultclock.dt = protocol.dt_ms * brian.ms
    sector = build_first_order_connected_sector(conventions=conventions, brian=brian)
    monitors = {
        name: brian.SpikeMonitor(population.group, name=f"bar_spikes_{name}")
        for name, population in sector.populations.items()
    }
    sector.network.add(*monitors.values())
    if protocol.warmup_ms:
        sector.network.run(protocol.warmup_ms * brian.ms)
    warmup = {name: int(monitor.num_spikes) for name, monitor in monitors.items()}
    stimulus = ClassicBarStimulus(
        protocol.orientation,
        duration_ms=protocol.stimulus_ms,
        source_value=protocol.source_value,
    )
    apply_bar_stimulus(sector, stimulus)
    sector.network.run(protocol.stimulus_ms * brian.ms)
    stimulus_counts = {
        name: int(monitor.num_spikes) - warmup[name] for name, monitor in monitors.items()
    }
    relay = monitors["thalamic_relay"]
    times_ms = np.asarray(relay.t / brian.ms)
    indices = np.asarray(relay.i)
    stimulus_mask = times_ms >= protocol.warmup_ms
    relay_counts = np.bincount(indices[stimulus_mask], minlength=81)
    active = tuple(int(relay_counts[index]) for index in stimulus.active_indices)
    inactive = int(relay_counts.sum() - sum(active))
    return FirstOrderBarResult(
        conventions=conventions,
        protocol=protocol,
        warmup_spikes=warmup,
        stimulus_spikes=stimulus_counts,
        active_relay_spikes=active,
        inactive_relay_spikes=inactive,
    )
