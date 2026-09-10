"""Network-blind isolated-cell targets for alternative neuron models."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..models.adapter import SmartPopulationAdapter
from ..models.adex import make_somatic_adex_factory
from ..models.adex_parameters import LITERATURE_REGULAR_SPIKING, AdExParameters
from ..models.table3 import CellSpec, get_cell_spec


@dataclass(frozen=True, slots=True)
class CurrentStepProtocol:
    training_currents_pA: tuple[float, ...] = (100.0, 300.0, 500.0, 700.0)
    holdout_currents_pA: tuple[float, ...] = (200.0, 400.0, 600.0)
    pre_ms: float = 100.0
    step_ms: float = 500.0
    post_ms: float = 100.0
    dt_ms: float = 0.01

    def __post_init__(self) -> None:
        currents = self.training_currents_pA + self.holdout_currents_pA
        if not currents or not all(np.isfinite(value) and value >= 0 for value in currents):
            raise ValueError("current steps must be finite and non-negative")
        if len(set(currents)) != len(currents):
            raise ValueError("training and holdout current steps must be disjoint")
        if self.pre_ms <= 0 or self.step_ms <= 0 or self.post_ms < 0 or self.dt_ms <= 0:
            raise ValueError("isolated-cell protocol durations must be positive")


@dataclass(frozen=True, slots=True)
class StepPhenotype:
    currents_pA: tuple[float, ...]
    resting_voltage_mV: float
    spike_counts: tuple[int, ...]
    firing_rates_hz: tuple[float, ...]
    first_spike_latencies_ms: tuple[float | None, ...]
    adaptation_ratios: tuple[float | None, ...]
    finite: bool


@dataclass(frozen=True, slots=True)
class CandidateFit:
    parameters: AdExParameters
    training_loss: float
    holdout_loss: float
    training_phenotype: StepPhenotype
    holdout_phenotype: StepPhenotype


@dataclass(frozen=True, slots=True)
class ReboundProtocol:
    baseline_ms: float = 100.0
    hyperpolarization_ms: float = 100.0
    release_ms: float = 100.0
    hyperpolarizing_passive_offset_mV: float = -100.0
    dt_ms: float = 0.02

    def __post_init__(self) -> None:
        if min(self.baseline_ms, self.hyperpolarization_ms, self.release_ms, self.dt_ms) <= 0:
            raise ValueError("rebound durations and dt must be positive")
        if self.hyperpolarizing_passive_offset_mV >= 0:
            raise ValueError("rebound current must be hyperpolarizing")


@dataclass(frozen=True, slots=True)
class ReboundPhenotype:
    baseline_spike_count: int
    hyperpolarization_spike_count: int
    release_spike_count: int
    baseline_rate_hz: float
    release_rate_hz: float
    first_release_spike_latency_ms: float | None
    finite: bool

    @property
    def rebound_present(self) -> bool:
        return self.release_rate_hz >= self.baseline_rate_hz + 10.0


@dataclass(frozen=True, slots=True)
class ReboundAssessment:
    finite: bool
    rebound_presence_matches: bool
    release_rate_matches: bool
    first_latency_matches: bool

    @property
    def promoted(self) -> bool:
        return bool(
            self.finite
            and self.rebound_presence_matches
            and self.release_rate_matches
            and self.first_latency_matches
        )


@dataclass(frozen=True, slots=True)
class ExternalInputProtocol:
    pre_ms: float = 50.0
    step_ms: float = 100.0
    post_ms: float = 20.0
    dt_ms: float = 0.02

    def __post_init__(self) -> None:
        if self.pre_ms <= 0 or self.step_ms <= 0 or self.post_ms < 0 or self.dt_ms <= 0:
            raise ValueError("external-input durations must be positive")


@dataclass(frozen=True, slots=True)
class ExternalInputPhenotype:
    source_values: tuple[float, ...]
    resting_voltage_mV: float
    spike_counts: tuple[int, ...]
    firing_rates_hz: tuple[float, ...]
    first_spike_latencies_ms: tuple[float | None, ...]
    peak_voltages_mV: tuple[float, ...]
    mean_voltages_mV: tuple[float, ...]
    finite: bool


@dataclass(frozen=True, slots=True)
class RelayInhibitionProtocol:
    pre_ms: float = 50.0
    step_ms: float = 100.0
    post_ms: float = 20.0
    dt_ms: float = 0.02


@dataclass(frozen=True, slots=True)
class RelayInhibitionPhenotype:
    source_values: tuple[float, ...]
    inhibition_scales: tuple[float, ...]
    resting_voltage_mV: float
    spike_counts: tuple[int, ...]
    firing_rates_hz: tuple[float, ...]
    first_spike_latencies_ms: tuple[float | None, ...]
    mean_voltages_mV: tuple[float, ...]
    finite: bool


PopulationFactory = Callable[..., SmartPopulationAdapter]

_FITTED_PARAMETER_NAMES = (
    "threshold_offset_mV",
    "slope_factor_mV",
    "reset_offset_mV",
    "subthreshold_adaptation_nS",
    "spike_adaptation_pA",
    "adaptation_time_constant_ms",
)

_EXTENDED_FITTED_PARAMETER_NAMES = (*_FITTED_PARAMETER_NAMES, "effective_leak_offset_mV")
_FULL_FITTED_PARAMETER_NAMES = (
    *_EXTENDED_FITTED_PARAMETER_NAMES,
    "somatic_capacitance_scale",
    "somatic_leak_conductance_scale",
)


def generate_adex_sobol_candidates(
    bounds: dict[str, tuple[float, float]],
    *,
    count: int = 32,
    include_literature: bool = True,
) -> tuple[AdExParameters, ...]:
    """Generate the preregistered deterministic, network-blind search set."""

    from scipy.stats import qmc

    if set(bounds) != set(_FITTED_PARAMETER_NAMES):
        raise ValueError(f"AdEx candidate bounds must be exactly {_FITTED_PARAMETER_NAMES}")
    if count <= 0 or count & (count - 1):
        raise ValueError("Sobol candidate count must be a positive power of two")
    lower = np.asarray([bounds[name][0] for name in _FITTED_PARAMETER_NAMES], dtype=float)
    upper = np.asarray([bounds[name][1] for name in _FITTED_PARAMETER_NAMES], dtype=float)
    if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)) or np.any(lower >= upper):
        raise ValueError("each AdEx bound must be finite and increasing")
    samples = qmc.Sobol(d=len(_FITTED_PARAMETER_NAMES), scramble=False).random_base2(
        m=int(np.log2(count))
    )
    scaled = qmc.scale(samples, lower, upper)
    candidates = [
        AdExParameters(
            **{
                name: float(value)
                for name, value in zip(_FITTED_PARAMETER_NAMES, row, strict=True)
            },
            peak_mV=20.0,
            refractory_ms=2.0,
        )
        for row in scaled
    ]
    if include_literature and LITERATURE_REGULAR_SPIKING not in candidates:
        candidates.append(LITERATURE_REGULAR_SPIKING)
    return tuple(candidates)


def generate_extended_adex_sobol_candidates(
    bounds: dict[str, tuple[float, float]],
    *,
    count: int = 128,
    include_literature: bool = True,
) -> tuple[AdExParameters, ...]:
    """Generate candidates that also fit the standard AdEx effective leak."""

    from scipy.stats import qmc

    if set(bounds) != set(_EXTENDED_FITTED_PARAMETER_NAMES):
        raise ValueError(
            "extended AdEx bounds must be exactly "
            f"{_EXTENDED_FITTED_PARAMETER_NAMES}"
        )
    if count <= 0 or count & (count - 1):
        raise ValueError("Sobol candidate count must be a positive power of two")
    lower = np.asarray(
        [bounds[name][0] for name in _EXTENDED_FITTED_PARAMETER_NAMES], dtype=float
    )
    upper = np.asarray(
        [bounds[name][1] for name in _EXTENDED_FITTED_PARAMETER_NAMES], dtype=float
    )
    if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)) or np.any(lower >= upper):
        raise ValueError("each AdEx bound must be finite and increasing")
    samples = qmc.Sobol(
        d=len(_EXTENDED_FITTED_PARAMETER_NAMES), scramble=False
    ).random_base2(m=int(np.log2(count)))
    scaled = qmc.scale(samples, lower, upper)
    candidates = [
        AdExParameters(
            **{
                name: float(value)
                for name, value in zip(
                    _EXTENDED_FITTED_PARAMETER_NAMES, row, strict=True
                )
            },
            peak_mV=20.0,
            refractory_ms=2.0,
        )
        for row in scaled
    ]
    if include_literature and LITERATURE_REGULAR_SPIKING not in candidates:
        candidates.append(LITERATURE_REGULAR_SPIKING)
    return tuple(candidates)


def generate_full_adex_sobol_candidates(
    bounds: dict[str, tuple[float, float]],
    *,
    count: int = 512,
    include_literature: bool = True,
) -> tuple[AdExParameters, ...]:
    """Generate candidates over the complete somatic AdEx parameterization."""

    from scipy.stats import qmc

    if set(bounds) != set(_FULL_FITTED_PARAMETER_NAMES):
        raise ValueError(f"full AdEx bounds must be exactly {_FULL_FITTED_PARAMETER_NAMES}")
    if count <= 0 or count & (count - 1):
        raise ValueError("Sobol candidate count must be a positive power of two")
    lower = np.asarray(
        [bounds[name][0] for name in _FULL_FITTED_PARAMETER_NAMES], dtype=float
    )
    upper = np.asarray(
        [bounds[name][1] for name in _FULL_FITTED_PARAMETER_NAMES], dtype=float
    )
    if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)) or np.any(lower >= upper):
        raise ValueError("each AdEx bound must be finite and increasing")
    samples = qmc.Sobol(d=len(_FULL_FITTED_PARAMETER_NAMES), scramble=False).random_base2(
        m=int(np.log2(count))
    )
    scaled = qmc.scale(samples, lower, upper)
    candidates = [
        AdExParameters(
            **{
                name: float(value)
                for name, value in zip(_FULL_FITTED_PARAMETER_NAMES, row, strict=True)
            },
            peak_mV=20.0,
            refractory_ms=2.0,
        )
        for row in scaled
    ]
    if include_literature and LITERATURE_REGULAR_SPIKING not in candidates:
        candidates.append(LITERATURE_REGULAR_SPIKING)
    return tuple(candidates)


def passive_normalized_currents_pA(
    population_params: dict[str, Any],
    voltage_offsets_mV: Sequence[float],
) -> tuple[float, ...]:
    """Scale steps by somatic leak so amplitudes are comparable across cells."""

    cell = population_params.get("cell_spec")
    if cell is None:
        cell = get_cell_spec(str(population_params["cell_class"]))
    if not isinstance(cell, CellSpec):
        raise TypeError("cell_spec must be a CellSpec")
    offsets = np.asarray(voltage_offsets_mV, dtype=float)
    if offsets.ndim != 1 or offsets.size == 0 or np.any(~np.isfinite(offsets)):
        raise ValueError("voltage offsets must be a non-empty finite sequence")
    # nS * mV = pA.
    somatic_leak_nS = cell.soma.conductance_nS("leak")
    return tuple(float(somatic_leak_nS * offset) for offset in offsets)


def run_current_step_protocol(
    *,
    population_factory: PopulationFactory,
    population_params: dict[str, Any],
    currents_pA: Sequence[float],
    protocol: CurrentStepProtocol | None = None,
    brian=None,
) -> StepPhenotype:
    """Measure one cell class without exposing any SMART network outcomes."""

    if brian is None:
        import brian2 as brian
    protocol = protocol or CurrentStepProtocol()

    currents = np.asarray(currents_pA, dtype=float)
    if currents.ndim != 1 or currents.size == 0:
        raise ValueError("currents_pA must be a non-empty one-dimensional sequence")
    if np.any(~np.isfinite(currents)):
        raise ValueError("currents_pA must be finite")

    brian.start_scope()
    brian.defaultclock.dt = protocol.dt_ms * brian.ms
    population = population_factory(
        name="isolated_cell_match",
        size=int(currents.size),
        params=population_params,
        brian=brian,
    )
    spikes = brian.SpikeMonitor(population.group)
    voltage = brian.StateMonitor(
        population.group,
        "v_soma",
        record=True,
        dt=max(protocol.dt_ms, 0.1) * brian.ms,
    )
    network = brian.Network(population.group, spikes, voltage)
    network.run(protocol.pre_ms * brian.ms)
    resting_voltage_mV = float(np.mean(np.asarray(population.group.v_soma / brian.mV)))
    population.group.i_drive_soma = currents * brian.pA
    network.run(protocol.step_ms * brian.ms)
    population.group.i_drive_soma = 0 * brian.pA
    if protocol.post_ms:
        network.run(protocol.post_ms * brian.ms)

    relative_times = np.asarray(spikes.t / brian.ms, dtype=float) - protocol.pre_ms
    spike_indices = np.asarray(spikes.i, dtype=int)
    counts: list[int] = []
    rates: list[float] = []
    latencies: list[float | None] = []
    adaptation: list[float | None] = []
    for index in range(currents.size):
        times = relative_times[
            (spike_indices == index)
            & (relative_times >= 0)
            & (relative_times < protocol.step_ms)
        ]
        counts.append(int(times.size))
        rates.append(float(times.size * 1000.0 / protocol.step_ms))
        latencies.append(float(times[0]) if times.size else None)
        if times.size >= 3:
            intervals = np.diff(times)
            adaptation.append(float(intervals[-1] / intervals[0]))
        else:
            adaptation.append(None)
    finite = bool(
        np.isfinite(resting_voltage_mV)
        and np.all(np.isfinite(np.asarray(voltage.v_soma / brian.mV, dtype=float)))
    )
    return StepPhenotype(
        currents_pA=tuple(float(value) for value in currents),
        resting_voltage_mV=resting_voltage_mV,
        spike_counts=tuple(counts),
        firing_rates_hz=tuple(rates),
        first_spike_latencies_ms=tuple(latencies),
        adaptation_ratios=tuple(adaptation),
        finite=finite,
    )


def subset_phenotype(
    phenotype: StepPhenotype,
    indices: Sequence[int],
) -> StepPhenotype:
    """Split a vectorized current assay without rerunning the cell model."""

    selected = tuple(int(index) for index in indices)
    if not selected or any(index < 0 or index >= len(phenotype.currents_pA) for index in selected):
        raise ValueError("phenotype subset indices are invalid")
    return StepPhenotype(
        currents_pA=tuple(phenotype.currents_pA[index] for index in selected),
        resting_voltage_mV=phenotype.resting_voltage_mV,
        spike_counts=tuple(phenotype.spike_counts[index] for index in selected),
        firing_rates_hz=tuple(phenotype.firing_rates_hz[index] for index in selected),
        first_spike_latencies_ms=tuple(
            phenotype.first_spike_latencies_ms[index] for index in selected
        ),
        adaptation_ratios=tuple(phenotype.adaptation_ratios[index] for index in selected),
        finite=phenotype.finite,
    )


def select_interleaved_current_levels(
    scan: StepPhenotype,
    *,
    level_count: int = 7,
) -> tuple[int, ...]:
    """Select rate-spanning classic-cell levels before any AdEx fit is seen."""

    if level_count < 3 or level_count % 2 == 0:
        raise ValueError("level_count must be an odd integer of at least three")
    if len(scan.currents_pA) < level_count:
        raise ValueError("scan has fewer current levels than requested")
    currents = np.asarray(scan.currents_pA)
    rates = np.asarray(scan.firing_rates_hz)
    if np.any(np.diff(currents) <= 0):
        raise ValueError("scan currents must be strictly increasing")
    unique_rates = np.unique(rates)
    if unique_rates.size >= level_count:
        targets = np.linspace(float(rates.min()), float(rates.max()), level_count)
        selected = [int(np.argmin(np.abs(rates - target))) for target in targets]
    else:
        selected = [
            round(index)
            for index in np.linspace(0, len(scan.currents_pA) - 1, level_count)
        ]
    # Non-monotonic burst curves can map two target rates to the same current.
    # Fill deterministically with maximally separated unused scan indices.
    selected = list(dict.fromkeys(selected))
    while len(selected) < level_count:
        unused = [index for index in range(len(currents)) if index not in selected]
        next_index = max(
            unused,
            key=lambda index: (
                min(abs(index - chosen) for chosen in selected),
                -index,
            ),
        )
        selected.append(next_index)
    return tuple(sorted(selected[:level_count]))


def run_adex_candidate_batch(
    *,
    candidates: Sequence[AdExParameters],
    population_params: dict[str, Any],
    currents_pA: Sequence[float],
    protocol: CurrentStepProtocol,
    brian=None,
) -> tuple[StepPhenotype, ...]:
    """Evaluate every candidate and current in one vectorized Brian2 group."""

    if brian is None:
        import brian2 as brian
    if not candidates:
        raise ValueError("at least one AdEx candidate is required")
    if any(candidate.refractory_ms != candidates[0].refractory_ms for candidate in candidates):
        raise ValueError("batched candidates must share the fixed refractory period")
    currents = np.asarray(currents_pA, dtype=float)
    if currents.ndim != 1 or currents.size == 0 or np.any(~np.isfinite(currents)):
        raise ValueError("currents_pA must be a non-empty finite sequence")

    brian.start_scope()
    brian.defaultclock.dt = protocol.dt_ms * brian.ms
    current_count = int(currents.size)
    population = make_somatic_adex_factory(candidates[0])(
        name="isolated_adex_candidate_batch",
        size=len(candidates) * current_count,
        params=population_params,
        brian=brian,
    )
    group = population.group
    repeat = lambda values: np.repeat(np.asarray(values, dtype=float), current_count)
    cell = population.cell_spec
    group.a_adex = repeat([item.subthreshold_adaptation_nS for item in candidates]) * brian.nsiemens
    group.b_adex = repeat([item.spike_adaptation_pA for item in candidates]) * brian.pA
    group.tau_w_adex = repeat(
        [item.adaptation_time_constant_ms for item in candidates]
    ) * brian.ms
    group.delta_t_adex = repeat([item.slope_factor_mV for item in candidates]) * brian.mV
    group.e_l_adex = repeat(
        [cell.soma.e_leak_mV + item.effective_leak_offset_mV for item in candidates]
    ) * brian.mV
    group.c_scale_adex = repeat(
        [item.somatic_capacitance_scale for item in candidates]
    )
    group.g_l_scale_adex = repeat(
        [item.somatic_leak_conductance_scale for item in candidates]
    )
    group.v_t_adex = repeat(
        [
            cell.soma.e_leak_mV
            + item.effective_leak_offset_mV
            + item.threshold_offset_mV
            for item in candidates
        ]
    ) * brian.mV
    group.v_reset_adex = repeat(
        [
            cell.soma.e_leak_mV
            + item.effective_leak_offset_mV
            + item.reset_offset_mV
            for item in candidates
        ]
    ) * brian.mV
    group.v_peak_adex = repeat([item.peak_mV for item in candidates]) * brian.mV

    spikes = brian.SpikeMonitor(group)
    voltage = brian.StateMonitor(
        group,
        "v_soma",
        record=True,
        dt=max(protocol.dt_ms, 0.1) * brian.ms,
    )
    network = brian.Network(group, spikes, voltage)
    network.run(protocol.pre_ms * brian.ms)
    resting = np.asarray(group.v_soma / brian.mV, dtype=float).reshape(
        len(candidates), current_count
    )
    group.i_drive_soma = np.tile(currents, len(candidates)) * brian.pA
    network.run(protocol.step_ms * brian.ms)
    group.i_drive_soma = 0 * brian.pA
    if protocol.post_ms:
        network.run(protocol.post_ms * brian.ms)

    relative_times = np.asarray(spikes.t / brian.ms, dtype=float) - protocol.pre_ms
    spike_indices = np.asarray(spikes.i, dtype=int)
    voltage_values = np.asarray(voltage.v_soma / brian.mV, dtype=float)
    phenotypes = []
    for candidate_index in range(len(candidates)):
        counts = []
        rates = []
        latencies = []
        adaptation = []
        start = candidate_index * current_count
        for local_index in range(current_count):
            neuron_index = start + local_index
            times = relative_times[
                (spike_indices == neuron_index)
                & (relative_times >= 0)
                & (relative_times < protocol.step_ms)
            ]
            counts.append(int(times.size))
            rates.append(float(times.size * 1000.0 / protocol.step_ms))
            latencies.append(float(times[0]) if times.size else None)
            adaptation.append(
                float(np.diff(times)[-1] / np.diff(times)[0]) if times.size >= 3 else None
            )
        finite = bool(
            np.all(np.isfinite(resting[candidate_index]))
            and np.all(np.isfinite(voltage_values[start : start + current_count]))
        )
        phenotypes.append(
            StepPhenotype(
                currents_pA=tuple(float(value) for value in currents),
                resting_voltage_mV=float(np.mean(resting[candidate_index])),
                spike_counts=tuple(counts),
                firing_rates_hz=tuple(rates),
                first_spike_latencies_ms=tuple(latencies),
                adaptation_ratios=tuple(adaptation),
                finite=finite,
            )
        )
    return tuple(phenotypes)


def run_rebound_protocol(
    *,
    population_factory: PopulationFactory,
    population_params: dict[str, Any],
    protocol: ReboundProtocol | None = None,
    brian=None,
) -> ReboundPhenotype:
    """Measure release from hyperpolarization with SMART T-current retained."""

    if brian is None:
        import brian2 as brian
    protocol = protocol or ReboundProtocol()
    brian.start_scope()
    brian.defaultclock.dt = protocol.dt_ms * brian.ms
    population = population_factory(
        name="isolated_rebound",
        size=1,
        params=population_params,
        brian=brian,
    )
    spikes = brian.SpikeMonitor(population.group)
    voltage = brian.StateMonitor(population.group, "v_soma", record=True, dt=0.1 * brian.ms)
    network = brian.Network(population.group, spikes, voltage)
    network.run(protocol.baseline_ms * brian.ms)
    current = passive_normalized_currents_pA(
        population_params, (protocol.hyperpolarizing_passive_offset_mV,)
    )[0]
    population.group.i_drive_soma = current * brian.pA
    network.run(protocol.hyperpolarization_ms * brian.ms)
    population.group.i_drive_soma = 0 * brian.pA
    network.run(protocol.release_ms * brian.ms)

    times = np.asarray(spikes.t / brian.ms, dtype=float)
    release_start = protocol.baseline_ms + protocol.hyperpolarization_ms
    baseline = times[times < protocol.baseline_ms]
    hyperpolarized = times[
        (times >= protocol.baseline_ms) & (times < release_start)
    ]
    release = times[(times >= release_start) & (times < release_start + protocol.release_ms)]
    finite = bool(np.all(np.isfinite(np.asarray(voltage.v_soma / brian.mV, dtype=float))))
    return ReboundPhenotype(
        baseline_spike_count=int(baseline.size),
        hyperpolarization_spike_count=int(hyperpolarized.size),
        release_spike_count=int(release.size),
        baseline_rate_hz=float(baseline.size * 1000.0 / protocol.baseline_ms),
        release_rate_hz=float(release.size * 1000.0 / protocol.release_ms),
        first_release_spike_latency_ms=(
            None if release.size == 0 else float(release[0] - release_start)
        ),
        finite=finite,
    )


def external_input_phenotype_loss(
    target: ExternalInputPhenotype,
    candidate: ExternalInputPhenotype,
    *,
    include_model_specific_peak: bool = True,
) -> float:
    """Score transfer through one unchanged SMART external-input port."""

    if target.source_values != candidate.source_values or not candidate.finite:
        return float("inf")
    target_rates = np.asarray(target.firing_rates_hz, dtype=float)
    candidate_rates = np.asarray(candidate.firing_rates_hz, dtype=float)
    rate_scale = max(20.0, float(np.ptp(target_rates)), float(np.max(target_rates)))
    rate_loss = float(np.mean(((candidate_rates - target_rates) / rate_scale) ** 2))
    peak_loss = 0.0
    if include_model_specific_peak:
        peak_loss = float(
            np.mean(
                (
                    (
                        np.asarray(candidate.peak_voltages_mV)
                        - np.asarray(target.peak_voltages_mV)
                    )
                    / 20.0
                )
                ** 2
            )
        )
    mean_loss = float(
        np.mean(
            (
                (
                    np.asarray(candidate.mean_voltages_mV)
                    - np.asarray(target.mean_voltages_mV)
                )
                / 10.0
            )
            ** 2
        )
    )
    latency_loss = 0.0
    for expected, observed in zip(
        target.first_spike_latencies_ms,
        candidate.first_spike_latencies_ms,
        strict=True,
    ):
        if expected is None and observed is None:
            continue
        if expected is None or observed is None:
            latency_loss += 1.0
        else:
            latency_loss += ((observed - expected) / 20.0) ** 2
    latency_loss /= len(target.source_values)
    rest_loss = ((candidate.resting_voltage_mV - target.resting_voltage_mV) / 10.0) ** 2
    return float(rate_loss + peak_loss + mean_loss + latency_loss + rest_loss)


def subset_external_input_phenotype(
    phenotype: ExternalInputPhenotype,
    indices: Sequence[int],
) -> ExternalInputPhenotype:
    selected = tuple(int(index) for index in indices)
    if not selected or any(
        index < 0 or index >= len(phenotype.source_values) for index in selected
    ):
        raise ValueError("external-input phenotype subset indices are invalid")
    return ExternalInputPhenotype(
        source_values=tuple(phenotype.source_values[index] for index in selected),
        resting_voltage_mV=phenotype.resting_voltage_mV,
        spike_counts=tuple(phenotype.spike_counts[index] for index in selected),
        firing_rates_hz=tuple(phenotype.firing_rates_hz[index] for index in selected),
        first_spike_latencies_ms=tuple(
            phenotype.first_spike_latencies_ms[index] for index in selected
        ),
        peak_voltages_mV=tuple(phenotype.peak_voltages_mV[index] for index in selected),
        mean_voltages_mV=tuple(phenotype.mean_voltages_mV[index] for index in selected),
        finite=phenotype.finite,
    )


def _external_input_phenotypes(
    *,
    population: SmartPopulationAdapter,
    candidate_count: int,
    source_values: np.ndarray,
    record_id: str,
    channel: str,
    protocol: ExternalInputProtocol,
    brian,
) -> tuple[ExternalInputPhenotype, ...]:
    group = population.group
    level_count = int(source_values.size)
    spikes = brian.SpikeMonitor(group)
    voltage = brian.StateMonitor(group, "v_soma", record=True, dt=0.1 * brian.ms)
    network = brian.Network(group, spikes, voltage)
    network.run(protocol.pre_ms * brian.ms)
    resting = np.asarray(group.v_soma / brian.mV, dtype=float).reshape(
        candidate_count, level_count
    )
    for candidate_index in range(candidate_count):
        start = candidate_index * level_count
        for local_index, value in enumerate(source_values):
            population.set_external_input(
                record_id,
                channel,
                float(value),
                indices=[start + local_index],
            )
    network.run(protocol.step_ms * brian.ms)
    relative_times = np.asarray(spikes.t / brian.ms, dtype=float) - protocol.pre_ms
    spike_indices = np.asarray(spikes.i, dtype=int)
    voltage_values = np.asarray(voltage.v_soma / brian.mV, dtype=float)
    sample_times = np.asarray(voltage.t / brian.ms, dtype=float)
    step_mask = (sample_times >= protocol.pre_ms) & (
        sample_times < protocol.pre_ms + protocol.step_ms
    )
    phenotypes = []
    for candidate_index in range(candidate_count):
        start = candidate_index * level_count
        counts = []
        rates = []
        latencies = []
        peaks = []
        means = []
        for local_index in range(level_count):
            neuron_index = start + local_index
            times = relative_times[
                (spike_indices == neuron_index)
                & (relative_times >= 0)
                & (relative_times < protocol.step_ms)
            ]
            trace = voltage_values[neuron_index, step_mask]
            counts.append(int(times.size))
            rates.append(float(times.size * 1000.0 / protocol.step_ms))
            latencies.append(float(times[0]) if times.size else None)
            peaks.append(float(np.max(trace)))
            means.append(float(np.mean(trace)))
        finite = bool(
            np.all(np.isfinite(resting[candidate_index]))
            and np.all(np.isfinite(voltage_values[start : start + level_count]))
        )
        phenotypes.append(
            ExternalInputPhenotype(
                source_values=tuple(float(value) for value in source_values),
                resting_voltage_mV=float(np.mean(resting[candidate_index])),
                spike_counts=tuple(counts),
                firing_rates_hz=tuple(rates),
                first_spike_latencies_ms=tuple(latencies),
                peak_voltages_mV=tuple(peaks),
                mean_voltages_mV=tuple(means),
                finite=finite,
            )
        )
    if protocol.post_ms:
        network.run(protocol.post_ms * brian.ms)
    return tuple(phenotypes)


def run_external_input_protocol(
    *,
    population_factory: PopulationFactory,
    population_params: dict[str, Any],
    source_values: Sequence[float],
    record_id: str,
    channel: str,
    protocol: ExternalInputProtocol | None = None,
    brian=None,
) -> ExternalInputPhenotype:
    """Measure one SMART external-input port without the surrounding network."""

    if brian is None:
        import brian2 as brian
    protocol = protocol or ExternalInputProtocol()
    values = np.asarray(source_values, dtype=float)
    if (
        values.ndim != 1
        or values.size == 0
        or np.any(~np.isfinite(values))
        or np.any((values < 0) | (values > 255))
    ):
        raise ValueError("source values must be a non-empty vector in [0, 255]")
    brian.start_scope()
    brian.defaultclock.dt = protocol.dt_ms * brian.ms
    population = population_factory(
        name="isolated_external_input",
        size=int(values.size),
        params=population_params,
        brian=brian,
    )
    return _external_input_phenotypes(
        population=population,
        candidate_count=1,
        source_values=values,
        record_id=record_id,
        channel=channel,
        protocol=protocol,
        brian=brian,
    )[0]


def run_adex_external_input_candidate_batch(
    *,
    candidates: Sequence[AdExParameters],
    population_params: dict[str, Any],
    source_values: Sequence[float],
    record_id: str,
    channel: str,
    protocol: ExternalInputProtocol,
    brian=None,
) -> tuple[ExternalInputPhenotype, ...]:
    """Evaluate AdEx candidates against one unchanged SMART input port."""

    if brian is None:
        import brian2 as brian
    if not candidates:
        raise ValueError("at least one AdEx candidate is required")
    if any(candidate.refractory_ms != candidates[0].refractory_ms for candidate in candidates):
        raise ValueError("batched candidates must share the fixed refractory period")
    values = np.asarray(source_values, dtype=float)
    if (
        values.ndim != 1
        or values.size == 0
        or np.any(~np.isfinite(values))
        or np.any((values < 0) | (values > 255))
    ):
        raise ValueError("source values must be a non-empty vector in [0, 255]")
    brian.start_scope()
    brian.defaultclock.dt = protocol.dt_ms * brian.ms
    level_count = int(values.size)
    population = make_somatic_adex_factory(candidates[0])(
        name="isolated_adex_external_input_batch",
        size=len(candidates) * level_count,
        params=population_params,
        brian=brian,
    )
    group = population.group
    repeat = lambda items: np.repeat(np.asarray(items, dtype=float), level_count)
    cell = population.cell_spec
    group.a_adex = repeat(
        [item.subthreshold_adaptation_nS for item in candidates]
    ) * brian.nsiemens
    group.b_adex = repeat([item.spike_adaptation_pA for item in candidates]) * brian.pA
    group.tau_w_adex = repeat(
        [item.adaptation_time_constant_ms for item in candidates]
    ) * brian.ms
    group.delta_t_adex = repeat(
        [item.slope_factor_mV for item in candidates]
    ) * brian.mV
    group.v_peak_adex = repeat([item.peak_mV for item in candidates]) * brian.mV
    group.e_l_adex = repeat(
        [cell.soma.e_leak_mV + item.effective_leak_offset_mV for item in candidates]
    ) * brian.mV
    group.c_scale_adex = repeat(
        [item.somatic_capacitance_scale for item in candidates]
    )
    group.g_l_scale_adex = repeat(
        [item.somatic_leak_conductance_scale for item in candidates]
    )
    group.v_t_adex = repeat(
        [
            cell.soma.e_leak_mV
            + item.effective_leak_offset_mV
            + item.threshold_offset_mV
            for item in candidates
        ]
    ) * brian.mV
    group.v_reset_adex = repeat(
        [
            cell.soma.e_leak_mV
            + item.effective_leak_offset_mV
            + item.reset_offset_mV
            for item in candidates
        ]
    ) * brian.mV
    return _external_input_phenotypes(
        population=population,
        candidate_count=len(candidates),
        source_values=values,
        record_id=record_id,
        channel=channel,
        protocol=protocol,
        brian=brian,
    )


def relay_inhibition_phenotype_loss(
    target: RelayInhibitionPhenotype,
    candidate: RelayInhibitionPhenotype,
) -> float:
    """Score relay transfer under matched external and inhibitory contexts."""

    if (
        target.source_values != candidate.source_values
        or target.inhibition_scales != candidate.inhibition_scales
        or not candidate.finite
    ):
        return float("inf")
    target_rates = np.asarray(target.firing_rates_hz, dtype=float)
    candidate_rates = np.asarray(candidate.firing_rates_hz, dtype=float)
    rate_scale = max(20.0, float(np.ptp(target_rates)), float(np.max(target_rates)))
    rate_loss = float(np.mean(((candidate_rates - target_rates) / rate_scale) ** 2))
    mean_loss = float(
        np.mean(
            (
                (
                    np.asarray(candidate.mean_voltages_mV)
                    - np.asarray(target.mean_voltages_mV)
                )
                / 10.0
            )
            ** 2
        )
    )
    latency_loss = 0.0
    for expected, observed in zip(
        target.first_spike_latencies_ms,
        candidate.first_spike_latencies_ms,
        strict=True,
    ):
        if expected is None and observed is None:
            continue
        if expected is None or observed is None:
            latency_loss += 1.0
        else:
            latency_loss += ((observed - expected) / 20.0) ** 2
    latency_loss /= len(target.source_values)
    rest_loss = ((candidate.resting_voltage_mV - target.resting_voltage_mV) / 10.0) ** 2
    return float(rate_loss + mean_loss + latency_loss + rest_loss)


def subset_relay_inhibition_phenotype(
    phenotype: RelayInhibitionPhenotype,
    indices: Sequence[int],
) -> RelayInhibitionPhenotype:
    selected = tuple(int(index) for index in indices)
    if not selected or any(
        index < 0 or index >= len(phenotype.source_values) for index in selected
    ):
        raise ValueError("relay-inhibition phenotype subset indices are invalid")
    return RelayInhibitionPhenotype(
        source_values=tuple(phenotype.source_values[index] for index in selected),
        inhibition_scales=tuple(
            phenotype.inhibition_scales[index] for index in selected
        ),
        resting_voltage_mV=phenotype.resting_voltage_mV,
        spike_counts=tuple(phenotype.spike_counts[index] for index in selected),
        firing_rates_hz=tuple(phenotype.firing_rates_hz[index] for index in selected),
        first_spike_latencies_ms=tuple(
            phenotype.first_spike_latencies_ms[index] for index in selected
        ),
        mean_voltages_mV=tuple(phenotype.mean_voltages_mV[index] for index in selected),
        finite=phenotype.finite,
    )


def _relay_inhibition_phenotypes(
    *,
    population: SmartPopulationAdapter,
    candidate_count: int,
    source_values: np.ndarray,
    inhibition_scales: np.ndarray,
    external_record_id: str,
    channel: str,
    inhibitory_gate_baselines: Mapping[str, float],
    protocol: RelayInhibitionProtocol,
    brian,
) -> tuple[RelayInhibitionPhenotype, ...]:
    group = population.group
    condition_count = int(source_values.size)
    spikes = brian.SpikeMonitor(group)
    voltage = brian.StateMonitor(group, "v_soma", record=True, dt=0.1 * brian.ms)
    network = brian.Network(group, spikes, voltage)
    network.run(protocol.pre_ms * brian.ms)
    resting = np.asarray(group.v_soma / brian.mV, dtype=float).reshape(
        candidate_count, condition_count
    )
    ports = {port.record_id: port for port in population.compiled.synaptic_ports}
    for record_id in inhibitory_gate_baselines:
        if record_id not in ports:
            raise ValueError(f"relay has no inhibitory port {record_id}")
    for candidate_index in range(candidate_count):
        start = candidate_index * condition_count
        for local_index, (source, scale) in enumerate(
            zip(source_values, inhibition_scales, strict=True)
        ):
            neuron_index = start + local_index
            population.set_external_input(
                external_record_id,
                channel,
                float(source),
                indices=[neuron_index],
            )
            for record_id, baseline in inhibitory_gate_baselines.items():
                port = ports[record_id]
                getattr(group, f"{port.name}_gate")[neuron_index] = baseline * scale
    network.run(protocol.step_ms * brian.ms)
    relative_times = np.asarray(spikes.t / brian.ms, dtype=float) - protocol.pre_ms
    spike_indices = np.asarray(spikes.i, dtype=int)
    voltage_values = np.asarray(voltage.v_soma / brian.mV, dtype=float)
    sample_times = np.asarray(voltage.t / brian.ms, dtype=float)
    step_mask = (sample_times >= protocol.pre_ms) & (
        sample_times < protocol.pre_ms + protocol.step_ms
    )
    phenotypes = []
    for candidate_index in range(candidate_count):
        start = candidate_index * condition_count
        counts = []
        rates = []
        latencies = []
        means = []
        for local_index in range(condition_count):
            neuron_index = start + local_index
            times = relative_times[
                (spike_indices == neuron_index)
                & (relative_times >= 0)
                & (relative_times < protocol.step_ms)
            ]
            trace = voltage_values[neuron_index, step_mask]
            counts.append(int(times.size))
            rates.append(float(times.size * 1000.0 / protocol.step_ms))
            latencies.append(float(times[0]) if times.size else None)
            means.append(float(np.mean(trace)))
        finite = bool(
            np.all(np.isfinite(resting[candidate_index]))
            and np.all(np.isfinite(voltage_values[start : start + condition_count]))
        )
        phenotypes.append(
            RelayInhibitionPhenotype(
                source_values=tuple(float(value) for value in source_values),
                inhibition_scales=tuple(float(value) for value in inhibition_scales),
                resting_voltage_mV=float(np.mean(resting[candidate_index])),
                spike_counts=tuple(counts),
                firing_rates_hz=tuple(rates),
                first_spike_latencies_ms=tuple(latencies),
                mean_voltages_mV=tuple(means),
                finite=finite,
            )
        )
    if protocol.post_ms:
        network.run(protocol.post_ms * brian.ms)
    return tuple(phenotypes)


def run_relay_inhibition_protocol(
    *,
    population_factory: PopulationFactory,
    population_params: dict[str, Any],
    source_values: Sequence[float],
    inhibition_scales: Sequence[float],
    external_record_id: str,
    channel: str,
    inhibitory_gate_baselines: Mapping[str, float],
    protocol: RelayInhibitionProtocol,
    brian=None,
) -> RelayInhibitionPhenotype:
    if brian is None:
        import brian2 as brian
    sources = np.asarray(source_values, dtype=float)
    scales = np.asarray(inhibition_scales, dtype=float)
    if sources.shape != scales.shape or sources.ndim != 1 or sources.size == 0:
        raise ValueError("relay context vectors must be aligned and non-empty")
    brian.start_scope()
    brian.defaultclock.dt = protocol.dt_ms * brian.ms
    population = population_factory(
        name="isolated_relay_inhibition",
        size=int(sources.size),
        params=population_params,
        brian=brian,
    )
    return _relay_inhibition_phenotypes(
        population=population,
        candidate_count=1,
        source_values=sources,
        inhibition_scales=scales,
        external_record_id=external_record_id,
        channel=channel,
        inhibitory_gate_baselines=inhibitory_gate_baselines,
        protocol=protocol,
        brian=brian,
    )[0]


def run_adex_relay_inhibition_candidate_batch(
    *,
    candidates: Sequence[AdExParameters],
    population_params: dict[str, Any],
    source_values: Sequence[float],
    inhibition_scales: Sequence[float],
    external_record_id: str,
    channel: str,
    inhibitory_gate_baselines: Mapping[str, float],
    protocol: RelayInhibitionProtocol,
    brian=None,
) -> tuple[RelayInhibitionPhenotype, ...]:
    if brian is None:
        import brian2 as brian
    if not candidates:
        raise ValueError("at least one AdEx candidate is required")
    sources = np.asarray(source_values, dtype=float)
    scales = np.asarray(inhibition_scales, dtype=float)
    if sources.shape != scales.shape or sources.ndim != 1 or sources.size == 0:
        raise ValueError("relay context vectors must be aligned and non-empty")
    brian.start_scope()
    brian.defaultclock.dt = protocol.dt_ms * brian.ms
    condition_count = int(sources.size)
    population = make_somatic_adex_factory(candidates[0])(
        name="isolated_adex_relay_inhibition_batch",
        size=len(candidates) * condition_count,
        params=population_params,
        brian=brian,
    )
    group = population.group
    repeat = lambda items: np.repeat(np.asarray(items, dtype=float), condition_count)
    cell = population.cell_spec
    group.a_adex = repeat(
        [item.subthreshold_adaptation_nS for item in candidates]
    ) * brian.nsiemens
    group.b_adex = repeat([item.spike_adaptation_pA for item in candidates]) * brian.pA
    group.tau_w_adex = repeat(
        [item.adaptation_time_constant_ms for item in candidates]
    ) * brian.ms
    group.delta_t_adex = repeat(
        [item.slope_factor_mV for item in candidates]
    ) * brian.mV
    group.e_l_adex = repeat(
        [cell.soma.e_leak_mV + item.effective_leak_offset_mV for item in candidates]
    ) * brian.mV
    group.c_scale_adex = repeat(
        [item.somatic_capacitance_scale for item in candidates]
    )
    group.g_l_scale_adex = repeat(
        [item.somatic_leak_conductance_scale for item in candidates]
    )
    group.v_t_adex = repeat(
        [
            cell.soma.e_leak_mV
            + item.effective_leak_offset_mV
            + item.threshold_offset_mV
            for item in candidates
        ]
    ) * brian.mV
    group.v_reset_adex = repeat(
        [
            cell.soma.e_leak_mV
            + item.effective_leak_offset_mV
            + item.reset_offset_mV
            for item in candidates
        ]
    ) * brian.mV
    group.v_peak_adex = repeat([item.peak_mV for item in candidates]) * brian.mV
    return _relay_inhibition_phenotypes(
        population=population,
        candidate_count=len(candidates),
        source_values=sources,
        inhibition_scales=scales,
        external_record_id=external_record_id,
        channel=channel,
        inhibitory_gate_baselines=inhibitory_gate_baselines,
        protocol=protocol,
        brian=brian,
    )


def assess_rebound_match(
    classic: ReboundPhenotype,
    alternative: ReboundPhenotype,
) -> ReboundAssessment:
    """Apply the preregistered qualitative and bounded quantitative gates."""

    rate_tolerance = max(20.0, 0.5 * classic.release_rate_hz)
    if classic.first_release_spike_latency_ms is None:
        latency_matches = alternative.first_release_spike_latency_ms is None
    elif alternative.first_release_spike_latency_ms is None:
        latency_matches = False
    else:
        latency_matches = (
            abs(
                alternative.first_release_spike_latency_ms
                - classic.first_release_spike_latency_ms
            )
            <= 20.0
        )
    return ReboundAssessment(
        finite=classic.finite and alternative.finite,
        rebound_presence_matches=(
            classic.rebound_present == alternative.rebound_present
        ),
        release_rate_matches=(
            abs(alternative.release_rate_hz - classic.release_rate_hz)
            <= rate_tolerance
        ),
        first_latency_matches=latency_matches,
    )


def phenotype_loss(target: StepPhenotype, candidate: StepPhenotype) -> float:
    """Predeclared scale-normalized loss using isolated-cell outputs only."""

    if target.currents_pA != candidate.currents_pA or not candidate.finite:
        return float("inf")
    rate_scale = max(10.0, max(target.firing_rates_hz, default=0.0))
    rate_loss = np.mean(
        ((np.asarray(candidate.firing_rates_hz) - np.asarray(target.firing_rates_hz)) / rate_scale)
        ** 2
    )
    rest_loss = ((candidate.resting_voltage_mV - target.resting_voltage_mV) / 5.0) ** 2
    latency_losses = []
    adaptation_losses = []
    for target_latency, candidate_latency in zip(
        target.first_spike_latencies_ms,
        candidate.first_spike_latencies_ms,
        strict=True,
    ):
        if target_latency is None or candidate_latency is None:
            latency_losses.append(float(target_latency is not candidate_latency))
        else:
            latency_losses.append(((candidate_latency - target_latency) / 50.0) ** 2)
    for target_ratio, candidate_ratio in zip(
        target.adaptation_ratios,
        candidate.adaptation_ratios,
        strict=True,
    ):
        if target_ratio is None or candidate_ratio is None:
            adaptation_losses.append(float(target_ratio is not candidate_ratio))
        else:
            adaptation_losses.append((candidate_ratio - target_ratio) ** 2)
    return float(
        rest_loss
        + rate_loss
        + np.mean(latency_losses)
        + np.mean(adaptation_losses)
    )


def evaluate_adex_candidate(
    *,
    parameters: AdExParameters,
    population_params: dict[str, Any],
    training_currents_pA: Sequence[float],
    holdout_currents_pA: Sequence[float],
    training_target: StepPhenotype,
    holdout_target: StepPhenotype,
    protocol: CurrentStepProtocol,
    brian=None,
) -> CandidateFit:
    """Score one preregistered candidate without constructing a SMART network."""

    factory = make_somatic_adex_factory(parameters)
    training = run_current_step_protocol(
        population_factory=factory,
        population_params=population_params,
        currents_pA=training_currents_pA,
        protocol=protocol,
        brian=brian,
    )
    holdout = run_current_step_protocol(
        population_factory=factory,
        population_params=population_params,
        currents_pA=holdout_currents_pA,
        protocol=protocol,
        brian=brian,
    )
    return CandidateFit(
        parameters=parameters,
        training_loss=phenotype_loss(training_target, training),
        holdout_loss=phenotype_loss(holdout_target, holdout),
        training_phenotype=training,
        holdout_phenotype=holdout,
    )


def select_adex_candidate(candidates: Sequence[CandidateFit]) -> CandidateFit:
    """Select on training loss only; holdout remains reporting-only."""

    if not candidates:
        raise ValueError("at least one candidate is required")
    return min(
        candidates,
        key=lambda candidate: (
            candidate.training_loss,
            tuple(candidate.parameters.as_dict().values()),
        ),
    )
