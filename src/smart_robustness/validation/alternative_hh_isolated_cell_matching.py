"""Network-blind isolated fitting for the Pospischil-type somatic HH arm."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import exp
from typing import Any

import numpy as np

from ..models.alternative_hh import make_somatic_alternative_hh_factory
from ..models.alternative_hh_parameters import (
    POSPISCHIL_FS_MEAN,
    POSPISCHIL_RS_MEAN,
    AlternativeHHParameters,
)
from ..models.currents import NaKRateConvention, traub_miles_rates
from .isolated_cell_matching import (
    CurrentStepProtocol,
    StepPhenotype,
    run_current_step_protocol,
)

PARAMETER_NAMES = (
    "threshold_mV",
    "sodium_density_mS_cm2",
    "potassium_density_mS_cm2",
    "m_current_density_mS_cm2",
    "m_current_tau_max_ms",
)


@dataclass(frozen=True, slots=True)
class AlternativeHHFit:
    parameters: AlternativeHHParameters
    training_loss: float
    training_phenotype: StepPhenotype


@dataclass(frozen=True, slots=True)
class AlternativeHHMatchAssessment:
    finite: bool
    resting_voltage_pass: bool
    silent_levels_pass: bool
    spike_counts_pass: bool
    first_spike_latencies_pass: bool
    adaptation_ratios_pass: bool

    @property
    def promoted(self) -> bool:
        return bool(
            self.finite
            and self.resting_voltage_pass
            and self.silent_levels_pass
            and self.spike_counts_pass
            and self.first_spike_latencies_pass
            and self.adaptation_ratios_pass
        )


def generate_alternative_hh_sobol_candidates(
    bounds: Mapping[str, Sequence[float]],
    *,
    count: int = 256,
    literature_seeds: Sequence[AlternativeHHParameters] = (
        POSPISCHIL_RS_MEAN,
        POSPISCHIL_FS_MEAN,
    ),
) -> tuple[AlternativeHHParameters, ...]:
    """Generate the preregistered deterministic parameter set."""

    from scipy.stats import qmc

    if set(bounds) != set(PARAMETER_NAMES):
        raise ValueError(f"alternative-HH bounds must be exactly {PARAMETER_NAMES}")
    if count <= 0 or count & (count - 1):
        raise ValueError("candidate count must be a positive power of two")
    lower = np.asarray([float(bounds[name][0]) for name in PARAMETER_NAMES])
    upper = np.asarray([float(bounds[name][1]) for name in PARAMETER_NAMES])
    if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)) or np.any(lower >= upper):
        raise ValueError("each alternative-HH bound must be finite and increasing")
    samples = qmc.Sobol(d=len(PARAMETER_NAMES), scramble=False).random_base2(
        m=int(np.log2(count))
    )
    scaled = qmc.scale(samples, lower, upper)
    candidates = [
        AlternativeHHParameters(
            **{
                name: float(value)
                for name, value in zip(
                    PARAMETER_NAMES,
                    scaled[index],
                    strict=True,
                )
            }
        )
        for index in range(count)
    ]
    for seed in literature_seeds:
        if seed not in candidates:
            candidates.append(seed)
    return tuple(candidates)


def _phenotypes_from_group(
    *,
    candidates: Sequence[AlternativeHHParameters],
    currents: np.ndarray,
    protocol: CurrentStepProtocol,
    spikes,
    voltage,
    resting: np.ndarray,
    brian,
) -> tuple[StepPhenotype, ...]:
    current_count = int(currents.size)
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
                float(np.diff(times)[-1] / np.diff(times)[0])
                if times.size >= 3
                else None
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


def run_alternative_hh_candidate_batch(
    *,
    candidates: Sequence[AlternativeHHParameters],
    population_params: dict[str, Any],
    currents_pA: Sequence[float],
    protocol: CurrentStepProtocol,
    brian=None,
) -> tuple[StepPhenotype, ...]:
    """Evaluate every candidate and training level in one Brian2 group."""

    if brian is None:
        import brian2 as brian
    if not candidates:
        raise ValueError("at least one alternative-HH candidate is required")
    currents = np.asarray(currents_pA, dtype=float)
    if currents.ndim != 1 or currents.size == 0 or np.any(~np.isfinite(currents)):
        raise ValueError("currents_pA must be a non-empty finite sequence")

    brian.start_scope()
    brian.defaultclock.dt = protocol.dt_ms * brian.ms
    current_count = int(currents.size)
    population = make_somatic_alternative_hh_factory(candidates[0])(
        name="isolated_alternative_hh_candidate_batch",
        size=len(candidates) * current_count,
        params=population_params,
        brian=brian,
    )
    group = population.group
    repeat = lambda values: np.repeat(np.asarray(values, dtype=float), current_count)
    area_cm2 = population.cell_spec.soma.lateral_area_cm2
    group.v_t_pospischil = repeat([item.threshold_mV for item in candidates]) * brian.mV
    group.g_na_soma = (
        repeat([item.sodium_density_mS_cm2 for item in candidates])
        * area_cm2
        * 1e6
        * brian.nsiemens
    )
    group.g_k_soma = (
        repeat([item.potassium_density_mS_cm2 for item in candidates])
        * area_cm2
        * 1e6
        * brian.nsiemens
    )
    group.g_m_soma = (
        repeat([item.m_current_density_mS_cm2 for item in candidates])
        * area_cm2
        * 1e6
        * brian.nsiemens
    )
    group.tau_max_m_pospischil = repeat(
        [item.m_current_tau_max_ms for item in candidates]
    ) * brian.ms

    initial_voltage_mV = np.asarray(group.v_soma / brian.mV, dtype=float)
    thresholds_mV = np.asarray(group.v_t_pospischil / brian.mV, dtype=float)
    rates = [
        traub_miles_rates(
            voltage - threshold,
            NaKRateConvention.STANDARD_TRAUB_MILES,
        )
        for voltage, threshold in zip(initial_voltage_mV, thresholds_mV, strict=True)
    ]
    group.m_soma = [item.alpha_m / (item.alpha_m + item.beta_m) for item in rates]
    group.h_soma = [item.alpha_h / (item.alpha_h + item.beta_h) for item in rates]
    group.n_soma = [item.alpha_n / (item.alpha_n + item.beta_n) for item in rates]
    group.p_m_soma = [1.0 / (1.0 + exp(-(value + 35.0) / 10.0)) for value in initial_voltage_mV]

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
    return _phenotypes_from_group(
        candidates=candidates,
        currents=currents,
        protocol=protocol,
        spikes=spikes,
        voltage=voltage,
        resting=resting,
        brian=brian,
    )


def alternative_hh_phenotype_loss(
    target: StepPhenotype,
    candidate: StepPhenotype,
    *,
    resting_scale_mV: float = 2.0,
    firing_rate_scale_hz: float = 5.0,
    latency_scale_ms: float = 10.0,
    adaptation_scale: float = 0.25,
    missing_penalty: float = 1000.0,
) -> float:
    """Registered scale-normalized training loss."""

    if target.currents_pA != candidate.currents_pA or not candidate.finite:
        return float("inf")
    loss = ((candidate.resting_voltage_mV - target.resting_voltage_mV) / resting_scale_mV) ** 2
    loss += float(
        np.mean(
            (
                (np.asarray(candidate.firing_rates_hz) - np.asarray(target.firing_rates_hz))
                / firing_rate_scale_hz
            )
            ** 2
        )
    )
    for target_value, candidate_value in zip(
        target.first_spike_latencies_ms,
        candidate.first_spike_latencies_ms,
        strict=True,
    ):
        if target_value is None or candidate_value is None:
            loss += 0.0 if target_value is candidate_value else missing_penalty
        else:
            loss += ((candidate_value - target_value) / latency_scale_ms) ** 2
    for target_value, candidate_value in zip(
        target.adaptation_ratios,
        candidate.adaptation_ratios,
        strict=True,
    ):
        if target_value is None:
            continue
        if candidate_value is None:
            loss += missing_penalty
        else:
            loss += ((candidate_value - target_value) / adaptation_scale) ** 2
    return float(loss)


def select_alternative_hh_candidate(
    candidates: Sequence[AlternativeHHFit],
) -> AlternativeHHFit:
    """Select solely on training loss with a deterministic parameter tie-break."""

    if not candidates:
        raise ValueError("at least one scored candidate is required")
    return min(
        candidates,
        key=lambda candidate: (
            candidate.training_loss,
            tuple(candidate.parameters.as_dict()[name] for name in PARAMETER_NAMES),
        ),
    )


def assess_alternative_hh_match(
    target: StepPhenotype,
    candidate: StepPhenotype,
    *,
    resting_voltage_error_mV_max: float = 2.0,
    spike_count_error_max: int = 1,
    latency_error_ms_max: float = 10.0,
    adaptation_error_max: float = 0.25,
    adaptation_minimum_target_spikes: int = 3,
) -> AlternativeHHMatchAssessment:
    """Apply the preregistered per-level promotion gates."""

    if target.currents_pA != candidate.currents_pA:
        raise ValueError("target and candidate currents differ")
    silent_pass = all(
        candidate_count == 0
        for target_count, candidate_count in zip(
            target.spike_counts, candidate.spike_counts, strict=True
        )
        if target_count == 0
    )
    spike_count_pass = all(
        abs(candidate_count - target_count) <= spike_count_error_max
        for target_count, candidate_count in zip(
            target.spike_counts, candidate.spike_counts, strict=True
        )
    )
    latency_pass = True
    adaptation_pass = True
    for target_count, target_latency, candidate_latency, target_ratio, candidate_ratio in zip(
        target.spike_counts,
        target.first_spike_latencies_ms,
        candidate.first_spike_latencies_ms,
        target.adaptation_ratios,
        candidate.adaptation_ratios,
        strict=True,
    ):
        if target_latency is not None:
            latency_pass &= bool(
                candidate_latency is not None
                and abs(candidate_latency - target_latency) <= latency_error_ms_max
            )
        if target_count >= adaptation_minimum_target_spikes:
            adaptation_pass &= bool(
                target_ratio is not None
                and candidate_ratio is not None
                and abs(candidate_ratio - target_ratio) <= adaptation_error_max
            )
    return AlternativeHHMatchAssessment(
        finite=candidate.finite,
        resting_voltage_pass=(
            abs(candidate.resting_voltage_mV - target.resting_voltage_mV)
            <= resting_voltage_error_mV_max
        ),
        silent_levels_pass=silent_pass,
        spike_counts_pass=spike_count_pass,
        first_spike_latencies_pass=latency_pass,
        adaptation_ratios_pass=adaptation_pass,
    )


def run_selected_alternative_hh(
    *,
    parameters: AlternativeHHParameters,
    population_params: dict[str, Any],
    currents_pA: Sequence[float],
    protocol: CurrentStepProtocol,
    brian=None,
) -> StepPhenotype:
    """Open the sealed holdout only for the training-selected candidate."""

    return run_current_step_protocol(
        population_factory=make_somatic_alternative_hh_factory(parameters),
        population_params=population_params,
        currents_pA=currents_pA,
        protocol=protocol,
        brian=brian,
    )
