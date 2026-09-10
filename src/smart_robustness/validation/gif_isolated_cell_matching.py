"""Network-blind stochastic GIF fitting helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..models.gif import make_somatic_gif_factory
from ..models.gif_parameters import GIFParameters
from .isolated_cell_matching import CurrentStepProtocol, StepPhenotype

_FITTED_PARAMETER_NAMES = (
    "threshold_offset_mV",
    "reset_offset_mV",
    "eta_fast_pA",
    "eta_slow_pA",
    "gamma_fast_mV",
    "gamma_slow_mV",
)


@dataclass(frozen=True, slots=True)
class GIFCandidateFit:
    parameters: GIFParameters
    mean_training_loss: float
    mean_seed_holdout_loss: float
    training_losses: tuple[float, ...]
    seed_holdout_losses: tuple[float, ...]


def generate_gif_sobol_candidates(
    bounds: dict[str, tuple[float, float]],
    *,
    base: GIFParameters,
    count: int = 32,
    include_literature: bool = True,
) -> tuple[GIFParameters, ...]:
    """Generate the fixed six-dimensional GIF search around one preset."""

    from scipy.stats import qmc

    if set(bounds) != set(_FITTED_PARAMETER_NAMES):
        raise ValueError(f"GIF candidate bounds must be exactly {_FITTED_PARAMETER_NAMES}")
    if count <= 0 or count & (count - 1):
        raise ValueError("Sobol candidate count must be a positive power of two")
    lower = np.asarray([bounds[name][0] for name in _FITTED_PARAMETER_NAMES], dtype=float)
    upper = np.asarray([bounds[name][1] for name in _FITTED_PARAMETER_NAMES], dtype=float)
    if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)) or np.any(lower >= upper):
        raise ValueError("each GIF bound must be finite and increasing")
    samples = qmc.Sobol(d=len(_FITTED_PARAMETER_NAMES), scramble=False).random_base2(
        m=int(np.log2(count))
    )
    scaled = qmc.scale(samples, lower, upper)
    fixed = base.as_dict()
    candidates = []
    for row in scaled:
        values = fixed | {
            name: float(value)
            for name, value in zip(_FITTED_PARAMETER_NAMES, row, strict=True)
        }
        candidates.append(GIFParameters.from_mapping(values))
    if include_literature and base not in candidates:
        candidates.append(base)
    return tuple(candidates)


def _assign_batched_parameters(group, cell, candidates: Sequence[GIFParameters], repeats: int, brian):
    repeat = lambda values: np.repeat(np.asarray(values, dtype=float), repeats)
    group.eta_fast_jump_gif = repeat([item.eta_fast_pA for item in candidates]) * brian.pA
    group.eta_slow_jump_gif = repeat([item.eta_slow_pA for item in candidates]) * brian.pA
    group.tau_eta_fast_gif = repeat([item.eta_fast_tau_ms for item in candidates]) * brian.ms
    group.tau_eta_slow_gif = repeat([item.eta_slow_tau_ms for item in candidates]) * brian.ms
    group.gamma_fast_jump_gif = repeat([item.gamma_fast_mV for item in candidates]) * brian.mV
    group.gamma_slow_jump_gif = repeat([item.gamma_slow_mV for item in candidates]) * brian.mV
    group.tau_gamma_fast_gif = repeat([item.gamma_fast_tau_ms for item in candidates]) * brian.ms
    group.tau_gamma_slow_gif = repeat([item.gamma_slow_tau_ms for item in candidates]) * brian.ms
    group.lambda0_gif = repeat([item.escape_rate_hz for item in candidates]) * brian.Hz
    group.delta_v_gif = repeat([item.stochasticity_mV for item in candidates]) * brian.mV
    group.e_l_gif = repeat(
        [cell.soma.e_leak_mV + item.effective_leak_offset_mV for item in candidates]
    ) * brian.mV
    group.v_t_star_gif = repeat(
        [
            cell.soma.e_leak_mV
            + item.effective_leak_offset_mV
            + item.threshold_offset_mV
            for item in candidates
        ]
    ) * brian.mV
    group.v_reset_gif = repeat(
        [
            cell.soma.e_leak_mV
            + item.effective_leak_offset_mV
            + item.reset_offset_mV
            for item in candidates
        ]
    ) * brian.mV
    group.c_scale_gif = repeat([item.somatic_capacitance_scale for item in candidates])
    group.g_l_scale_gif = repeat(
        [item.somatic_leak_conductance_scale for item in candidates]
    )


def run_gif_candidate_batch(
    *,
    candidates: Sequence[GIFParameters],
    population_params: dict[str, Any],
    currents_pA: Sequence[float],
    protocol: CurrentStepProtocol,
    seed: int,
    brian=None,
) -> tuple[StepPhenotype, ...]:
    """Evaluate all GIF candidates under one registered random seed."""

    if brian is None:
        import brian2 as brian
    if not candidates:
        raise ValueError("at least one GIF candidate is required")
    if any(candidate.refractory_ms != candidates[0].refractory_ms for candidate in candidates):
        raise ValueError("batched candidates must share the fixed refractory period")
    currents = np.asarray(currents_pA, dtype=float)
    if currents.ndim != 1 or currents.size == 0 or np.any(~np.isfinite(currents)):
        raise ValueError("currents_pA must be a non-empty finite sequence")

    brian.start_scope()
    brian.seed(int(seed))
    brian.defaultclock.dt = protocol.dt_ms * brian.ms
    level_count = int(currents.size)
    population = make_somatic_gif_factory(candidates[0])(
        name="isolated_gif_candidate_batch",
        size=len(candidates) * level_count,
        params=population_params,
        brian=brian,
    )
    group = population.group
    _assign_batched_parameters(group, population.cell_spec, candidates, level_count, brian)

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
        len(candidates), level_count
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
        counts: list[int] = []
        rates: list[float] = []
        latencies: list[float | None] = []
        adaptation: list[float | None] = []
        start = candidate_index * level_count
        for local_index in range(level_count):
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
            and np.all(np.isfinite(voltage_values[start : start + level_count]))
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


def select_gif_candidate(candidates: Sequence[GIFCandidateFit]) -> GIFCandidateFit:
    """Select by training seeds/levels only; both holdouts remain sealed."""

    if not candidates:
        raise ValueError("at least one candidate is required")
    return min(
        candidates,
        key=lambda candidate: (
            candidate.mean_training_loss,
            tuple(candidate.parameters.as_dict().values()),
        ),
    )
