"""Isolated conductance-factorization traces; not connected to SMART.

The functions here make the temporal-summation and transmitter-resource
conventions explicit.  They return dimensionless gates; the unchanged
projection density, spatial weight, and membrane driving force are outside
this isolated comparison.
"""

from __future__ import annotations

import math

import numpy as np

from .pulvinar_matching import first_event_match
from .pulvinar_stp import STPParameters, release_history

CONTROL_NAMES = (
    "classic_union_continuous_resource",
    "additive_continuous_resource",
    "additive_emission_snapshot",
    "type2_peak_matched",
    "type2_area_matched",
)


def alpha_kernel(elapsed_ms, tau_ms: float) -> np.ndarray:
    """Unit-peak alpha kernel, causal and safe for arbitrary finite samples."""
    elapsed = np.asarray(elapsed_ms, dtype=float)
    if not np.all(np.isfinite(elapsed)) or not math.isfinite(tau_ms) or tau_ms <= 0:
        raise ValueError("elapsed and positive tau must be finite")
    ratio = np.maximum(elapsed, 0) / tau_ms
    wave = math.e * ratio * np.exp(-ratio)
    return np.where(elapsed >= 0, wave, 0.0)


def resource_history(
    times_ms, emissions_ms, *, depletion_fraction: float, recovery_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Right-continuous source resource and pre-depletion emission snapshots."""
    times, emissions = _validated_times(times_ms, emissions_ms)
    if not (math.isfinite(depletion_fraction) and 0 <= depletion_fraction <= 1):
        raise ValueError("depletion_fraction must be finite and in [0, 1]")
    if not math.isfinite(recovery_ms) or recovery_ms <= 0:
        raise ValueError("recovery_ms must be finite and positive")

    snapshots = np.empty(emissions.size)
    post = np.empty(emissions.size)
    resource, previous = 1.0, 0.0
    for index, emission in enumerate(emissions):
        resource = 1 - (1 - resource) * math.exp(-(emission - previous) / recovery_ms)
        snapshots[index] = resource
        resource *= 1 - depletion_fraction
        post[index] = resource
        previous = float(emission)

    current = np.ones(times.size)
    for index, time in enumerate(times):
        latest = np.searchsorted(emissions, time, side="right") - 1
        if latest >= 0:
            current[index] = 1 - (1 - post[latest]) * math.exp(
                -(time - emissions[latest]) / recovery_ms
            )
    return current, snapshots


def conductance_controls(
    times_ms,
    emissions_ms,
    *,
    tau_ms: float,
    delay_ms: float,
    depletion_fraction: float,
    recovery_ms: float,
    type2_parameters: STPParameters,
) -> dict[str, np.ndarray]:
    """Return all five preregistered gates on one physical time axis."""
    times, emissions = _validated_times(times_ms, emissions_ms)
    if not math.isfinite(delay_ms) or delay_ms < 0:
        raise ValueError("delay_ms must be finite and nonnegative")
    current_resource, snapshots = resource_history(
        times,
        emissions,
        depletion_fraction=depletion_fraction,
        recovery_ms=recovery_ms,
    )
    arrivals = emissions + delay_ms
    additive = np.zeros(times.size)
    snapshot_additive = np.zeros(times.size)
    dynamic_additive = np.zeros(times.size)
    releases = release_history(emissions / 1000, type2_parameters)["released"]
    for arrival, snapshot, release in zip(arrivals, snapshots, releases, strict=True):
        wave = alpha_kernel(times - arrival, tau_ms)
        additive += wave
        snapshot_additive += snapshot * wave
        dynamic_additive += release * wave

    recent = np.searchsorted(arrivals, times, side="right") - 1
    last = np.zeros(times.size)
    previous = np.zeros(times.size)
    valid = recent >= 0
    last[valid] = alpha_kernel(times[valid] - arrivals[recent[valid]], tau_ms)
    valid_previous = recent >= 1
    previous[valid_previous] = alpha_kernel(
        times[valid_previous] - arrivals[recent[valid_previous] - 1], tau_ms,
    )
    union = last + previous - last * previous

    matching = first_event_match(
        alpha_tau_ms=tau_ms,
        recovery_ms=recovery_ms,
        depletion_fraction=depletion_fraction,
        delay_ms=delay_ms,
        first_release=type2_parameters.utilization,
    )
    return {
        "classic_union_continuous_resource": union * current_resource,
        "additive_continuous_resource": additive * current_resource,
        "additive_emission_snapshot": snapshot_additive,
        "type2_peak_matched": matching.peak_matched_gain * dynamic_additive,
        "type2_area_matched": matching.area_matched_gain * dynamic_additive,
    }


def _validated_times(times_ms, emissions_ms) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(times_ms, dtype=float)
    emissions = np.asarray(emissions_ms, dtype=float)
    if times.ndim != 1 or emissions.ndim != 1:
        raise ValueError("times and emissions must be one-dimensional")
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(emissions)):
        raise ValueError("times and emissions must be finite")
    if np.any(np.diff(times) <= 0) or np.any(np.diff(emissions) <= 0):
        raise ValueError("times and emissions must be strictly increasing")
    if np.any(times < 0) or np.any(emissions < 0):
        raise ValueError("times and emissions must be nonnegative")
    return times, emissions
