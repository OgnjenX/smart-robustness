"""Source-faithful primitives for Grossberg and Versace (2008) learning."""

from __future__ import annotations

import numpy as np


def gated_weight_derivative(
    *,
    weight: float,
    minimum_weight: float,
    baseline_weight: float,
    maximum_weight: float,
    pre_signal: float,
    post_signal: float,
    learning_rate: float,
    learning_rule: str,
) -> float:
    """Evaluate KInNeSS Equations 25/28 for one SMART synapse."""

    if not minimum_weight <= baseline_weight <= maximum_weight:
        raise ValueError("learning weights must satisfy minimum <= baseline <= maximum")
    if learning_rule == "Presynaptically gated":
        gate = pre_signal**2
    elif learning_rule == "Postsynaptically gated":
        gate = post_signal**2
    elif learning_rule == "Dual AND gated":
        gate = pre_signal * post_signal**2
    else:
        raise ValueError(f"unsupported SMART learning rule {learning_rule!r}")
    bounded_drive = (
        pre_signal * post_signal * (maximum_weight - minimum_weight)
        + baseline_weight
        - weight
    )
    return learning_rate * gate * bounded_drive


def equilibrium_depression_scale(
    *, minimum_weight: float, baseline_weight: float, maximum_weight: float
) -> float:
    """Return Equation 6's D scaling from the three learning bounds."""

    if not minimum_weight <= baseline_weight <= maximum_weight:
        raise ValueError("learning weights must satisfy minimum <= baseline <= maximum")
    if maximum_weight == minimum_weight:
        raise ValueError("learning weight interval must be nonzero")
    return (minimum_weight - baseline_weight) / (maximum_weight - minimum_weight)


def postsynaptic_spike_gate(
    elapsed_ms: float | np.ndarray,
    *,
    depression_scale: float,
    positive_phase_ms: float = 0.1,
    depotentiation_ms: float = 25.0,
) -> float | np.ndarray:
    """Evaluate the post-spike portion of the paper's Equation 6 gate.

    The separate ``V >= V_theta`` branch equals ``D + 1`` and belongs in the
    simulator's threshold condition. This function covers elapsed time after
    the spike: a 0.1-ms linear fall from ``D + 1`` to ``D``, followed by a
    linear fall to zero over the serialized depotentiation interval.
    """

    if not -1 < depression_scale <= 0:
        raise ValueError("depression_scale must satisfy -1 < D <= 0")
    if positive_phase_ms <= 0 or depotentiation_ms <= 0:
        raise ValueError("gate durations must be positive")
    elapsed = np.asarray(elapsed_ms, dtype=float)
    result = np.zeros_like(elapsed)
    first = (elapsed >= 0) & (elapsed < positive_phase_ms)
    result[first] = (
        depression_scale
        + 1.0
        - elapsed[first] / positive_phase_ms
    )
    second = (elapsed >= positive_phase_ms) & (
        elapsed < positive_phase_ms + depotentiation_ms
    )
    result[second] = depression_scale * (
        1.0 - (elapsed[second] - positive_phase_ms) / depotentiation_ms
    )
    if np.isscalar(elapsed_ms):
        return float(result)
    return result
