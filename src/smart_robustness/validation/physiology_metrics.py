"""Fixed operational spike metrics, distinct from Allen explained-variance scores."""

from __future__ import annotations

import numpy as np


def vector(values):
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or not np.isfinite(result).all():
        raise ValueError("expected finite one-dimensional data")
    return result


def native_trace_values(values, conversion, pipeline_version):
    """Pinned SDK conversion predicate, not an inferred physical unit correction."""
    data = vector(values)
    if not np.isfinite(conversion):
        raise ValueError("nonfinite conversion")
    major, minor = pipeline_version
    if major < 0 or minor < 0:
        raise ValueError("invalid pipeline version")
    return data * conversion if major > 1 or (major == 1 and minor > 0) else data.copy()


def upward_crossings(voltage, rate, *, threshold=0.0, refractory=0.001):
    """Linear-interpolated upward voltage crossings; no smoothing or adaptive threshold."""
    data = vector(voltage)
    if (
        not np.isfinite(rate)
        or rate <= 0
        or not np.isfinite(refractory)
        or refractory < 0
        or not np.isfinite(threshold)
    ):
        raise ValueError("invalid detector parameters")
    indices = np.flatnonzero((data[:-1] < threshold) & (data[1:] >= threshold))
    events = []
    for index in indices:
        time = (index + (threshold - data[index]) / (data[index + 1] - data[index])) / rate
        if not events or time - events[-1] >= refractory:
            events.append(float(time))
    return np.asarray(events)


def spike_agreement(observed, predicted, *, tolerance=0.005):
    """Maximum cardinality monotonic one-to-one coincidence matching."""
    a, b = vector(observed), vector(predicted)
    if tolerance < 0 or not np.isfinite(tolerance):
        raise ValueError("invalid coincidence window")
    if np.any(np.diff(a) < 0) or np.any(np.diff(b) < 0):
        raise ValueError("spike times must be sorted")
    i = j = matched = 0
    while i < len(a) and j < len(b):
        if abs(a[i] - b[j]) <= tolerance:
            matched += 1
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    denominator = len(a) + len(b)
    return {
        "observed_spikes": len(a),
        "predicted_spikes": len(b),
        "matches": matched,
        "f1": 2 * matched / denominator if denominator else None,
    }


def recruitment_gate(observed, predicted, duration, repeat_reliability):
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("invalid epoch duration")
    agreement = spike_agreement(observed, predicted)
    rate = agreement["observed_spikes"] / duration
    predicted_rate = agreement["predicted_spikes"] / duration
    error = abs(predicted_rate - rate)
    reliable = (
        repeat_reliability is not None
        and np.isfinite(repeat_reliability)
        and 0.5 <= repeat_reliability <= 1.0
    )
    passed = (
        reliable
        and agreement["f1"] is not None
        and agreement["f1"] >= 0.7
        and error <= max(5.0, 0.2 * rate)
    )
    return {
        **agreement,
        "observed_rate_hz": rate,
        "predicted_rate_hz": predicted_rate,
        "rate_error_hz": error,
        "reliable_recording": bool(reliable),
        "passed": bool(passed),
    }
