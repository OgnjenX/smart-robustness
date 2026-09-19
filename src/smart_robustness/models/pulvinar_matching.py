"""Isolated first-event normalization math; never modifies a SMART network.

Classic continuous-resource delivery changes an alpha kernel's shape even on
the first event. Peak and conductance-area matching therefore need distinct
gains. This is a continuous-time reference, not a Brian scheduling replica.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.optimize import brentq


@dataclass(frozen=True)
class FirstEventMatch:
    classic_peak: float
    classic_peak_after_arrival_ms: float
    classic_area_ms: float
    static_area_ms: float
    peak_matched_gain: float
    area_matched_gain: float


def first_event_match(
    *, alpha_tau_ms: float, recovery_ms: float, depletion_fraction: float,
    delay_ms: float, first_release: float,
) -> FirstEventMatch:
    """Match r*alpha(s) to alpha(s)*[1-epsilon*exp(-(s+delay)/T)].

    s is time after arrival; the source depleted at emission. Initial resource
    is one. Both integrals extend to infinity, excluding all later events.
    Gains multiply the *unnormalized* new release r, not r/r_first.
    No membrane voltage or driving force is included in these quantities.
    """
    values = (alpha_tau_ms, recovery_ms, depletion_fraction, delay_ms, first_release)
    if not all(math.isfinite(v) for v in values):
        raise ValueError("matching parameters must be finite")
    if min(alpha_tau_ms, recovery_ms, first_release) <= 0 or first_release > 1:
        raise ValueError("positive times and first release in (0, 1] required")
    if not 0 <= depletion_fraction <= 1 or delay_ms < 0:
        raise ValueError("invalid depletion fraction or delay")
    tau, recovery = alpha_tau_ms, recovery_ms
    remaining_depletion = depletion_fraction * math.exp(-delay_ms / recovery)

    def log_derivative(s):
        depleted = remaining_depletion * math.exp(-s / recovery)
        # Stable 1 - epsilon*exp(-t/T), even for epsilon=1 and small t/T.
        resource = (1 - remaining_depletion) - remaining_depletion * math.expm1(-s / recovery)
        return 1 / s - 1 / tau + depleted / (recovery * resource)

    peak_time = tau if remaining_depletion == 0 else brentq(
        log_derivative, tau, 2 * tau, xtol=1e-12,
    )
    resource_at_peak = (
        (1 - remaining_depletion)
        - remaining_depletion * math.expm1(-peak_time / recovery)
    )
    peak = math.e * (peak_time / tau) * math.exp(-peak_time / tau) * resource_at_peak
    static_area = math.e * tau
    # Algebraically equivalent to 1-a*(T/(T+tau))**2, avoiding cancellation.
    small_ratio = min(tau, recovery) / max(tau, recovery)
    q = (small_ratio / (1 + small_ratio) if tau <= recovery else 1 / (1 + small_ratio))
    area_fraction = (1 - remaining_depletion) + remaining_depletion * (
        q * (2 - q)
    )
    area = static_area * area_fraction
    return FirstEventMatch(
        peak, peak_time, area, static_area,
        peak / first_release, area_fraction / first_release,
    )
