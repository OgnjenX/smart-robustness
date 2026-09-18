"""Isolated STP event oracle (registration 955); not wired into SMART.

State names disambiguate decay of facilitation from recovery of resource.
There are intentionally no biological parameter presets or conductance gains.
"""

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class STPParameters:
    utilization: float
    facilitation_decay_per_s: float
    resource_recovery_per_s: float

    def __post_init__(self):
        values = (
            self.utilization,
            self.facilitation_decay_per_s,
            self.resource_recovery_per_s,
        )
        if not all(math.isfinite(v) for v in values):
            raise ValueError("STP parameters must be finite")
        if not 0 < self.utilization <= 1:
            raise ValueError("utilization must be in (0, 1]")
        if min(self.facilitation_decay_per_s, self.resource_recovery_per_s) <= 0:
            raise ValueError("decay and recovery rates must be positive")


def release_history(arrivals_s, parameters: STPParameters) -> dict[str, np.ndarray]:
    """Exact between-event solution, starting fully recovered with u=0.

    At an arrival, update u first, release u*x_pre, then deplete x. All
    returned states and release fractions are dimensionless except time_s.
    Coincident events are rejected: they need an explicit ordering contract.
    """
    arrivals = np.asarray(arrivals_s, dtype=float)
    if arrivals.ndim != 1 or not np.all(np.isfinite(arrivals)):
        raise ValueError("arrivals must be a finite one-dimensional sequence")
    if np.any(arrivals < 0) or np.any(np.diff(arrivals) <= 0):
        raise ValueError("arrivals must be nonnegative and strictly increasing")
    result = {"time_s": arrivals.copy()}
    for key in ("u_pre", "x_pre", "u_post", "x_post", "released"):
        result[key] = np.empty(arrivals.size, dtype=float)
    u, x, previous = 0.0, 1.0, 0.0
    for i, arrival in enumerate(arrivals):
        elapsed = float(arrival) - previous
        u *= math.exp(-parameters.facilitation_decay_per_s * elapsed)
        x = 1.0 - (1.0 - x) * math.exp(-parameters.resource_recovery_per_s * elapsed)
        result["u_pre"][i], result["x_pre"][i] = u, x
        u += parameters.utilization * (1.0 - u)
        released = u * x
        x -= released
        result["u_post"][i], result["x_post"][i] = u, x
        result["released"][i] = released
        previous = float(arrival)
    return result
