"""SMART local-field-potential and current-source-density analysis."""

from __future__ import annotations

import numpy as np

SMART_EXTRACELLULAR_CONDUCTIVITY_MS_CM = 15.0


def extracellular_potential_uV(
    transmembrane_current_pA: np.ndarray,
    distance_um: np.ndarray,
    *,
    conductivity_mS_cm: float = SMART_EXTRACELLULAR_CONDUCTIVITY_MS_CM,
) -> np.ndarray:
    """Evaluate Grossberg and Versace Equation 31 at each electrode tip.

    Currents are shaped ``(compartment, time)`` and use Equation 32's sign:
    positive values equal axial current entering the compartment. An electrode
    geometry determines whether each signed contribution is a source or sink.
    Distances are shaped ``(tip, compartment)``. The result is ``(tip, time)``
    in microvolts. The conversion is direct because pA/um equals microvolts
    times S/m, and 1 mS/cm equals 0.1 S/m.
    """

    currents = np.asarray(transmembrane_current_pA, dtype=float)
    distances = np.asarray(distance_um, dtype=float)
    if currents.ndim != 2 or distances.ndim != 2:
        raise ValueError("currents and distances must both be two-dimensional")
    if distances.shape[1] != currents.shape[0]:
        raise ValueError("distance compartments must match current compartments")
    if not np.all(np.isfinite(currents)) or not np.all(np.isfinite(distances)):
        raise ValueError("currents and distances must be finite")
    if np.any(distances <= 0):
        raise ValueError("electrode distances must be positive")
    if not np.isfinite(conductivity_mS_cm) or conductivity_mS_cm <= 0:
        raise ValueError("extracellular conductivity must be finite and positive")
    conductivity_s_m = conductivity_mS_cm * 0.1
    return (1.0 / (4.0 * np.pi * conductivity_s_m)) * (1.0 / distances) @ currents


def current_source_density_uV_per_um2(
    electrode_potential_uV: np.ndarray, tip_spacing_um: float
) -> np.ndarray:
    """Apply the centered second-difference approximation in Equation 33.

    End tips do not have two neighbors and are therefore omitted. Input shape
    is ``(tip, time)`` and output shape is ``(tip - 2, time)``.
    """

    potential = np.asarray(electrode_potential_uV, dtype=float)
    if potential.ndim != 2 or potential.shape[0] < 3:
        raise ValueError("potential must contain at least three electrode-tip traces")
    if not np.all(np.isfinite(potential)):
        raise ValueError("electrode potential must be finite")
    if not np.isfinite(tip_spacing_um) or tip_spacing_um <= 0:
        raise ValueError("tip spacing must be finite and positive")
    return (potential[:-2] - 2.0 * potential[1:-1] + potential[2:]) / tip_spacing_um**2
