"""Synthetic-audit-only precision contract registered in artifact 1019."""

from __future__ import annotations

import math

import numpy as np

RTOL = ATOL = 1e-12
BANDS = ((2.0, 4.0), (4.0, 8.0), (8.0, 12.0), (12.0, 20.0), (20.0, 100.0))
LAGS = np.arange(-500, 500)


def compare_band(stored: dict, reference: dict, *, energy: float, band: tuple) -> dict:
    """Check one band without changing normalized metrics or scientific gates."""
    if not math.isfinite(energy) or energy < 0:
        raise ValueError("Nonfinite or negative signal energy")
    if band not in BANDS or any(tuple(v["band_hz"]) != band for v in (stored, reference)):
        raise ValueError("Band identity or order changed")
    for value in (stored, reference):
        for key in ("lag_ms", "raw", "normalized"):
            array = np.asarray(value[key], dtype=float)
            if array.shape != (1000,) or not np.all(np.isfinite(array)):
                raise ValueError(f"Invalid {key} inventory")
        if not np.array_equal(value["lag_ms"], LAGS):
            raise ValueError("Lag order changed")
        peak = np.asarray(value["peak_absolute_normalized"], dtype=float)
        if peak.shape != () or not np.isfinite(peak):
            raise ValueError("Invalid peak")
    raw, expected_raw = np.asarray(stored["raw"]), np.asarray(reference["raw"])
    if energy == 0:
        raw_pass = np.all(raw == 0) and np.all(expected_raw == 0)
        scaled_error = 0.0 if raw_pass else None
    else:
        raw_pass = np.allclose(raw / energy, expected_raw / energy, rtol=RTOL, atol=ATOL)
        scaled_error = float(np.max(np.abs(raw / energy - expected_raw / energy)))
    if not raw_pass:
        raise ValueError("Scale-normalized raw correlation disagrees")
    for key in ("normalized", "peak_absolute_normalized"):
        if not np.allclose(stored[key], reference[key], rtol=RTOL, atol=ATOL):
            raise ValueError(f"Unchanged {key} comparison failed")
    return {
        "original_dimensional_raw_pass": bool(np.allclose(raw, expected_raw, rtol=RTOL, atol=ATOL)),
        "maximum_scaled_raw_error": scaled_error,
        "maximum_normalized_error": float(
            np.max(np.abs(np.asarray(stored["normalized"]) - np.asarray(reference["normalized"])))
        ),
    }
