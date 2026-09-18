"""Outcome-independent measurements for isolated apical design 949."""

from __future__ import annotations

import numpy as np

from .active_apical_isolated import ARRIVALS_MS, assay_cases

COMPARTMENTS = ("soma", "proximal_dendrite", "distal_dendrite")


def summarize_response(arrays: dict[str, np.ndarray], dt_ms: float) -> list[dict]:
    """Measure the complete [1000, 1400) ms window on the recording grid.

    Areas and threshold durations use equal-width sample bins (rectangle rule).
    Each response is referenced to the no-input cell at the same drive factor.
    Latency is measured from the first delayed arrival; negative values expose
    pre-input events rather than silently excluding them.
    """
    if dt_ms not in (0.01, 0.005):
        raise ValueError("unregistered timestep")
    cases = assay_cases()
    t = np.asarray(arrays["time_ms"])
    mask = (t >= 1000.0 - dt_ms / 10) & (t < 1400.0 - dt_ms / 10)
    expected = 1000.0 + np.arange(round(400.0 / dt_ms)) * dt_ms
    if t.ndim != 1 or not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise ValueError("time must be finite and strictly increasing")
    if t[mask].shape != expected.shape or not np.allclose(t[mask], expected, atol=1e-8, rtol=0):
        raise ValueError("incomplete or irregular response recording")
    voltage = {}
    for compartment in COMPARTMENTS:
        v = np.asarray(arrays[f"v_{compartment}_mV"])
        if v.shape != (len(cases), len(t)) or not np.all(np.isfinite(v)):
            raise ValueError("invalid or nonfinite voltage array")
        voltage[compartment] = v[:, mask]
    spike_t = np.asarray(arrays["spike_time_ms"])
    spike_i = np.asarray(arrays["spike_cell_index"])
    if (
        spike_t.ndim != 1
        or spike_t.shape != spike_i.shape
        or not np.all(np.isfinite(spike_t))
        or not np.all(np.isfinite(spike_i))
        or np.any(spike_i != np.floor(spike_i))
        or np.any((spike_i < 0) | (spike_i >= len(cases)))
    ):
        raise ValueError("invalid spike records")
    records = []
    for index, case in enumerate(cases):
        reference = next(
            i
            for i, c in enumerate(cases)
            if c.drive_factor == case.drive_factor and c.input_kind == "none"
        )
        spikes = spike_t[(spike_i == index) & (spike_t >= 1000.0) & (spike_t < 1400.0)]
        metrics = {}
        for compartment, values in voltage.items():
            depolarization = values[index] - values[reference]
            metrics[compartment] = {
                "peak_voltage_mV": float(np.max(values[index])),
                "peak_depolarization_mV": float(np.max(depolarization)),
                "positive_area_mV_ms": float(np.maximum(depolarization, 0.0).sum() * dt_ms),
            }
        records.append(
            {
                "drive_factor": case.drive_factor,
                "input_kind": case.input_kind,
                "compartments": metrics,
                "soma_spike_count": len(spikes),
                "first_event_latency_ms": float(np.min(spikes) - ARRIVALS_MS[0])
                if len(spikes)
                else None,
                "distal_above_minus40_ms": float(
                    (voltage["distal_dendrite"][index] > -40.0).sum() * dt_ms
                ),
            }
        )
    for record in records:
        if record["input_kind"] != "distal_and_proximal":
            continue
        singles = [
            r
            for r in records
            if r["drive_factor"] == record["drive_factor"]
            and r["input_kind"] in ("distal_only", "proximal_only")
        ]
        record["paired_area_excess_mV_ms"] = {
            c: record["compartments"][c]["positive_area_mV_ms"]
            - sum(r["compartments"][c]["positive_area_mV_ms"] for r in singles)
            for c in COMPARTMENTS
        }
    return records


def numerical_gate(coarse: list[dict], fine: list[dict]) -> dict:
    """Use the fine-step area as denominator, floored at 1 mV ms."""
    expected = [(c.drive_factor, c.input_kind) for c in assay_cases()]
    for records in (coarse, fine):
        if [(r["drive_factor"], r["input_kind"]) for r in records] != expected:
            raise ValueError("incomplete or reordered assay records")
    failures = []
    for a, b in zip(coarse, fine, strict=True):
        reasons = []
        if a["soma_spike_count"] != b["soma_spike_count"]:
            reasons.append("soma_event_count")
        for c in ("soma", "distal_dendrite"):
            x, y = a["compartments"][c], b["compartments"][c]
            values = [
                x["peak_voltage_mV"],
                y["peak_voltage_mV"],
                x["positive_area_mV_ms"],
                y["positive_area_mV_ms"],
            ]
            if not np.all(np.isfinite(values)):
                reasons.append(f"{c}_nonfinite")
                continue
            if abs(x["peak_voltage_mV"] - y["peak_voltage_mV"]) > 1.0:
                reasons.append(f"{c}_peak")
            if (
                abs(x["positive_area_mV_ms"] - y["positive_area_mV_ms"])
                / max(abs(y["positive_area_mV_ms"]), 1.0)
                > 0.05
            ):
                reasons.append(f"{c}_area")
        if reasons:
            failures.append(
                {
                    "drive_factor": a["drive_factor"],
                    "input_kind": a["input_kind"],
                    "reasons": reasons,
                }
            )
    return {"pass": not failures, "failures": failures}


def mechanism_points(active: list[dict], fixed: list[dict]) -> list[float]:
    """Qualifying paired-input grid points at one timestep; not promotion alone.

    Promotion requires the intersection of points at both timesteps plus all
    engineering and numerical gates. Positive excess must belong to the same
    paired-input response that clears the active-versus-fixed effect floor.
    """
    numerical_gate(active, fixed)  # validates complete case identity, not equality
    points = []
    for a, b in zip(active, fixed, strict=True):
        if a["input_kind"] != "distal_and_proximal" or a["drive_factor"] == 0:
            continue
        x = a["compartments"]["distal_dendrite"]["positive_area_mV_ms"]
        y = b["compartments"]["distal_dendrite"]["positive_area_mV_ms"]
        excess = a["paired_area_excess_mV_ms"]["distal_dendrite"]
        if (
            np.all(np.isfinite([x, y, excess]))
            and x - y >= 1.0
            and x - y >= 0.05 * y
            and excess > 0
        ):
            points.append(a["drive_factor"])
    return points
