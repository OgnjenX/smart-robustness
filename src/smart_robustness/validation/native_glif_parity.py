"""Fixed synthetic fixtures and comparison gates; no native source is executed here."""

from __future__ import annotations

import numpy as np

FIELDS = (
    "voltage",
    "threshold",
    "AScurrents",
    "grid_spike_times",
    "spike_time_steps",
    "interpolated_spike_times",
    "interpolated_spike_voltage",
    "interpolated_spike_threshold",
)


def fixtures():
    cases = []
    for family in ("LIF", "LIF-R", "LIF-ASC", "LIF-R-ASC", "LIF-R-ASC-A"):
        for cut in (0, 3):
            asc = "ASC" in family
            reset = family not in ("LIF", "LIF-ASC")
            threshold = (
                "three_components_exact"
                if family.endswith("-A")
                else "spike_component"
                if reset
                else "inf"
            )
            p = {
                "El": 0.0,
                "dt": 0.0001,
                "R_input": 1.0e8,
                "C": 1.0e-10,
                "asc_tau_array": [0.005, 0.02],
                "asc_amp_array": [-5.0e-11, -1.0e-11],
                "spike_cut_length": cut,
                "th_inf": 0.02,
                "th_adapt": None,
                "init_voltage": 0.0,
                "init_threshold": 0.022,
                "init_AScurrents": [0.0, 0.0],
                "coeffs": {
                    "G": 0.8,
                    "C": 1.2,
                    "th_inf": 1.1,
                    "a": 1.3,
                    "b": 0.9,
                    "asc_amp_array": [1.2, 0.8],
                },
                "AScurrent_dynamics_method": {"name": "exp" if asc else "none", "params": {}},
                "AScurrent_reset_method": {
                    "name": "sum" if asc else "none",
                    "params": {"r": [0.5, 1.0]} if asc else {},
                },
                "voltage_dynamics_method": {"name": "linear_forward_euler", "params": {}},
                "voltage_reset_method": {
                    "name": "v_before" if reset else "zero",
                    "params": {"a": 0.2, "b": -0.002} if reset else {},
                },
                "threshold_dynamics_method": {
                    "name": threshold,
                    "params": {
                        "a_spike": 0.005,
                        "b_spike": 50.0,
                        "a_voltage": 10.0,
                        "b_voltage": 20.0,
                    }
                    if reset
                    else {},
                },
                "threshold_reset_method": {
                    "name": "three_components" if reset else "inf",
                    "params": {"a_spike": 0.005, "b_spike": 50.0} if reset else {},
                },
            }
            cases.append(
                {"identity": f"fixture-{family}-cut-{cut}", "family": family, "parameters": p}
            )
    return cases


def stimuli(parameters, samples=2000):
    p = parameters
    c = p.get("coeffs", {})
    scale = (c.get("G", 1) / p["R_input"]) * (c.get("th_inf", 1) * p["th_inf"] - p["El"])
    if samples != 2000 or not np.isfinite(scale):
        raise ValueError("registered stimulus recipe changed or has nonfinite scale")
    return {
        "zero": np.zeros(samples),
        "subthreshold": np.full(samples, 0.5 * scale),
        "step": np.where(np.arange(samples) < 200, 0.0, 2 * scale),
        "pulses": np.where((np.arange(samples) // 100) % 2 == 0, 2 * scale, 0.0),
    }


def compare(reference, candidate, *, rtol=1e-10, atol=1e-12):
    checks = {}
    if set(reference) != set(FIELDS) or set(candidate) != set(FIELDS):
        raise ValueError("parity output field coverage changed")
    for key in FIELDS:
        a, b = np.asarray(reference[key]), np.asarray(candidate[key])
        shape = a.shape == b.shape
        nan_match = shape and np.array_equal(np.isnan(a), np.isnan(b))
        finite = shape and not np.isinf(a).any() and not np.isinf(b).any()
        values = False
        if nan_match and finite:
            mask = ~np.isnan(a)
            values = (
                np.array_equal(a[mask], b[mask])
                if key in ("spike_time_steps", "grid_spike_times")
                else np.allclose(a[mask], b[mask], rtol=rtol, atol=atol)
            )
        checks[key] = {
            "shape": bool(shape),
            "nan_mask": bool(nan_match),
            "no_infinite_values": bool(finite),
            "values": bool(values),
        }
    return {"passed": all(all(v.values()) for v in checks.values()), "fields": checks}


def exact_repeat(a, b):
    return set(a) == set(b) == set(FIELDS) and all(
        np.array_equal(a[k], b[k], equal_nan=True) for k in FIELDS
    )
