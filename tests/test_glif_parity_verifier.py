"""Independent saved-array gate reconstruction and corruption controls."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

SPEC = importlib.util.spec_from_file_location(
    "independent_glif_verifier",
    Path(__file__).parents[1] / "scripts/verify_sst_vip_glif_parity.py",
)
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


def arrays():
    return {
        prefix + "_" + field: np.asarray([0.0, 1.0])
        for prefix in ("native0", "native1", "candidate0", "candidate1")
        for field in VERIFIER.FIELDS
    }


def test_reconstruct_exact_and_nan_repeats():
    data = arrays()
    for prefix in ("native0", "native1", "candidate0", "candidate1"):
        data[prefix + "_voltage"][1] = np.nan
    gates, repeats = VERIFIER.reconstruct(data)
    assert gates["passed"]
    assert all(repeats.values())


def test_reject_nan_mask_and_infinity():
    data = arrays()
    data["candidate0_voltage"][1] = np.nan
    data["candidate0_threshold"][1] = np.inf
    gates, repeats = VERIFIER.reconstruct(data)
    assert not gates["passed"]
    assert not gates["fields"]["voltage"]["nan_mask"]
    assert not gates["fields"]["threshold"]["no_infinite_values"]
    assert not repeats["candidate"]


def test_spike_grid_exact_not_tolerant_and_repeat_exact():
    data = arrays()
    data["candidate0_grid_spike_times"][1] += 1e-13
    data["candidate1_voltage"][1] += 1e-13
    gates, repeats = VERIFIER.reconstruct(data)
    assert not gates["fields"]["grid_spike_times"]["values"]
    assert gates["fields"]["voltage"]["values"]
    assert not repeats["candidate"]
