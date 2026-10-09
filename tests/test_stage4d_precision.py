"""Local contract checks, separate from the registered 120-case audit."""

from copy import deepcopy
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

from smart_robustness.validation.stage4d_precision import LAGS, compare_band


def curves():
    return {
        "band_hz": [2.0, 4.0],
        "lag_ms": LAGS.tolist(),
        "raw": np.zeros(1000).tolist(),
        "normalized": np.zeros(1000).tolist(),
        "peak_absolute_normalized": 0.0,
    }


def test_zero_energy_requires_exact_zero_raw():
    value = curves()
    assert compare_band(value, value, energy=0, band=(2.0, 4.0))["maximum_scaled_raw_error"] == 0
    changed = deepcopy(value)
    changed["raw"][500] = 1e-30
    with pytest.raises(ValueError, match="raw"):
        compare_band(changed, value, energy=0, band=(2.0, 4.0))


@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
def test_raw_contract_is_dimensionless_but_material_error_rejected(scale):
    value, expected = curves(), curves()
    energy = scale**2
    value["raw"][500] = energy * 1e-13
    result = compare_band(value, expected, energy=energy, band=(2.0, 4.0))
    assert result["maximum_scaled_raw_error"] == pytest.approx(1e-13)
    value["raw"][500] = energy * 1e-4
    with pytest.raises(ValueError, match="raw"):
        compare_band(value, expected, energy=energy, band=(2.0, 4.0))


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -1.0])
def test_energy_must_be_finite_nonnegative(invalid):
    with pytest.raises(ValueError, match="energy"):
        compare_band(curves(), curves(), energy=invalid, band=(2.0, 4.0))


@pytest.mark.parametrize("key", ["raw", "normalized", "peak_absolute_normalized", "lag_ms"])
def test_nonfinite_values_rejected(key):
    value = curves()
    if key == "peak_absolute_normalized":
        value[key] = float("nan")
    else:
        value[key][500] = float("nan")
    with pytest.raises(ValueError):
        compare_band(value, curves(), energy=1, band=(2.0, 4.0))


def test_normalized_tolerances_are_not_rescaled_with_energy():
    value = curves()
    value["normalized"][500] = 1e-4
    with pytest.raises(ValueError, match="normalized"):
        compare_band(value, curves(), energy=1e30, band=(2.0, 4.0))


def test_lag_order_band_and_inventory_remain_exact():
    for mutation in ("lag", "band", "length"):
        value = curves()
        if mutation == "lag":
            value["lag_ms"].reverse()
        elif mutation == "band":
            value["band_hz"] = [4, 8]
        else:
            value["raw"].pop()
        with pytest.raises(ValueError):
            compare_band(value, curves(), energy=1, band=(2.0, 4.0))


def test_registered_positive_exponent_amplitudes_parse_before_any_case(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    runner = import_module("run_l5_sst_like_stage4d_numerical_audit")

    def stop_before_generating(*args):
        raise RuntimeError("Inventory accepted; intentionally stop before synthetic case")

    monkeypatch.setattr(runner, "signals", stop_before_generating)
    with pytest.raises(RuntimeError, match="Inventory accepted"):
        runner.run_audit()
