"""Synthetic baseline statistics do not authorize real pulse/recording joins."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/synphys_baseline_statistics.py"
spec = importlib.util.spec_from_file_location("baseline_statistics", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_repeated_recording_retains_pulse_weight():
    result = module.pulse_baseline_statistics([-0.060, -0.060, -0.080])
    assert result["source_nanmean_reference_volts"] == pytest.approx(-0.2 / 3)
    assert result["source_nanmean_reference_volts"] != pytest.approx(-0.070)
    assert result["complete_finite_baseline_coverage"]
    assert result["finite_only_population_std_volts"] == pytest.approx(np.std([-0.060, -0.060, -0.080]))
    assert result["biological_eligibility_established"] is False


def test_numeric_nan_is_omitted_but_never_complete():
    result = module.pulse_baseline_statistics([-0.06, float("nan"), -0.08])
    assert result["source_nanmean_unambiguous"]
    assert result["source_nanmean_reference_volts"] == pytest.approx(-0.07)
    assert result["nan_count"] == 1
    assert not result["complete_finite_baseline_coverage"]


@pytest.mark.parametrize("missing,field", [(None, "null_count"), (float("inf"), "infinity_count"),
                                        (True, "invalid_type_count"), ("-0.06", "invalid_type_count")])
def test_missing_baseline_is_not_a_source_nan_or_nominal_voltage(missing, field):
    result = module.pulse_baseline_statistics([-0.06, missing])
    assert result[field] == 1
    assert result["finite_only_mean_volts"] == -0.06
    assert result["source_nanmean_reference_volts"] is None
    assert not result["complete_finite_baseline_coverage"]


@pytest.mark.parametrize("values", [[], [None], [float("nan")], [float("nan"), None]])
def test_empty_or_all_missing_has_no_reconstructed_mean(values):
    result = module.pulse_baseline_statistics(values)
    assert result["finite_only_mean_volts"] is None
    assert result["source_nanmean_reference_volts"] is None
    assert result["finite_only_population_std_volts"] is None
    assert not result["complete_finite_baseline_coverage"]


@pytest.mark.parametrize("count", [1, 3, 100, 10000])
def test_roundoff_contract_bounds_numpy_and_reversed_orders(count):
    values = np.linspace(-0.080, -0.060, count).tolist()
    reference = math.fsum(values) / count
    tolerance = module.numerical_mean_bound(values)
    assert abs(float(np.mean(values)) - reference) <= tolerance
    assert abs(float(np.mean(values[::-1])) - reference) <= tolerance
    assert abs(reference - (reference + 1e-6)) > tolerance


@pytest.mark.parametrize("values", [[], [None], [True], [float("nan")], [float("inf")]])
def test_roundoff_bound_not_defined_for_missingness(values):
    with pytest.raises(ValueError):
        module.numerical_mean_bound(values)


def test_pulse_extent_bound():
    with pytest.raises(ValueError, match="bound"):
        module.pulse_baseline_statistics([0.0] * 100001)
    with pytest.raises(ValueError):
        module.numerical_mean_bound([0.0] * 100001)
