"""Construction and summary tests; the sealed scientific grid runs separately."""

import importlib.util
from pathlib import Path

import numpy as np
import yaml


def _runner():
    path = Path(__file__).parents[1] / "scripts/run_pulvinar_conductance_isolated.py"
    spec = importlib.util.spec_from_file_location("pulvinar_conductance_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_small_condition_is_deterministic_and_complete():
    runner = _runner()
    first = runner.build_condition(frequency_hz=20, dt_ms=0.1, events=3, tail_ms=20)
    second = runner.build_condition(frequency_hz=20, dt_ms=0.1, events=3, tail_ms=20)
    assert first.keys() == second.keys()
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    summary = runner.summarize(first)
    assert len(summary) == 5
    assert all(v["finite"] and v["nonnegative"] for v in summary.values())


def test_frequency_changes_emission_schedule_not_registered_mechanisms():
    runner = _runner()
    low = runner.build_condition(frequency_hz=5, dt_ms=0.1, events=3, tail_ms=20)
    high = runner.build_condition(frequency_hz=20, dt_ms=0.1, events=3, tail_ms=20)
    np.testing.assert_array_equal(low["emissions_ms"], [0, 200, 400])
    np.testing.assert_array_equal(high["emissions_ms"], [0, 50, 100])
    assert set(low) == set(high)


def test_sealed_registration_grid_is_at_runner_expected_location():
    runner = _runner()
    registration = yaml.safe_load(runner.REGISTRATION.read_text())
    grid = registration["scope"]
    assert grid["frequencies_hz"] == [0.5, 2, 5, 10, 20]
    assert grid["dt_ms"] == [0.01, 0.005]
    assert grid["events_per_train"] == 10
    assert grid["tail_ms"] == 100.0
