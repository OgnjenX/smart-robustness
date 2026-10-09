"""Outcome-blind metadata eligibility controls."""

import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "physiology_eligibility",
    Path(__file__).parents[1] / "scripts/audit_sst_vip_physiology_eligibility.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def structure():
    s = {
        "unit": "Amps",
        "data_shape": [100],
        "sampling_rate": 200000.0,
        "index_start": 0,
        "count": 100,
    }
    r = {**s, "unit": "Volts"}
    return {
        "errors": [],
        "sampling_rates_agree": True,
        "stimulus": s,
        "response": r,
        "experiment_stimulus": s.copy(),
        "experiment_response": r.copy(),
    }


def test_noise_roles_do_not_infer_training_history():
    for name, role in (
        ("Noise 1", "source-fitting-protocol"),
        ("Noise 2", "source-holdout-protocol"),
    ):
        result = MODULE.classify({"stimulus_name": name, "stimulus_units": "Amps"}, structure())
        assert result["eligible_noise_sweep"]
        assert result["role"] == role


def test_missing_and_nonnoise_retained():
    result = MODULE.classify({"stimulus_name": "Test", "stimulus_units": "Volts"}, None)
    assert not result["eligible_noise_sweep"]
    assert set(result["reasons"]) == {
        "outside-noise-validation-scope",
        "not-source-labelled-current-clamp",
        "missing-file-sweep",
    }


def test_alignment_disagreement_rejected():
    data = structure()
    data["experiment_response"]["count"] = 99
    result = MODULE.classify({"stimulus_name": "Noise 2", "stimulus_units": "Amps"}, data)
    assert result["reasons"] == ["experiment_series-alignment-disagreement"]
