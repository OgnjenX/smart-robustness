"""Synthetic full-runner composition and authorization controls; no native source execution."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from smart_robustness.validation.native_glif_parity import fixtures
from smart_robustness.validation.native_glif_state_machine import simulate_glif
from smart_robustness.validation.physiology_storage import ArrayStore

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
try:
    SPEC = importlib.util.spec_from_file_location(
        "physiology_runner", SCRIPTS / "run_sst_vip_physiology.py"
    )
    RUNNER = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(RUNNER)
finally:
    sys.path.remove(str(SCRIPTS))


def recording(store):
    stimulus = np.zeros(2000)
    response = np.full(2000, -0.06)
    response[[200, 400, 600, 800]] = 0.03
    return {
        "status": "complete",
        "metadata": {
            "rate": 10000.0,
            "start": 0,
            "count": 2000,
            "stimulus_sha256": hashlib.sha256(stimulus.tobytes()).hexdigest(),
        },
        "arrays": {"stimulus": store.put(stimulus), "response": store.put(response)},
    }


def test_case_repeat_checkpoints_and_resume_no_simulation(tmp_path, monkeypatch):
    case = {"specimen_id": 1, "model_id": 2, "sweep_number": 3}
    parameters = fixtures()[0]["parameters"]
    with ArrayStore(tmp_path, reserve_bytes=0) as store:
        r = recording(store)
        first = RUNNER.run_case(store, 0, case, parameters, r, simulate_glif, {"seal": "synthetic"})
        assert first["parity_passed"]
        assert first["native_exact_repeat"] and first["candidate_exact_repeat"]
        assert len(first["attempts"]) == 4

        def no_run(*_):
            pytest.fail("completed attempts must not run again")

        monkeypatch.setattr(RUNNER, "simulate_glif", no_run)
        assert (
            RUNNER.run_case(store, 0, case, parameters, r, no_run, {"seal": "synthetic"}) == first
        )
        second = {**first, "sweep_number": 4}
        summary = RUNNER.summarize_groups(store, [first, second])
        assert summary["groups"][0]["repeats"] == 2
        assert not summary["models"][0][
            "passed"
        ]  # Silent model is not recruited by synthetic input.


def test_failed_native_does_not_skip_candidate_and_blocks_parity(tmp_path):
    case = {"specimen_id": 1, "model_id": 2, "sweep_number": 3}

    def failure(*_):
        raise ValueError("synthetic native exception")

    with ArrayStore(tmp_path, reserve_bytes=0) as store:
        result = RUNNER.run_case(
            store, 0, case, fixtures()[0]["parameters"], recording(store), failure, {}
        )
        assert not result["parity_passed"]
        assert result["attempts"]["native0"]["status"] == "simulation-exception"
        assert result["attempts"]["candidate1"]["status"] == "complete"


def test_recording_failure_is_retained_and_cannot_vacuously_promote(tmp_path):
    case = {"specimen_id": 1, "model_id": 2, "sweep_number": 3}
    with ArrayStore(tmp_path, reserve_bytes=0) as store:
        result = RUNNER.run_case(store, 0, case, {}, {"status": "recording-failure"}, None, {})
        summary = RUNNER.summarize_groups(store, [result])
        assert len(summary["recording_failures"]) == 1
        assert not summary["models"][0]["passed"]


def test_plan_excludes_noise1_and_identity_exploratory_without_family_selection():
    eligibility = {
        "records": [
            {
                "specimen_id": 1,
                "primary_candidate": True,
                "model_ids": [10, 11],
                "sweeps": [
                    {"sweep_number": 3, "stimulus_name": "Noise 2", "eligible_noise_sweep": True},
                    {"sweep_number": 4, "stimulus_name": "Noise 1", "eligible_noise_sweep": True},
                ],
            },
            {"specimen_id": 2, "primary_candidate": False},
        ]
    }
    schema = {
        "records": [
            {"source": {"specimen_ids": [1]}, "inspection": {"sweeps": [{"sweep_number": 3}]}}
        ]
    }
    parameters = [
        {"source": {"model_id": model, "specimen_id": 1, "template": {"name": "LIF"}}}
        for model in (10, 11)
    ]
    plan = RUNNER.build_plan(eligibility, schema, parameters)
    assert [(c["sweep_number"], c["model_id"]) for c in plan] == [(3, 10), (3, 11)]
    parameters[0]["source"]["specimen_id"] = 2
    with pytest.raises(ValueError, match="attachment mismatch"):
        RUNNER.build_plan(eligibility, schema, parameters)


def test_missing_seal_stops_before_trace_reads(tmp_path):
    registration = tmp_path / "reg.yaml"
    registration.write_text("scope: synthetic\n")
    with pytest.raises(FileNotFoundError):
        RUNNER.verify_seal(registration, tmp_path / "absent-seal.yaml")


def test_authorized_seal_uses_registered_native_source_hash_field_names(tmp_path):
    source = tmp_path / "synthetic-source"
    source.write_bytes(b"synthetic source")
    reg = {"metric_implementation_sha256": {}}
    for key in ("parent", "eligibility_result"):
        reg[key], reg[key + "_sha256"] = str(source), RUNNER.digest(source)
    for key in ("native_reader", "native_simulation"):
        reg[key + "_source"], reg[key + "_sha256"] = str(source), RUNNER.digest(source)
    registration = tmp_path / "reg.yaml"
    registration.write_text(yaml.safe_dump(reg))
    seal = {
        "registration_sha256": RUNNER.digest(registration),
        "response_trace_read_authorized": True,
        "cell_simulation_authorized": True,
        "network_execution_authorized": False,
        "parameter_fitting_authorized": False,
        "implementation_sha256": {str(source): RUNNER.digest(source)},
    }
    seal_path = tmp_path / "seal.yaml"
    seal_path.write_text(yaml.safe_dump(seal))
    assert RUNNER.verify_seal(registration, seal_path)[0] == reg
    seal["cell_simulation_authorized"] = False
    seal_path.write_text(yaml.safe_dump(seal))
    with pytest.raises(ValueError, match="not authorized"):
        RUNNER.verify_seal(registration, seal_path)


def test_recording_checkpoint_resume_validates_source_without_rereading(tmp_path, monkeypatch):
    source_file = tmp_path / "synthetic-source"
    source_file.write_bytes(b"synthetic source bytes, not an experimental recording")
    case = {
        "specimen_id": 1,
        "sweep_number": 3,
        "inventory": {"stimulus": {"data_shape": [2000]}},
        "source": {
            "raw_sha256": RUNNER.digest(source_file),
            "inspection": {"path": str(source_file), "pipeline_version": {}},
        },
    }

    def synthetic_loader(*_, **kwargs):
        assert kwargs["authorized"] is True
        return {
            "stimulus": np.zeros(2000),
            "response": np.zeros(2000),
            "rate": 10000.0,
            "start": 0,
            "count": 2000,
            "stimulus_sha256": "synthetic",
        }

    monkeypatch.setattr(RUNNER, "load_recording", synthetic_loader)
    directory = tmp_path / "store"
    with ArrayStore(directory, reserve_bytes=0) as store:
        first = RUNNER.recording_checkpoint(store, case, {"seal": "synthetic"})

        def no_reread(*_, **__):
            pytest.fail("saved recording must not be reread")

        monkeypatch.setattr(RUNNER, "load_recording", no_reread)
        assert RUNNER.recording_checkpoint(store, case, {"seal": "synthetic"}) == first
        source_file.write_bytes(b"changed synthetic source")
        with pytest.raises(ValueError, match="source changed"):
            RUNNER.recording_checkpoint(store, case, {"seal": "synthetic"})
