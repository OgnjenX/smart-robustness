"""Synthetic checkpoint execution and preserved failure controls."""

from __future__ import annotations

import numpy as np
import pytest

from smart_robustness.validation.native_glif_parity import FIELDS
from smart_robustness.validation.physiology_attempts import attempt_outputs, stored_attempt
from smart_robustness.validation.physiology_storage import ArrayStore


def reference(parameters, stimulus):
    parameters["asc_tau_array"].append(99.0)
    stimulus[0] = 99.0
    return {key: np.arange(4.0) for key in FIELDS}, {"bad_reset_stop": False}


def test_checkpoint_resume_without_reexecution_and_attempt_dedup(tmp_path):
    p, stimulus = {"asc_tau_array": [1.0, 2.0]}, np.zeros(4)
    context = {"seal": "fixed"}
    with ArrayStore(tmp_path, reserve_bytes=0) as store:
        first = stored_attempt(store, "native0", p, stimulus, reference, context)
        second = stored_attempt(store, "native1", p, stimulus, reference, context)
        assert first["arrays"] == second["arrays"]
        assert len(list(tmp_path.glob("*.npz"))) == 1

        def no_rerun(*_):
            pytest.fail("completed attempt must not execute again")

        assert stored_attempt(store, "native0", p, stimulus, no_rerun, context) == first
        assert set(attempt_outputs(store, first)) == set(FIELDS)
        assert p == {"asc_tau_array": [1.0, 2.0]}
        np.testing.assert_array_equal(stimulus, np.zeros(4))
        with pytest.raises(ValueError, match="context changed"):
            stored_attempt(store, "native0", p, stimulus, no_rerun, {"seal": "changed"})


def test_exception_retained_without_retry(tmp_path):
    calls = []

    def failure(*_):
        calls.append(1)
        raise ValueError("synthetic reset failure")

    with ArrayStore(tmp_path, reserve_bytes=0) as store:
        p, stimulus = {"asc_tau_array": []}, np.zeros(4)
        first = stored_attempt(store, "native0", p, stimulus, failure, {})
        assert first["status"] == "simulation-exception"
        assert stored_attempt(store, "native0", p, stimulus, failure, {}) == first
        assert len(calls) == 1
        with pytest.raises(ValueError, match="failed attempts"):
            attempt_outputs(store, first)


def test_corruption_blocks_resume_not_reexecution(tmp_path):
    with ArrayStore(tmp_path, reserve_bytes=0) as store:
        p, stimulus = {"asc_tau_array": []}, np.zeros(4)
        result = stored_attempt(store, "native0", p, stimulus, reference, {})
        blob = store.path(result["arrays"]["voltage"]["content_sha256"])
        blob.write_bytes(b"corrupted test evidence")
        with pytest.raises(ValueError, match="file changed"):
            stored_attempt(store, "native0", p, stimulus, reference, {})


def test_interrupted_write_retained_without_automatic_resimulation(tmp_path, monkeypatch):
    with ArrayStore(tmp_path, reserve_bytes=0) as store:
        p, stimulus = {"asc_tau_array": []}, np.zeros(4)

        def disk_failure(*_):
            raise OSError("synthetic interruption after simulation")

        monkeypatch.setattr(store, "put", disk_failure)
        with pytest.raises(OSError, match="interruption"):
            stored_attempt(store, "native0", p, stimulus, reference, {})
        assert (tmp_path / "native0-started.json").exists()
        with pytest.raises(ValueError, match="requires assessment"):
            stored_attempt(store, "native0", p, stimulus, reference, {})
