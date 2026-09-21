from __future__ import annotations

from pathlib import Path

import brian2 as brian
import numpy as np
import pytest

from smart_robustness.baseline import load_frozen_classic_baseline
from smart_robustness.validation.l5_sst_like_isolated import (
    CLAMP_MV,
    CONVERGENCE_RELATIVE_TOLERANCES,
    DELAYS_MS,
    DT_MS,
    EMISSIONS_MS,
    RESOURCE_FRACTIONS,
    TOTAL_CONDUCTANCES_NS,
    build_assay,
    exact_trace_repeat,
    load_trace,
    numerical_gate,
    save_trace,
    summarize,
)


def baseline():
    return load_frozen_classic_baseline(
        "configs/baselines/classic_smart_calibrated_v1.yaml"
    )


def synthetic_arrays(*, total_nS: float = TOTAL_CONDUCTANCES_NS[0]):
    time = np.array([0.0, 1.0, 2.0])
    return {
        "time_ms": time,
        "source_spike_times_ms": EMISSIONS_MS.copy(),
        "source_spike_indices": np.zeros(5, dtype=np.int64),
        "target_spike_times_ms": np.array([], dtype=float),
        "target_spike_indices": np.array([], dtype=np.int64),
        "distal_gate": np.array([0.0, 1.0, 0.0]),
        "distal_synaptic_current_pA": np.array([0.0, -15.0, 0.0]),
        "v_soma_mV": np.full(3, CLAMP_MV),
        "v_proximal_dendrite_mV": np.full(3, CLAMP_MV),
        "v_distal_dendrite_mV": np.full(3, CLAMP_MV),
        "delivery_count": np.array([5], dtype=np.int64),
        "requested_total_conductance_nS": np.array(total_nS),
        "realized_total_conductance_nS": np.array(total_nS),
        "resource_fraction": np.array(total_nS / 193.28648801211204),
        "delay_ms": np.array(DELAYS_MS[0]),
        "dt_ms": np.array(DT_MS[0]),
    }


def test_registered_grid_and_construction_are_exact_without_running() -> None:
    assert len(RESOURCE_FRACTIONS) * len(DELAYS_MS) == 12
    assert len(DT_MS) == 2
    for total_nS in TOTAL_CONDUCTANCES_NS:
        brian.start_scope()
        brian.prefs.codegen.target = "numpy"
        assay = build_assay(
            baseline=baseline(),
            total_conductance_nS=total_nS,
            delay_ms=DELAYS_MS[0],
            dt_ms=DT_MS[0],
            brian=brian,
        )
        port = assay["port"]
        area = assay["target"].cell_spec.compartment(port.compartment).lateral_area_cm2
        assert port.conductance_density_mS_cm2 * area * 1e6 == pytest.approx(
            total_nS, abs=1e-12
        )
        assert list(assay["synapse"].i[:]) == [0]
        assert list(assay["synapse"].j[:]) == [0]
        assert all(assay["synapse"].delay[:] == DELAYS_MS[0] * brian.ms)
        assert "delivered" in assay["synapse"].variables


def test_unregistered_coordinates_fail_closed() -> None:
    brian.start_scope()
    with pytest.raises(ValueError, match="unregistered"):
        build_assay(
            baseline=baseline(),
            total_conductance_nS=1.0,
            delay_ms=3.0,
            dt_ms=0.01,
            brian=brian,
        )


def test_synthetic_metrics_repeat_and_convergence_contract() -> None:
    arrays = synthetic_arrays()
    metrics = summarize(arrays)
    assert metrics["per_run_gates_pass"] is True
    assert exact_trace_repeat(arrays, {key: value.copy() for key, value in arrays.items()})
    gate = numerical_gate(metrics, metrics)
    assert gate["pass"] is True
    assert gate["relative_tolerances"] == CONVERGENCE_RELATIVE_TOLERANCES


def test_trace_archive_is_exclusive_and_detects_tampering(tmp_path: Path) -> None:
    path = tmp_path / "trace.npz"
    arrays = synthetic_arrays()
    digest = save_trace(path, arrays)
    restored = load_trace(path, digest)
    assert exact_trace_repeat(arrays, restored)
    with pytest.raises(FileExistsError):
        save_trace(path, arrays)
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="fingerprint"):
        load_trace(path, digest)
