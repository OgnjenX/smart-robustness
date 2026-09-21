"""Execution and metrics for the isolated relay assay; no network integration."""

from __future__ import annotations

import tempfile

import numpy as np

from ..models.pulvinar_conductance import CONTROL_NAMES
from ..standalone import build_and_run_cpp_standalone
from .pulvinar_relay_isolated import (
    SETTLING_MS,
    build_isolated_pulvinar_relay_assay,
    merged_recording_intervals,
)


def simulate(*, baseline, frequency_hz: float, dt_ms: float) -> dict[str, np.ndarray]:
    """Run one fresh standalone assay and retain rest plus response windows."""
    import brian2 as brian

    with tempfile.TemporaryDirectory(prefix="smart-pulvinar-relay-", dir="/private/tmp") as temp:
        try:
            brian.set_device("cpp_standalone", directory=temp, build_on_run=False)
            brian.start_scope()
            assay = build_isolated_pulvinar_relay_assay(
                baseline=baseline, frequency_hz=frequency_hz, dt_ms=dt_ms, brian=brian,
            )
            assay.network.run(900 * brian.ms)
            assay.state_monitor.active = True
            assay.network.run(100 * brian.ms)
            current = 0.0
            for start, end in merged_recording_intervals(frequency_hz):
                if start > current:
                    assay.state_monitor.active = False
                    assay.network.run((start - current) * brian.ms)
                assay.state_monitor.active = True
                assay.network.run((end - start) * brian.ms)
                current = end
            build_and_run_cpp_standalone(brian, temp)
            state = assay.state_monitor
            time = np.array(state.t / brian.ms)
            segment = np.zeros(time.size, dtype=np.int16)
            for index, (start, end) in enumerate(merged_recording_intervals(frequency_hz), 1):
                absolute_start = SETTLING_MS + start
                absolute_end = SETTLING_MS + end
                segment[(time >= absolute_start - dt_ms / 10) &
                        (time < absolute_end - dt_ms / 10)] = index
            arrays = {
                "time_ms": time,
                "record_segment_id": segment,
                "spike_time_ms": np.array(assay.spike_monitor.t / brian.ms),
                "spike_cell_index": np.array(assay.spike_monitor.i),
                "dt_ms": np.array(dt_ms),
                "frequency_hz": np.array(frequency_hz),
                "expected_gate": expected_recorded_gate(
                    time, segment, assay.inputs["gate"], dt_ms,
                ),
            }
            for name in state.record_variables:
                value = getattr(state, name)
                if name.startswith("v_"):
                    key, value = name + "_mV", value / brian.mV
                elif name.startswith("i_"):
                    key, value = name + "_pA", value / brian.pA
                else:
                    key = name
                arrays[key] = np.array(value)
            return arrays
        finally:
            brian.device.reinit()
            brian.set_device("runtime")


def expected_recorded_gate(
    recorded_time_ms: np.ndarray,
    segment_id: np.ndarray,
    response_gate: np.ndarray,
    dt_ms: float,
) -> np.ndarray:
    """Map the sealed input to retained physical samples; rest remains zero."""
    expected = np.zeros((len(CONTROL_NAMES), recorded_time_ms.size))
    response = segment_id > 0
    indices = np.rint((recorded_time_ms[response] - SETTLING_MS) / dt_ms).astype(int)
    if np.any(indices < 0) or np.any(indices >= response_gate.shape[0]):
        raise ValueError("recorded response time escapes sealed input")
    expected[:, response] = response_gate[indices].T
    return expected


def summarize(arrays: dict[str, np.ndarray]) -> dict:
    """Summarize each cell without integrating across unrecorded quiet gaps."""
    time = arrays["time_ms"]
    segment = arrays["record_segment_id"]
    dt_ms = float(arrays["dt_ms"])
    if time.ndim != 1 or segment.shape != time.shape or np.any(segment < 0):
        raise ValueError("invalid recording timeline")
    finite = bool(all(np.all(np.isfinite(value)) for value in arrays.values()))
    if not finite:
        raise ValueError("nonfinite relay recording")
    gate = arrays["port_003_gate"]
    if gate.shape != arrays["expected_gate"].shape or not np.array_equal(
        gate, arrays["expected_gate"]
    ):
        raise ValueError("realized gate differs from sealed input")
    rest = segment == 0
    expected_rest_samples = round(100 / dt_ms)
    if np.count_nonzero(rest) != expected_rest_samples:
        raise ValueError("incomplete rest recording")
    soma = arrays["v_soma_mV"]
    proximal = arrays["v_proximal_dendrite_mV"]
    current = arrays["i_port_003_pA"]
    if any(value.shape != gate.shape for value in (soma, proximal, current)):
        raise ValueError("relay state shape mismatch")
    rest_mean = np.mean(soma[:, rest], axis=1)
    rows = []
    for cell, name in enumerate(CONTROL_NAMES):
        response = segment > 0
        positive_area = 0.0
        inward_charge = 0.0
        for segment_index in np.unique(segment[response]):
            mask = segment == segment_index
            positive_area += float(np.trapz(np.maximum(soma[cell, mask] - rest_mean[cell], 0), time[mask]))
            inward_charge += float(np.trapz(np.maximum(-current[cell, mask], 0), time[mask]))
        spike_mask = (
            (arrays["spike_cell_index"] == cell)
            & (arrays["spike_time_ms"] >= SETTLING_MS)
        )
        spike_times = arrays["spike_time_ms"][spike_mask]
        rows.append({
            "control": name,
            "rest_soma_mean_mV": float(rest_mean[cell]),
            "rest_soma_peak_to_peak_mV": float(np.ptp(soma[cell, rest])),
            "response_soma_peak_mV": float(np.max(soma[cell, response])),
            "response_soma_minimum_mV": float(np.min(soma[cell, response])),
            "response_proximal_peak_mV": float(np.max(proximal[cell, response])),
            "positive_soma_area_mV_ms": positive_area,
            "inward_current_charge_pA_ms": inward_charge,
            "response_spike_count": int(spike_times.size),
            "first_response_spike_latency_ms": (
                None if not spike_times.size else float(spike_times[0] - SETTLING_MS)
            ),
        })
    rest_spikes = int(np.count_nonzero(
        (arrays["spike_time_ms"] >= 900) & (arrays["spike_time_ms"] < SETTLING_MS)
    ))
    return {
        "all_recorded_values_finite": finite,
        "realized_gate_exact": True,
        "rest_cells_exactly_identical": bool(np.all(soma[:, rest] == soma[0, rest])),
        "maximum_rest_soma_peak_to_peak_mV": float(np.max(np.ptp(soma[:, rest], axis=1))),
        "rest_spike_count": rest_spikes,
        "controls": rows,
    }


def numerical_gate(coarse: dict, fine: dict) -> dict:
    """Prospective relay numerical gate for 0.01- versus 0.005-ms summaries."""
    failures = []
    for result_name, result in (("coarse", coarse), ("fine", fine)):
        if not (
            result["all_recorded_values_finite"]
            and result["realized_gate_exact"]
            and result["rest_cells_exactly_identical"]
            and result["maximum_rest_soma_peak_to_peak_mV"] <= 0.1
            and result["rest_spike_count"] == 0
        ):
            failures.append({"scope": result_name, "reasons": ["rest-or-integrity"]})
    for left, right in zip(coarse["controls"], fine["controls"], strict=True):
        if left["control"] != right["control"]:
            raise ValueError("control order differs")
        reasons = []
        for metric in ("response_soma_peak_mV", "response_proximal_peak_mV"):
            if abs(left[metric] - right[metric]) > 0.5:
                reasons.append(metric)
        for metric in ("positive_soma_area_mV_ms", "inward_current_charge_pA_ms"):
            denominator = max(abs(right[metric]), 1.0)
            if abs(left[metric] - right[metric]) / denominator > 0.01:
                reasons.append(metric)
        if left["response_spike_count"] != right["response_spike_count"]:
            reasons.append("response_spike_count")
        left_latency, right_latency = (
            left["first_response_spike_latency_ms"], right["first_response_spike_latency_ms"]
        )
        if (left_latency is None) != (right_latency is None) or (
            left_latency is not None and abs(left_latency - right_latency) > 0.05
        ):
            reasons.append("first_response_spike_latency_ms")
        if reasons:
            failures.append({"scope": left["control"], "reasons": reasons})
    return {"pass": not failures, "failures": failures}
