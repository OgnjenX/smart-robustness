import runpy
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

summarize = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/audit_figure7_relay_events.py")
)["summarize"]


def test_event_summary_preserves_compartments_without_inferring_bursts():
    result = summarize({
        "relay_spike_indices": [22, 22],
        "relay_spike_times_ms": [10.0, 30.0],
        "relay_pre_event_current_samples_pA": [
            [22, 10.0, 0.2, "proximal_calcium", 2.0],
            [22, 30.0, 0.2, "proximal_calcium", 3.0],
            [22, 10.0, 0.2, "soma_axial", 200.0],
        ],
        "relay_pre_event_voltage_samples_mV": [[22, 10.0, 0.2, "soma", 40.0]],
    })
    cell = result["cells"][22]
    assert cell["interevent_intervals_ms"] == [20.0]
    assert cell["sampled_current_ranges_pA_by_label"]["proximal_calcium"] == [2.0, 3.0]
    assert cell["positive_soma_samples_before_emitted_event"] == 1
    assert not result["interpretation"]["burst_mechanism_identified"]


def test_event_summary_rejects_misaligned_samples():
    with pytest.raises(ValueError, match="recorded event"):
        summarize({
            "relay_spike_indices": [22],
            "relay_spike_times_ms": [10.0],
            "relay_pre_event_voltage_samples_mV": [[22, 11.0, 0.2, "soma", 40.0]],
        })


def test_continuous_trace_requires_exact_events_and_preserves_epoch_ranges():
    summarize_trace = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts/audit_figure7_relay_events.py")
    )["summarize_trace"]
    result = {"nonspecific_spike_times_ms": [1.0]}
    for population in ("relay", "trn", "category"):
        result[f"{population}_spike_indices"] = [22]
        result[f"{population}_spike_times_ms"] = [1.0]
    artifact = {"runtime_fingerprint": "test", "mismatch_result": result}
    names = ["v_soma"] + [
        f"{prefix}_{compartment}"
        for prefix in ("v", "m_ca", "h_ca", "i_ca")
        for compartment in ("distal_dendrite", "proximal_dendrite")
    ]
    trace = {
        "schema_version": np.array(1), "condition": np.array("mismatch"),
        "runtime_fingerprint": np.array("test"), "monitor_when": np.array("start"),
        "time_ms": np.array([-1.0, 0.0, 1.0]), "cell_indices": np.array([22]),
        "variable_names": np.array(names),
        "variable_units": np.array([
            "mV" if n.startswith("v_") else "pA" if n.startswith("i_") else "dimensionless"
            for n in names
        ]),
        **{n: np.array([[0.1, 0.2, 0.3]]) for n in names},
    }
    summary = summarize_trace(trace, artifact, deepcopy(artifact))
    assert summary["event_trains_identical_to_reference"]
    assert summary["cells"][22]["pre_stimulus"]["h_ca_distal_dendrite"]["max"] == 0.1
    assert summary["cells"][22]["stimulus"]["h_ca_distal_dendrite"]["max"] == 0.3
    assert not summary["causal_calcium_contribution_identified"]
    changed = deepcopy(artifact)
    changed["mismatch_result"]["relay_spike_times_ms"] = [1.01]
    with pytest.raises(ValueError, match="event train changed"):
        summarize_trace(trace, changed, artifact)
    del trace["h_ca_distal_dendrite"]
    trace["variable_names"] = np.array([n for n in names if n != "h_ca_distal_dendrite"])
    trace["variable_units"] = np.array(["dimensionless"] * (len(names) - 1))
    with pytest.raises(ValueError, match="lacks required"):
        summarize_trace(trace, artifact, artifact)
