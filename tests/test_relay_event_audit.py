import runpy
from pathlib import Path

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
