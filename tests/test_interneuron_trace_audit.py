import runpy
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

module = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/audit_declared_input_interneuron_trace.py")
)
audit = module["audit"]
event_fields = module["EVENT_FIELDS"]
variables = module["REQUIRED_VARIABLES"]
cells = module["EXPECTED_CELLS"]


def fixture():
    match = {field: [] for field in event_fields}
    match.update(
        interneuron_trace_sha256="digest",
        interneuron_spike_indices=[],
        interneuron_spike_times_ms=[],
        cue_lead_interneuron_spike_indices=[],
        cue_lead_interneuron_spike_times_ms=[],
        relay_driven_current_range_pA_by_index_and_source=[
            [cell, "interneuron_gaba", 0.0, 0.0] for cell in cells
        ],
    )
    result = {
        "registration": "registration.yaml",
        "runtime_fingerprint": "runtime",
        "recorded_match_events_repeat_exactly": True,
        "match_result": match,
    }
    reference = {"match_result": {field: [] for field in event_fields}}
    time = np.asarray([-1.0, 0.0, 1.0])
    trace = {
        "schema_version": np.asarray(1),
        "population": np.asarray("thalamic_interneuron"),
        "condition": np.asarray("match"),
        "runtime_fingerprint": np.asarray("runtime"),
        "monitor_when": np.asarray("start"),
        "cell_indices": np.asarray(cells),
        "time_ms": time,
        "variable_names": np.asarray(variables),
        "variable_units": np.asarray([
            "mV" if name.startswith("v_") else "pA" if name.startswith("i_") else "dimensionless"
            for name in variables
        ]),
    }
    for name in variables:
        trace[name] = np.zeros((len(cells), len(time)))
    trace["external_mixed_input_input_source_count"][:] = 1
    for row, cell in enumerate(cells):
        if cell in (38, 39, 40, 41, 42):
            trace["external_mixed_input_input_green"][row, 1:] = 120
    return result, reference, trace


def test_audit_distinguishes_no_events_from_transmission_failure():
    result, reference, trace = fixture()
    report = audit(result, reference, trace, trace_sha256="digest")
    assert report["pathway_classification"] == "no_interneuron_events_during_stimulus"
    changed = deepcopy(result)
    changed["match_result"]["interneuron_spike_indices"] = [40]
    changed["match_result"]["interneuron_spike_times_ms"] = [2.0]
    report = audit(changed, reference, trace, trace_sha256="digest")
    assert report["pathway_classification"] == "interneuron_events_present_but_sampled_relay_output_zero"


@pytest.mark.parametrize("failure", ["checksum", "events", "image", "finite"])
def test_audit_rejects_integrity_failure(failure):
    result, reference, trace = fixture()
    digest = "digest"
    if failure == "checksum":
        digest = "changed"
    elif failure == "events":
        result["match_result"]["relay_spike_times_ms"] = [1]
    elif failure == "image":
        trace["external_mixed_input_input_green"][0, 1] = 120
    else:
        trace["v_soma"][0, 0] = np.nan
    with pytest.raises(ValueError):
        audit(result, reference, trace, trace_sha256=digest)
