import runpy
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

compare = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/audit_relay_calcium_ablation.py"))["compare"]


def fixture_data():
    control = {"relay_spike_indices": [22], "relay_spike_times_ms": [1.0]}
    for population in ("relay", "trn", "category", "nonspecific"):
        control[f"cue_lead_{population}_spike_times_ms"] = []
        if population != "nonspecific":
            control[f"cue_lead_{population}_spike_indices"] = []
    intervention = deepcopy(control)
    intervention["relay_calcium_ablated_at_stimulus"] = True
    names = ["i_ca_distal_dendrite", "i_ca_proximal_dendrite", "v_soma"]
    trace = {
        "time_ms": np.array([-1.0, 0.0, 1.0]), "cell_indices": np.array([22]),
        "variable_names": np.array(names), "variable_units": np.array(["pA", "pA", "mV"]),
        "monitor_when": np.array("start"), "monitor_order": np.array(0),
        "runtime_fingerprint": np.array("test"), "condition": np.array("mismatch"),
        **{n: np.ones((1, 3)) for n in names},
    }
    changed = deepcopy(trace)
    for name in names[:2]:
        changed[name][:, 1:] = 0
    return control, intervention, trace, changed


def test_verified_ablation_does_not_become_reproduction():
    report = compare(*fixture_data())
    assert report["pre_stimulus_state_and_cue_events_identical"]
    assert report["sampled_relay_dendritic_calcium_currents_zero_during_stimulus"]
    assert report["ablation_scope"] == "distal_and_proximal_dendrites_only_somatic_calcium_retained"
    assert report["nonoverlap_cells_still_firing"] == [22]
    assert not report["reproduction_eligible"]


@pytest.mark.parametrize("failure", ["cue", "state", "current"])
def test_audit_rejects_changed_control_or_failed_ablation(failure):
    control, intervention, trace, changed = fixture_data()
    if failure == "cue":
        intervention["cue_lead_relay_spike_times_ms"] = [0.5]
    elif failure == "state":
        changed["v_soma"][0, 0] = 2
    else:
        changed["i_ca_distal_dendrite"][0, 1] = 1
    with pytest.raises(ValueError):
        compare(control, intervention, trace, changed)


def test_whole_relay_ablation_requires_zero_somatic_current():
    control, intervention, trace, changed = fixture_data()
    intervention["relay_calcium_ablation_scope"] = "all_relay_compartments"
    with pytest.raises(ValueError, match="missing intervention observable"):
        compare(control, intervention, trace, changed)
    for item in (trace, changed):
        item["variable_names"] = np.append(item["variable_names"], "i_ca_soma")
        item["variable_units"] = np.append(item["variable_units"], "pA")
        item["i_ca_soma"] = np.ones((1, 3))
    with pytest.raises(ValueError, match="persists after ablation"):
        compare(control, intervention, trace, changed)
    changed["i_ca_soma"][:, 1:] = 0
    report = compare(control, intervention, trace, changed)
    assert report["ablation_scope"] == "all_relay_compartments"
    assert "i_ca_soma" in report["verified_zero_current_variables"]
    assert not report["reproduction_eligible"]
