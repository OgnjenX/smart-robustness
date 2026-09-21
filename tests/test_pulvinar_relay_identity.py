"""Source-identity structure and synthetic metric tests only."""

from copy import deepcopy

import numpy as np

from smart_robustness.baseline import load_frozen_classic_baseline
from smart_robustness.validation.pulvinar_relay_identity import (
    build_relay_identity_assay,
    relay_parameters,
    source_faithful_numerical_gate,
    summarize_rest,
)


def _baseline():
    return load_frozen_classic_baseline("configs/baselines/classic_smart_calibrated_v1.yaml")


def test_source_faithful_arm_uses_serialized_cell_without_changing_frozen_arm():
    baseline = _baseline()
    frozen = relay_parameters(baseline, "frozen_dispatch")
    source = relay_parameters(baseline, "source_faithful_modeldb")
    assert frozen["cell_spec"].name == "thalamic_relay"
    assert source["cell_spec"].name == "modeldb112923_v2_relay"
    assert frozen["axial_convention"] == "paper_literal"
    assert source["axial_convention"] == "kinness_serialized_edge"
    assert [c.g_ca_mS_cm2 for c in frozen["cell_spec"].compartments] == [None, 10, 10]
    assert [c.g_ca_mS_cm2 for c in source["cell_spec"].compartments] == [0.1, 0.1, 0.1]
    assert frozen["e_k_mV"] == -90
    assert source["e_k_mV"] == -100


def test_zero_time_compile_has_two_silent_independent_cells():
    import brian2 as brian

    brian.start_scope()
    assay = build_relay_identity_assay(baseline=_baseline(), dt_ms=0.01, brian=brian)
    assay.network.run(0 * brian.ms)
    assert set(assay.populations) == {"frozen_dispatch", "source_faithful_modeldb"}
    for population in assay.populations.values():
        for port in population.compiled.synaptic_ports:
            assert np.all(np.asarray(getattr(population.group, f"{port.name}_gate")) == 0)


def _arrays():
    time = np.arange(900, 1000, 0.1)
    arrays = {"time_ms": time, "dt_ms": np.array(0.1)}
    for arm in ("frozen_dispatch", "source_faithful_modeldb"):
        arrays[f"{arm}_v_soma_mV"] = np.full(time.size, -60.0)
        arrays[f"{arm}_v_proximal_dendrite_mV"] = np.full(time.size, -60.0)
        arrays[f"{arm}_v_distal_dendrite_mV"] = np.full(time.size, -60.0)
        arrays[f"{arm}_spike_time_ms"] = np.array([])
    return arrays


def test_summary_and_prospective_gate():
    summary = summarize_rest(_arrays())
    assert source_faithful_numerical_gate(summary, deepcopy(summary))["pass"]
    changed = deepcopy(summary)
    row = next(x for x in changed["arms"] if x["arm"] == "source_faithful_modeldb")
    row["soma_peak_to_peak_mV"] = 0.11
    row["spike_count"] = 1
    gate = source_faithful_numerical_gate(changed, summary)
    assert not gate["pass"]
    assert set(gate["failures"][0]["reasons"]) == {"soma_not_quiet", "spikes"}
