from __future__ import annotations

import numpy as np
import pytest

brian = pytest.importorskip("brian2")

from smart_robustness.baseline import load_frozen_classic_baseline
from smart_robustness.classic_sector import first_order_population_parameters
from smart_robustness.models.expanded_layer5 import (
    EXPANDED_LAYER5_TOPOLOGY,
    create_expanded_layer5_population,
    expand_layer5_cell_spec,
    expanded_layer5_parameters,
)
from smart_robustness.models.modeldb112923 import first_order_population_facts


def _params() -> dict[str, object]:
    baseline = load_frozen_classic_baseline(
        "configs/baselines/classic_smart_calibrated_v1.yaml"
    )
    facts = next(
        item
        for item in first_order_population_facts()
        if item.canonical_name == "layer5_excitatory_v1"
    )
    return first_order_population_parameters(
        facts,
        conventions=baseline.runtime_conventions(),
    )


def _total(cell, attribute: str) -> float:
    return sum(
        float(getattr(compartment, attribute) or 0.0)
        * compartment.lateral_area_cm2
        for compartment in cell.compartments
    )


def test_expanded_layer5_conserves_area_and_membrane_conductances() -> None:
    source = _params()["cell_spec"]
    expanded = expand_layer5_cell_spec(source)
    assert tuple(c.name for c in expanded.compartments) == (
        "soma",
        "basal_dendrite",
        "proximal_apical_dendrite",
        "distal_tuft",
    )
    assert sum(c.lateral_area_cm2 for c in expanded.compartments) == pytest.approx(
        sum(c.lateral_area_cm2 for c in source.compartments), rel=1e-12
    )
    for attribute in (
        "g_leak_mS_cm2",
        "g_na_mS_cm2",
        "g_k_mS_cm2",
        "g_ca_mS_cm2",
    ):
        assert _total(expanded, attribute) == pytest.approx(
            _total(source, attribute), rel=1e-12, abs=1e-18
        )
    assert expanded.soma is source.soma
    assert expanded.compartment("distal_tuft").g_na_mS_cm2 == 50.0
    assert expanded.compartment("distal_tuft").g_k_mS_cm2 == 30.0


def test_expanded_layer5_retargets_ports_and_conserves_each_total() -> None:
    params = _params()
    transformed = expanded_layer5_parameters(params)
    source = params["cell_spec"]
    expanded = transformed["cell_spec"]
    mapping = {
        "proximal_dendrite": "basal_dendrite",
        "distal_dendrite": "distal_tuft",
    }
    assert transformed["axial_topology_pairs"] == EXPANDED_LAYER5_TOPOLOGY
    for old, new in zip(
        params["synaptic_ports"], transformed["synaptic_ports"], strict=True
    ):
        assert new.compartment == mapping[old.compartment]
        old_total = (
            old.conductance_density_mS_cm2
            * source.compartment(old.compartment).lateral_area_cm2
        )
        new_total = (
            new.conductance_density_mS_cm2
            * expanded.compartment(new.compartment).lateral_area_cm2
        )
        assert new_total == pytest.approx(old_total, rel=1e-12)
        if old.compartment == "proximal_dendrite":
            assert new.conductance_density_mS_cm2 == pytest.approx(
                old.conductance_density_mS_cm2 * 2.0
            )


def test_expanded_layer5_factory_compiles_registered_branched_tree() -> None:
    brian.start_scope()
    population = create_expanded_layer5_population(
        name="expanded_layer5_test",
        size=1,
        params=_params(),
        brian=brian,
    )
    assert population.compartments == (
        "soma",
        "basal_dendrite",
        "proximal_apical_dendrite",
        "distal_tuft",
    )
    assert population.compiled.axial_topology_pairs == EXPANDED_LAYER5_TOPOLOGY
    assert len(population.compiled.axial_parameter_names) == 6
    assert "v_basal_dendrite" in population.group.variables
    assert "v_proximal_apical_dendrite" in population.group.variables
    assert "v_distal_tuft" in population.group.variables
    assert "i_ahp" in population.group.variables


def test_expanded_layer5_somatic_and_tuft_probe_stays_finite() -> None:
    brian.start_scope()
    brian.prefs.codegen.target = "numpy"
    brian.defaultclock.dt = 0.02 * brian.ms
    population = create_expanded_layer5_population(
        name="expanded_layer5_finite",
        size=3,
        params=_params(),
        brian=brian,
    )
    monitor = brian.StateMonitor(
        population.group,
        ("v_soma", "v_distal_tuft"),
        record=True,
        dt=0.1 * brian.ms,
    )
    network = brian.Network(population.group, monitor)
    network.run(20 * brian.ms)
    population.group.i_drive_soma = [0, 500, 1500] * brian.pA
    population.group.i_drive_distal_tuft = [0, 500, 1500] * brian.pA
    network.run(30 * brian.ms)
    assert np.all(np.isfinite(np.asarray(monitor.v_soma / brian.mV)))
    assert np.all(np.isfinite(np.asarray(monitor.v_distal_tuft / brian.mV)))


def test_expanded_layer5_rejects_wrong_class_and_nested_controls() -> None:
    params = _params()
    with pytest.raises(ValueError, match="applies only"):
        expanded_layer5_parameters(
            params | {"cell_class": "layer4_excitatory_v1"}
        )
    with pytest.raises(ValueError, match="controlled by"):
        expanded_layer5_parameters(
            params | {"axial_topology_pairs": EXPANDED_LAYER5_TOPOLOGY}
        )
