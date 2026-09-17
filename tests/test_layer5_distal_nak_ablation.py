from __future__ import annotations

import numpy as np
import pytest

brian = pytest.importorskip("brian2")

from smart_robustness.baseline import load_frozen_classic_baseline
from smart_robustness.classic_sector import first_order_population_parameters
from smart_robustness.models.compartmental_hh import create_compartmental_hh_population
from smart_robustness.models.mechanism_interventions import (
    create_layer5_distal_nak_disabled_population,
)
from smart_robustness.models.modeldb112923 import first_order_population_facts


def _layer5_params() -> dict[str, object]:
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


def test_layer5_distal_nak_ablation_removes_only_registered_intrinsic_states() -> None:
    brian.start_scope()
    control = create_compartmental_hh_population(
        name="layer5_distal_nak_control",
        size=1,
        params=_layer5_params(),
        brian=brian,
    )
    ablated = create_layer5_distal_nak_disabled_population(
        name="layer5_distal_nak_ablated",
        size=1,
        params=_layer5_params(),
        brian=brian,
    )

    assert control.compiled.disabled_nak_compartments == frozenset()
    assert ablated.compiled.disabled_nak_compartments == frozenset(
        {"distal_dendrite"}
    )
    for variable in (
        "i_na_distal_dendrite",
        "i_k_distal_dendrite",
        "g_na_distal_dendrite",
        "g_k_distal_dendrite",
        "m_distal_dendrite",
        "h_distal_dendrite",
        "n_distal_dendrite",
    ):
        assert variable in control.group.variables
        assert variable not in ablated.group.variables
    for retained in (
        "v_distal_dendrite",
        "C_distal_dendrite",
        "g_l_distal_dendrite",
        "i_syn_distal_dendrite",
        "i_drive_distal_dendrite",
        "i_ahp",
        "i_na_soma",
        "i_k_soma",
    ):
        assert retained in control.group.variables
        assert retained in ablated.group.variables


def test_layer5_distal_nak_ablation_preserves_structure_ports_and_passive_values() -> None:
    brian.start_scope()
    control = create_compartmental_hh_population(
        name="layer5_structure_control",
        size=1,
        params=_layer5_params(),
        brian=brian,
    )
    ablated = create_layer5_distal_nak_disabled_population(
        name="layer5_structure_ablated",
        size=1,
        params=_layer5_params(),
        brian=brian,
    )

    assert ablated.compartments == control.compartments
    assert ablated.compiled.synaptic_ports == control.compiled.synaptic_ports
    assert ablated.compiled.external_input_ports == control.compiled.external_input_ports
    assert ablated.compiled.injection_ports == control.compiled.injection_ports
    assert ablated.compiled.axial_parameter_names == control.compiled.axial_parameter_names
    for name in ablated.compartments:
        assert float(getattr(ablated.group, f"C_{name}")[0] / brian.pfarad) == pytest.approx(
            float(getattr(control.group, f"C_{name}")[0] / brian.pfarad)
        )
        assert float(
            getattr(ablated.group, f"g_l_{name}")[0] / brian.nsiemens
        ) == pytest.approx(float(getattr(control.group, f"g_l_{name}")[0] / brian.nsiemens))


@pytest.mark.parametrize("factory", [
    create_compartmental_hh_population,
    create_layer5_distal_nak_disabled_population,
])
def test_layer5_distal_current_probe_remains_finite(factory) -> None:
    brian.start_scope()
    brian.prefs.codegen.target = "numpy"
    brian.defaultclock.dt = 0.02 * brian.ms
    population = factory(
        name=f"layer5_finite_{factory.__name__}",
        size=3,
        params=_layer5_params(),
        brian=brian,
    )
    voltage = brian.StateMonitor(
        population.group,
        ("v_soma", "v_distal_dendrite"),
        record=True,
    )
    network = brian.Network(population.group, voltage)
    network.run(20 * brian.ms)
    population.group.i_drive_distal_dendrite = [0, 500, 1500] * brian.pA
    network.run(30 * brian.ms)

    assert np.all(np.isfinite(np.asarray(voltage.v_soma / brian.mV)))
    assert np.all(np.isfinite(np.asarray(voltage.v_distal_dendrite / brian.mV)))


def test_layer5_intervention_rejects_unregistered_targets_and_nested_controls() -> None:
    params = _layer5_params()
    with pytest.raises(ValueError, match="applies only"):
        create_layer5_distal_nak_disabled_population(
            name="wrong_class",
            size=1,
            params=params | {"cell_class": "layer4_excitatory_v1"},
            brian=brian,
        )
    with pytest.raises(ValueError, match="controlled by the intervention"):
        create_layer5_distal_nak_disabled_population(
            name="nested_control",
            size=1,
            params=params | {"disabled_nak_compartments": frozenset()},
            brian=brian,
        )
