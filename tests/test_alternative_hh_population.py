from __future__ import annotations

import numpy as np
import pytest

brian = pytest.importorskip("brian2")

from smart_robustness.models.alternative_hh import (
    create_somatic_alternative_hh_population,
)
from smart_robustness.models.alternative_hh_parameters import (
    POSPISCHIL_FS_MEAN,
    POSPISCHIL_RS_MEAN,
    AlternativeHHParameters,
)


def _params(cell_class: str) -> dict[str, object]:
    params: dict[str, object] = {
        "cell_class": cell_class,
        "axial_convention": "symmetric_cable",
        "leak_convention": "table3_reversal",
        "voltage_coordinate": "relative_to_table3_leak",
        "nak_rate_convention": "printed_smart",
        "calcium_gate_convention": "reciprocal",
        "calcium_voltage_coordinate": "integrated_voltage",
        "gate_initialization_convention": "steady_state_at_initial_voltage",
        "membrane_initialization_convention": "physical_leak_voltage",
        "spike_event_coordinate": "absolute_physical",
        "spike_event_threshold_mV": -20.0,
        "spike_event_rule": "falling_threshold_crossing",
        "calcium_density_convention": "table3",
        "ahp_convention": "paper_text",
        "specific_capacitance_uF_cm2": 1.0,
        "enable_ahp_ach": cell_class == "layer5_excitatory",
    }
    if cell_class == "layer5_excitatory":
        params.update(
            {
                "ahp_max_conductance_nS": 1.0,
                "ahp_event_weight": 1.0,
                "e_ahp_mV": -90.0,
            }
        )
    return params


def test_alternative_hh_parameter_validation() -> None:
    assert POSPISCHIL_RS_MEAN.sodium_density_mS_cm2 == 50.0
    assert POSPISCHIL_FS_MEAN.potassium_density_mS_cm2 == 5.1
    with pytest.raises(ValueError, match="sodium density must be positive"):
        AlternativeHHParameters.from_mapping(
            POSPISCHIL_RS_MEAN.as_dict() | {"sodium_density_mS_cm2": 0.0}
        )
    with pytest.raises(ValueError, match="parameter keys differ"):
        AlternativeHHParameters.from_mapping(
            POSPISCHIL_RS_MEAN.as_dict() | {"unexpected": 1.0}
        )


def test_alternative_hh_spikes_and_retains_smart_compartments() -> None:
    brian.start_scope()
    brian.prefs.codegen.target = "numpy"
    brian.defaultclock.dt = 0.01 * brian.ms
    population = create_somatic_alternative_hh_population(
        name="alternative_hh_layer4",
        size=2,
        params=_params("layer4_excitatory"),
        brian=brian,
    )
    spikes = brian.SpikeMonitor(population.group)
    voltage = brian.StateMonitor(population.group, "v_soma", record=True)
    population.group.i_drive_soma = 1000 * brian.pA
    brian.Network(population.group, spikes, voltage).run(20 * brian.ms)

    assert population.compiled.somatic_spike_model == "pospischil_hh"
    assert population.compartments == ("soma", "proximal_dendrite")
    assert "i_na_soma" in population.group.variables
    assert "i_k_soma" in population.group.variables
    assert "i_m_soma" in population.group.variables
    assert "p_m_soma" in population.group.variables
    assert len(spikes.t) > 0
    assert np.all(np.isfinite(np.asarray(voltage.v_soma / brian.mV)))


def test_alternative_hh_preserves_active_layer5_dendrite_and_ahp() -> None:
    brian.start_scope()
    layer5 = create_somatic_alternative_hh_population(
        name="alternative_hh_layer5",
        size=1,
        params=_params("layer5_excitatory"),
        brian=brian,
    )
    assert "i_m_soma" in layer5.group.variables
    assert "i_na_distal_dendrite" in layer5.group.variables
    assert "i_ahp" in layer5.group.variables
    assert float(layer5.group.v_t_pospischil[0] / brian.mV) == pytest.approx(-61.5)


def test_inhibitory_class_uses_fast_spiking_literature_seed() -> None:
    brian.start_scope()
    inhibitory = create_somatic_alternative_hh_population(
        name="alternative_hh_layer4i",
        size=1,
        params=_params("layer4_inhibitory"),
        brian=brian,
    )
    assert float(inhibitory.group.v_t_pospischil[0] / brian.mV) == pytest.approx(-61.84)

