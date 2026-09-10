from __future__ import annotations

from dataclasses import replace

import pytest

brian = pytest.importorskip("brian2")

from smart_robustness.models.adex import create_somatic_adex_population
from smart_robustness.models.adex_parameters import LITERATURE_REGULAR_SPIKING
from smart_robustness.validation.isolated_cell_matching import (
    CurrentStepProtocol,
    ReboundPhenotype,
    assess_rebound_match,
    generate_adex_sobol_candidates,
    generate_full_adex_sobol_candidates,
    passive_normalized_currents_pA,
    phenotype_loss,
    run_adex_candidate_batch,
    run_current_step_protocol,
    select_interleaved_current_levels,
)


def _params() -> dict[str, object]:
    return {
        "cell_class": "layer4_excitatory",
        "axial_convention": "symmetric_cable",
        "leak_convention": "table3_reversal",
        "voltage_coordinate": "relative_to_table3_leak",
        "nak_rate_convention": "printed_smart",
        "calcium_gate_convention": "reciprocal",
        "calcium_voltage_coordinate": "integrated_voltage",
        "gate_initialization_convention": "steady_state_at_initial_voltage",
        "membrane_initialization_convention": "physical_leak_voltage",
        "spike_event_coordinate": "absolute_physical",
        "spike_event_threshold_mV": 30.0,
        "spike_event_rule": "latched_peak_then_zero",
        "calcium_density_convention": "table3",
        "ahp_convention": "paper_text",
        "specific_capacitance_uF_cm2": 1.0,
        "enable_ahp_ach": False,
    }


def test_current_step_assay_is_finite_and_self_loss_is_zero() -> None:
    brian.prefs.codegen.target = "numpy"
    protocol = CurrentStepProtocol(
        training_currents_pA=(300.0, 700.0),
        holdout_currents_pA=(500.0,),
        pre_ms=5.0,
        step_ms=20.0,
        post_ms=1.0,
        dt_ms=0.01,
    )
    phenotype = run_current_step_protocol(
        population_factory=create_somatic_adex_population,
        population_params=_params(),
        currents_pA=protocol.training_currents_pA,
        protocol=protocol,
        brian=brian,
    )
    assert phenotype.finite
    assert phenotype.currents_pA == protocol.training_currents_pA
    assert phenotype_loss(phenotype, phenotype) == pytest.approx(0.0)


def test_passive_normalized_currents_use_each_cells_somatic_leak() -> None:
    currents = passive_normalized_currents_pA(_params(), (10.0, 30.0))
    # Layer-4 excitatory soma: 0.01 mS/cm2 * pi*0.05*0.05/100 cm2.
    assert currents == pytest.approx((7.853981633974484, 23.56194490192345))


def test_sobol_candidate_set_is_deterministic_and_includes_literature() -> None:
    bounds = {
        "threshold_offset_mV": (10.0, 30.0),
        "slope_factor_mV": (0.5, 5.0),
        "reset_offset_mV": (-15.0, 5.0),
        "subthreshold_adaptation_nS": (0.0, 10.0),
        "spike_adaptation_pA": (0.0, 150.0),
        "adaptation_time_constant_ms": (40.0, 400.0),
    }
    first = generate_adex_sobol_candidates(bounds, count=32)
    second = generate_adex_sobol_candidates(bounds, count=32)
    assert first == second
    assert len(first) == 33
    assert first[-1].threshold_offset_mV == pytest.approx(20.2)


def test_full_adex_candidates_include_somatic_passive_scales() -> None:
    bounds = {
        "threshold_offset_mV": (0.0, 30.0),
        "slope_factor_mV": (0.25, 5.0),
        "reset_offset_mV": (-20.0, 5.0),
        "subthreshold_adaptation_nS": (0.0, 20.0),
        "spike_adaptation_pA": (0.0, 200.0),
        "adaptation_time_constant_ms": (20.0, 400.0),
        "effective_leak_offset_mV": (-30.0, 10.0),
        "somatic_capacitance_scale": (0.25, 4.0),
        "somatic_leak_conductance_scale": (0.25, 4.0),
    }
    candidates = generate_full_adex_sobol_candidates(
        bounds, count=8, include_literature=False
    )
    assert len(candidates) == 8
    assert candidates[0].somatic_capacitance_scale == pytest.approx(0.25)
    assert candidates[0].somatic_leak_conductance_scale == pytest.approx(0.25)


def test_candidate_batch_and_classic_only_level_selection_are_vectorized() -> None:
    brian.prefs.codegen.target = "numpy"
    protocol = CurrentStepProtocol(
        training_currents_pA=(10.0, 30.0),
        holdout_currents_pA=(20.0,),
        pre_ms=2.0,
        step_ms=5.0,
        post_ms=0.0,
        dt_ms=0.02,
    )
    candidates = (
        LITERATURE_REGULAR_SPIKING,
        replace(LITERATURE_REGULAR_SPIKING, spike_adaptation_pA=40.0),
    )
    phenotypes = run_adex_candidate_batch(
        candidates=candidates,
        population_params=_params(),
        currents_pA=(10.0, 20.0, 30.0),
        protocol=protocol,
        brian=brian,
    )
    assert len(phenotypes) == 2
    assert all(phenotype.finite for phenotype in phenotypes)
    assert select_interleaved_current_levels(phenotypes[0], level_count=3) == (0, 1, 2)


def test_rebound_assessment_requires_matching_presence_rate_and_latency() -> None:
    classic = ReboundPhenotype(0, 0, 3, 0.0, 30.0, 5.0, True)
    matched = ReboundPhenotype(0, 0, 2, 0.0, 20.0, 12.0, True)
    missing = ReboundPhenotype(0, 0, 0, 0.0, 0.0, None, True)
    assert assess_rebound_match(classic, matched).promoted
    assert not assess_rebound_match(classic, missing).promoted
