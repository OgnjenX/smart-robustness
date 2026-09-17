from __future__ import annotations

import numpy as np
import pytest

brian = pytest.importorskip("brian2")

from smart_robustness.models.alternative_hh_parameters import (
    POSPISCHIL_FS_MEAN,
    POSPISCHIL_RS_MEAN,
    AlternativeHHParameters,
)
from smart_robustness.validation.alternative_hh_isolated_cell_matching import (
    AlternativeHHFit,
    alternative_hh_phenotype_loss,
    assess_alternative_hh_match,
    generate_alternative_hh_sobol_candidates,
    run_alternative_hh_candidate_batch,
    select_alternative_hh_candidate,
)
from smart_robustness.validation.isolated_cell_matching import (
    CurrentStepProtocol,
    StepPhenotype,
)

BOUNDS = {
    "threshold_mV": (-75.0, -45.0),
    "sodium_density_mS_cm2": (20.0, 70.0),
    "potassium_density_mS_cm2": (1.0, 12.0),
    "m_current_density_mS_cm2": (0.0, 0.25),
    "m_current_tau_max_ms": (400.0, 2500.0),
}


def _population_params() -> dict[str, object]:
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
        "spike_event_threshold_mV": -20.0,
        "spike_event_rule": "falling_threshold_crossing",
        "calcium_density_convention": "table3",
        "ahp_convention": "paper_text",
        "specific_capacitance_uF_cm2": 1.0,
        "enable_ahp_ach": False,
    }


def _phenotype(
    *,
    resting_voltage_mV: float = -65.0,
    spike_counts: tuple[int, ...] = (0, 3),
    latencies: tuple[float | None, ...] = (None, 20.0),
    adaptation: tuple[float | None, ...] = (None, 1.2),
    finite: bool = True,
) -> StepPhenotype:
    return StepPhenotype(
        currents_pA=(0.0, 500.0),
        resting_voltage_mV=resting_voltage_mV,
        spike_counts=spike_counts,
        firing_rates_hz=tuple(count * 5.0 for count in spike_counts),
        first_spike_latencies_ms=latencies,
        adaptation_ratios=adaptation,
        finite=finite,
    )


def test_sobol_candidates_are_deterministic_bounded_and_seeded() -> None:
    first = generate_alternative_hh_sobol_candidates(BOUNDS, count=8)
    second = generate_alternative_hh_sobol_candidates(BOUNDS, count=8)

    assert first == second
    assert len(first) == 10
    assert first[-2:] == (POSPISCHIL_RS_MEAN, POSPISCHIL_FS_MEAN)
    for candidate in first[:-2]:
        for name, value in candidate.as_dict().items():
            lower, upper = BOUNDS[name]
            assert lower <= value <= upper


def test_candidate_generation_rejects_unregistered_search_spaces() -> None:
    with pytest.raises(ValueError, match="bounds must be exactly"):
        generate_alternative_hh_sobol_candidates(
            BOUNDS | {"unregistered": (0.0, 1.0)}, count=8
        )
    with pytest.raises(ValueError, match="positive power of two"):
        generate_alternative_hh_sobol_candidates(BOUNDS, count=7)


def test_selection_uses_training_loss_then_registered_parameter_order() -> None:
    phenotype = _phenotype()
    lower_threshold = AlternativeHHParameters(
        threshold_mV=-65.0,
        sodium_density_mS_cm2=50.0,
        potassium_density_mS_cm2=5.0,
        m_current_density_mS_cm2=0.1,
        m_current_tau_max_ms=1000.0,
    )
    higher_threshold = AlternativeHHParameters(
        threshold_mV=-60.0,
        sodium_density_mS_cm2=50.0,
        potassium_density_mS_cm2=5.0,
        m_current_density_mS_cm2=0.1,
        m_current_tau_max_ms=1000.0,
    )
    selected = select_alternative_hh_candidate(
        (
            AlternativeHHFit(higher_threshold, 1.0, phenotype),
            AlternativeHHFit(lower_threshold, 1.0, phenotype),
        )
    )
    assert selected.parameters == lower_threshold


def test_loss_and_promotion_gates_detect_phenotype_divergence() -> None:
    target = _phenotype()
    exact = _phenotype()
    divergent = _phenotype(
        resting_voltage_mV=-60.0,
        spike_counts=(1, 6),
        latencies=(10.0, 50.0),
        adaptation=(None, 2.0),
    )

    assert alternative_hh_phenotype_loss(target, exact) == pytest.approx(0.0)
    assert alternative_hh_phenotype_loss(target, divergent) > 0.0
    assert assess_alternative_hh_match(target, exact).promoted is True
    failed = assess_alternative_hh_match(target, divergent)
    assert failed.promoted is False
    assert failed.resting_voltage_pass is False
    assert failed.silent_levels_pass is False
    assert failed.spike_counts_pass is False
    assert failed.first_spike_latencies_pass is False
    assert failed.adaptation_ratios_pass is False


def test_vectorized_batch_returns_one_finite_phenotype_per_candidate() -> None:
    brian.prefs.codegen.target = "numpy"
    protocol = CurrentStepProtocol(
        training_currents_pA=(0.0,),
        holdout_currents_pA=(1.0,),
        pre_ms=2.0,
        step_ms=5.0,
        post_ms=0.0,
        dt_ms=0.02,
    )
    candidates = (POSPISCHIL_RS_MEAN, POSPISCHIL_FS_MEAN)
    phenotypes = run_alternative_hh_candidate_batch(
        candidates=candidates,
        population_params=_population_params(),
        currents_pA=(0.0, 1000.0),
        protocol=protocol,
        brian=brian,
    )

    assert len(phenotypes) == len(candidates)
    assert all(item.currents_pA == (0.0, 1000.0) for item in phenotypes)
    assert all(item.finite for item in phenotypes)
    assert all(np.isfinite(item.resting_voltage_mV) for item in phenotypes)
