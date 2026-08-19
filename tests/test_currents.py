import math

import pytest

from smart_robustness.models.currents import (
    ACH_FALL_MS,
    ACH_NORMALIZATION,
    ACH_RISE_MS,
    AHP_ACH_SOURCE,
    AHP_FALL_MS,
    AHP_NORMALIZATION,
    AHP_RISE_MS,
    E_CA_MV,
    E_K_MV,
    E_NA_MV,
    G_CA_MSIEMENS_CM2,
    LEAK_CURRENT_EQUATION,
    LEAK_SOURCE,
    T_TYPE_CALCIUM_SOURCE,
    TRAUB_MILES_EQUATIONS,
    TRAUB_MILES_SOURCE,
    NaKRateConvention,
    TTypeGateConvention,
    alpha_h_per_ms,
    alpha_m_per_ms,
    alpha_n_per_ms,
    beta_h_per_ms,
    beta_m_per_ms,
    beta_n_per_ms,
    biexponential_normalization,
    biexponential_peak_time_ms,
    modeldb_t_type_h_inf,
    modeldb_t_type_m_inf,
    modeldb_t_type_tau_h_ms,
    modeldb_t_type_tau_m_ms,
    t_type_calcium_equations,
    t_type_h_inf,
    t_type_m_inf,
    t_type_tau_h_ms,
    t_type_tau_m_ms,
    traub_miles_rates,
)


def test_published_ionic_and_modulatory_constants_are_exact() -> None:
    assert E_K_MV == -90.0
    assert E_NA_MV == 50.0
    assert E_CA_MV == 180.0
    assert G_CA_MSIEMENS_CM2 == 250.0
    assert (AHP_RISE_MS, AHP_FALL_MS) == (80.0, 100.0)
    assert (ACH_RISE_MS, ACH_FALL_MS) == (5.0, 6.0)


@pytest.mark.parametrize(
    ("rise_ms", "fall_ms", "normalization"),
    [
        (AHP_RISE_MS, AHP_FALL_MS, AHP_NORMALIZATION),
        (ACH_RISE_MS, ACH_FALL_MS, ACH_NORMALIZATION),
    ],
)
def test_biexponential_events_are_normalized_to_unit_peak(
    rise_ms: float, fall_ms: float, normalization: float
) -> None:
    peak_ms = biexponential_peak_time_ms(rise_ms, fall_ms)
    value = normalization * (math.exp(-peak_ms / fall_ms) - math.exp(-peak_ms / rise_ms))
    assert value == pytest.approx(1.0)
    assert normalization == pytest.approx(biexponential_normalization(rise_ms, fall_ms))


@pytest.mark.parametrize("rise_ms,fall_ms", [(0, 1), (2, 1), (1, 1), (-1, 2)])
def test_biexponential_normalization_rejects_invalid_constants(
    rise_ms: float, fall_ms: float
) -> None:
    with pytest.raises(ValueError, match="0 < rise_ms < fall_ms"):
        biexponential_normalization(rise_ms, fall_ms)


@pytest.mark.parametrize(
    ("helper", "voltage_mV", "expected_per_ms"),
    [
        (alpha_n_per_ms, 15.0, 0.16),
        (alpha_m_per_ms, 13.0, 0.128),
        (beta_m_per_ms, 40.0, 1.4),
    ],
)
def test_traub_miles_removable_singularities_are_exact(
    helper, voltage_mV: float, expected_per_ms: float
) -> None:
    assert helper(voltage_mV) == expected_per_ms
    assert helper(voltage_mV - 1e-8) == pytest.approx(expected_per_ms, rel=1e-8)
    assert helper(voltage_mV + 1e-8) == pytest.approx(expected_per_ms, rel=1e-8)


def test_traub_miles_rates_match_literal_values_at_zero_millivolts() -> None:
    rates = traub_miles_rates(0.0)
    assert rates.alpha_n == pytest.approx(0.0251499343)
    assert rates.beta_n == pytest.approx(0.64201271)
    assert rates.alpha_m == pytest.approx(0.0167807300)
    assert rates.beta_m == pytest.approx(11.20375844)
    assert rates.alpha_h == pytest.approx(0.57365620)
    assert rates.beta_h == pytest.approx(0.00134140)


def test_standard_traub_miles_correction_is_explicit_and_distinct() -> None:
    printed = traub_miles_rates(0.0, NaKRateConvention.PRINTED_SMART)
    standard = traub_miles_rates(0.0, NaKRateConvention.STANDARD_TRAUB_MILES)
    assert standard.alpha_m == pytest.approx(10 * printed.alpha_m)
    assert standard.alpha_h == pytest.approx(0.128 * math.exp(17 / 18))
    with pytest.raises(TypeError, match="explicit NaKRateConvention"):
        traub_miles_rates(0.0, "standard_traub_miles")  # type: ignore[arg-type]


def test_na_rate_source_conflicts_can_be_decomposed_independently() -> None:
    printed = traub_miles_rates(0.0, NaKRateConvention.PRINTED_SMART)
    archived = traub_miles_rates(0.0, NaKRateConvention.STANDARD_TRAUB_MILES)
    activation_only = traub_miles_rates(
        0.0, NaKRateConvention.ARCHIVED_ACTIVATION_PRINTED_INACTIVATION
    )
    inactivation_only = traub_miles_rates(
        0.0, NaKRateConvention.PRINTED_ACTIVATION_ARCHIVED_INACTIVATION
    )
    assert activation_only.alpha_m == archived.alpha_m
    assert activation_only.alpha_h == printed.alpha_h
    assert inactivation_only.alpha_m == printed.alpha_m
    assert inactivation_only.alpha_h == archived.alpha_h


@pytest.mark.parametrize("voltage_mV", [-200.0, -100.0, -60.0, 0.0, 50.0, 100.0])
def test_all_traub_miles_rates_are_finite_and_nonnegative(voltage_mV: float) -> None:
    rates = traub_miles_rates(voltage_mV)
    for rate in (
        rates.alpha_n,
        rates.beta_n,
        rates.alpha_m,
        rates.beta_m,
        rates.alpha_h,
        rates.beta_h,
    ):
        assert math.isfinite(rate)
        assert rate >= 0.0


def test_individual_rate_helpers_are_the_printed_equations() -> None:
    voltage_mV = -20.0
    assert alpha_n_per_ms(voltage_mV) == pytest.approx(
        0.032 * (15.0 - voltage_mV) / math.expm1((15.0 - voltage_mV) / 5.0)
    )
    assert beta_n_per_ms(voltage_mV) == pytest.approx(0.5 * math.exp((10.0 - voltage_mV) / 40.0))
    assert alpha_m_per_ms(voltage_mV) == pytest.approx(
        0.032 * (13.0 - voltage_mV) / math.expm1((13.0 - voltage_mV) / 4.0)
    )
    assert beta_m_per_ms(voltage_mV) == pytest.approx(
        -0.28 * (40.0 - voltage_mV) / math.expm1((40.0 - voltage_mV) / -5.0)
    )
    assert alpha_h_per_ms(voltage_mV) == pytest.approx(0.128 * math.exp((27.0 - voltage_mV) / 18.0))
    assert beta_h_per_ms(voltage_mV) == pytest.approx(
        4.0 / (math.exp((40.0 - voltage_mV) / 5.0) + 1.0)
    )


def test_t_type_literal_values_preserve_printed_equations() -> None:
    voltage_mV = -60.0
    expected_m = 2.44 + 2.506e-2 * math.exp(-9.84e-2 * voltage_mV)
    expected_h = 19.5 + 7.171e-2 * math.exp(-10.54e-2 * voltage_mV)
    literal = TTypeGateConvention.PRINTED_LITERAL
    assert t_type_m_inf(voltage_mV, literal) == pytest.approx(expected_m)
    assert t_type_h_inf(voltage_mV, literal) == pytest.approx(expected_h)
    assert t_type_m_inf(voltage_mV, literal) > 1.0
    assert t_type_h_inf(voltage_mV, literal) > 1.0


def test_modeldb_t_type_interpretation_has_bounded_voltage_dependent_gates() -> None:
    convention = TTypeGateConvention.MODELDB_112923
    assert t_type_m_inf(-60, convention) == pytest.approx(modeldb_t_type_m_inf(-60))
    assert t_type_h_inf(-80, convention) == pytest.approx(modeldb_t_type_h_inf(-80))
    assert modeldb_t_type_m_inf(-60) > modeldb_t_type_m_inf(-80)
    assert modeldb_t_type_h_inf(-80) > modeldb_t_type_h_inf(-60)
    assert modeldb_t_type_tau_m_ms(-80) > modeldb_t_type_tau_m_ms(-60) > 0
    assert modeldb_t_type_tau_h_ms(-80) > modeldb_t_type_tau_h_ms(-60) > 0


@pytest.mark.parametrize("voltage_mV", range(-120, 61, 10))
def test_t_type_reciprocal_gates_are_bounded_and_exact(voltage_mV: float) -> None:
    literal = TTypeGateConvention.PRINTED_LITERAL
    reciprocal = TTypeGateConvention.RECIPROCAL
    literal_m = t_type_m_inf(voltage_mV, literal)
    literal_h = t_type_h_inf(voltage_mV, literal)
    reciprocal_m = t_type_m_inf(voltage_mV, reciprocal)
    reciprocal_h = t_type_h_inf(voltage_mV, reciprocal)
    assert reciprocal_m == pytest.approx(1.0 / literal_m)
    assert reciprocal_h == pytest.approx(1.0 / literal_h)
    assert 0.0 < reciprocal_m < 1.0
    assert 0.0 < reciprocal_h < 1.0


@pytest.mark.parametrize("voltage_mV", range(-120, 61, 10))
def test_t_type_time_constants_are_finite_and_positive(voltage_mV: float) -> None:
    for time_constant in (t_type_tau_m_ms(voltage_mV), t_type_tau_h_ms(voltage_mV)):
        assert math.isfinite(time_constant)
        assert time_constant > 0.0


def test_t_type_convention_must_be_explicit() -> None:
    with pytest.raises(TypeError, match="explicit TTypeGateConvention"):
        t_type_m_inf(-60.0, "reciprocal")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="explicit TTypeGateConvention"):
        t_type_h_inf(-60.0, None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="explicit TTypeGateConvention"):
        t_type_calcium_equations("printed_literal")  # type: ignore[arg-type]


def test_calcium_equation_fragments_expose_the_selected_interpretation() -> None:
    literal = t_type_calcium_equations(TTypeGateConvention.PRINTED_LITERAL)
    reciprocal = t_type_calcium_equations(TTypeGateConvention.RECIPROCAL)
    assert "m_ca_inf = 2.44 +" in literal
    assert "h_ca_inf = 19.5 +" in literal
    assert "m_ca_inf = 1/(2.44 +" in reciprocal
    assert "h_ca_inf = 1/(19.5 +" in reciprocal
    assert literal != reciprocal
    modeldb = t_type_calcium_equations(TTypeGateConvention.MODELDB_112923)
    assert "m_ca_inf = 1/(exp((-63*mV-v_membrane)" in modeldb
    assert "tau_m_ca = (2.44+" in modeldb
    assert "tau_h_ca = (19.15+" in modeldb


def test_brian2_fragments_parse_and_have_consistent_units() -> None:
    brian = pytest.importorskip("brian2")
    brian.start_scope()
    equations = "\n".join(
        (
            TRAUB_MILES_EQUATIONS,
            LEAK_CURRENT_EQUATION,
            t_type_calcium_equations(TTypeGateConvention.RECIPROCAL),
            "dv_membrane/dt = (i_k+i_na+i_ca+i_leak_printed)/capacitance : volt",
            "v_paper = v_membrane : volt",
            "g_k : siemens (constant)",
            "g_na : siemens (constant)",
            "g_ca : siemens (constant)",
            "g_leak_channel : siemens (constant)",
            "leak_density : 1 (constant)",
            "e_k : volt (constant)",
            "e_na : volt (constant)",
            "e_ca : volt (constant)",
            "capacitance : farad (constant)",
        )
    )
    group = brian.NeuronGroup(1, equations, method="exponential_euler")
    group.v_membrane = 0 * brian.mV
    group.capacitance = 1 * brian.pfarad
    group.g_k = group.g_na = group.g_ca = group.g_leak_channel = 0 * brian.nsiemens
    group.leak_density = 1
    group.e_k = E_K_MV * brian.mV
    group.e_na = E_NA_MV * brian.mV
    group.e_ca = E_CA_MV * brian.mV
    brian.Network(group).run(0 * brian.ms)


def test_primary_source_metadata_records_equations_and_ambiguities() -> None:
    assert TRAUB_MILES_SOURCE.equations == "9-19"
    assert LEAK_SOURCE.equations == "20"
    assert T_TYPE_CALCIUM_SOURCE.equations == "21-27"
    assert AHP_ACH_SOURCE.equations == "3 (reused for AHP) and 28 (ACh complement)"
    assert all(
        source.doi == "10.1016/j.brainres.2008.04.024"
        for source in (
            TRAUB_MILES_SOURCE,
            LEAK_SOURCE,
            T_TYPE_CALCIUM_SOURCE,
            AHP_ACH_SOURCE,
        )
    )
    assert any("without reciprocals" in note for note in T_TYPE_CALCIUM_SOURCE.notes)
    assert any("Table 3" in note for note in LEAK_SOURCE.notes)
