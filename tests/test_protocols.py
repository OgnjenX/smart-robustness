from __future__ import annotations

import numpy as np
import pytest

brian = pytest.importorskip("brian2")

from smart_robustness.classic_sector import build_first_order_intrinsic_sector
from smart_robustness.protocols import (
    BarOrientation,
    ClassicBarStimulus,
    ClassicMatchMismatchCue,
    ConvergentExternalSourceScope,
    MatchCondition,
    apply_bar_stimulus,
    apply_layer6ii_somatic_cue,
    apply_match_mismatch_cue,
    clear_bar_stimulus,
    clear_layer6ii_somatic_cue,
    clear_match_mismatch_cue,
    initialize_convergent_external_input,
)


def test_recovered_bar_patterns_are_centered_five_cell_stimuli() -> None:
    horizontal = ClassicBarStimulus(BarOrientation.HORIZONTAL)
    vertical = ClassicBarStimulus(BarOrientation.VERTICAL)
    assert horizontal.active_indices == (38, 39, 40, 41, 42)
    assert vertical.active_indices == (22, 31, 40, 49, 58)
    assert np.count_nonzero(horizontal.source_grid()) == 5
    assert np.count_nonzero(vertical.source_grid()) == 5
    assert np.all(horizontal.source_grid()[4, 2:7] == 120)
    assert np.all(vertical.source_grid()[2:7, 4] == 120)
    assert np.count_nonzero(horizontal.rgba_grid()[..., 1]) == 5
    assert horizontal.rgba_grid()[4, 4].tolist() == [0, 120, 70, 0]
    assert vertical.rgba_grid()[4, 4].tolist() == [0, 120, 70, 0]


def test_bar_input_reconstructs_published_minus_12mV_drive() -> None:
    brian.start_scope()
    sector = build_first_order_intrinsic_sector(brian=brian)
    stimulus = ClassicBarStimulus(BarOrientation.HORIZONTAL)
    apply_bar_stimulus(sector, stimulus)
    relay = sector.populations["thalamic_relay"]
    port = next(
        port
        for port in relay.compiled.external_input_ports
        if port.record_id == stimulus.relay_input_record_id
    )
    source = np.asarray(getattr(relay.group, f"{port.name}_input_green")[:])
    effective_mV = np.asarray(getattr(relay.group, f"e_{port.name}_effective")[:] / brian.mV)
    assert set(np.flatnonzero(source)) == set(stimulus.active_indices)
    assert effective_mV[list(stimulus.active_indices)] == pytest.approx(-12.0)
    assert effective_mV[0] == pytest.approx(-60.0)

    category = sector.populations["layer6ii_excitatory_v1"]
    category_port = next(
        port
        for port in category.compiled.external_input_ports
        if port.record_id == stimulus.category_input_record_id
    )
    blue = np.asarray(getattr(category.group, f"{category_port.name}_input_blue")[:])
    assert np.flatnonzero(blue).tolist() == [stimulus.center_index]
    assert blue[stimulus.center_index] == stimulus.category_source_value

    nonspecific = sector.populations["thalamic_nonspecific"]
    nonspecific_port = next(
        port
        for port in nonspecific.compiled.external_input_ports
        if port.record_id == stimulus.nonspecific_input_record_id
    )
    assert getattr(nonspecific.group, f"{nonspecific_port.name}_input_green")[0] == 600
    assert getattr(
        nonspecific.group, f"{nonspecific_port.name}_input_source_count"
    )[0] == pytest.approx(5)
    assert getattr(
        nonspecific.group, f"e_{nonspecific_port.name}_effective"
    )[0] / brian.mV == pytest.approx(-40.0)
    matrix = sector.populations["thalamic_matrix"]
    matrix_port = next(
        port
        for port in matrix.compiled.external_input_ports
        if port.record_id == stimulus.matrix_input_record_id
    )
    assert getattr(matrix.group, f"{matrix_port.name}_input_green")[0] == 600
    assert getattr(matrix.group, f"{matrix_port.name}_input_source_count")[0] == pytest.approx(5)

    clear_bar_stimulus(sector, stimulus)
    assert np.count_nonzero(getattr(relay.group, f"{port.name}_input_green")[:]) == 0
    assert np.count_nonzero(getattr(category.group, f"{category_port.name}_input_blue")[:]) == 0
    assert getattr(nonspecific.group, f"{nonspecific_port.name}_input_green")[0] == 0
    assert getattr(nonspecific.group, f"{nonspecific_port.name}_input_source_count")[0] == 1
    assert getattr(matrix.group, f"{matrix_port.name}_input_green")[0] == 0


def test_full_grid_connect_from_all_counts_zero_valued_pixels() -> None:
    brian.start_scope()
    sector = build_first_order_intrinsic_sector(brian=brian)
    stimulus = ClassicBarStimulus(BarOrientation.HORIZONTAL)
    apply_bar_stimulus(
        sector,
        stimulus,
        convergent_source_scope=ConvergentExternalSourceScope.FULL_INPUT_GRID,
    )
    for population_name, record_id in (
        ("thalamic_nonspecific", stimulus.nonspecific_input_record_id),
        ("thalamic_matrix", stimulus.matrix_input_record_id),
    ):
        population = sector.populations[population_name]
        port = next(
            item
            for item in population.compiled.external_input_ports
            if item.record_id == record_id
        )
        assert getattr(population.group, f"{port.name}_input_green")[0] == 600
        assert getattr(
            population.group, f"{port.name}_input_source_count"
        )[0] == pytest.approx(81)
        # At a displaced voltage, black pixels still carry current under the
        # independent-conductance interpretation. Test the explicit sum,
        # rather than just the stored multiplicity.
        voltage_mV = -50.0
        setattr(population.group, f"v_{port.compartment}", voltage_mV * brian.mV)
        leak_mV = float(
            getattr(population.group, f"e_l_{port.compartment}")[0] / brian.mV
        )
        conductance_nS = float(
            getattr(population.group, f"g_{port.name}")[0] / brian.nsiemens
        )
        sources = stimulus.source_grid().ravel()
        expected_pA = conductance_nS * np.sum(
            leak_mV + port.reversal_mV
            + port.sensitivities_mV[1] * sources - voltage_mV
        )
        assert float(getattr(population.group, f"i_{port.name}")[0] / brian.pA) == (
            pytest.approx(expected_pA)
        )

    # Preserve and expose the historical epoch-only lifecycle. It must not be
    # mistaken for a persistent connection whose black pixels remain present.
    clear_bar_stimulus(sector, stimulus)
    for population_name, record_id in (
        ("thalamic_nonspecific", stimulus.nonspecific_input_record_id),
        ("thalamic_matrix", stimulus.matrix_input_record_id),
    ):
        population = sector.populations[population_name]
        port = next(
            item for item in population.compiled.external_input_ports
            if item.record_id == record_id
        )
        assert getattr(population.group, f"{port.name}_input_source_count")[0] == 1


def test_persistent_input_topology_retains_black_pixel_currents_across_epochs() -> None:
    brian.start_scope()
    sector = build_first_order_intrinsic_sector(brian=brian)
    stimulus = ClassicBarStimulus(BarOrientation.HORIZONTAL)
    scope = ConvergentExternalSourceScope.PERSISTENT_FULL_INPUT_GRID
    initialize_convergent_external_input(sector, stimulus, convergent_source_scope=scope)
    for epoch in ("before", "during", "after"):
        if epoch == "during":
            apply_bar_stimulus(sector, stimulus, convergent_source_scope=scope)
        elif epoch == "after":
            clear_bar_stimulus(sector, stimulus, convergent_source_scope=scope)
        for name, record_id in (
            ("thalamic_nonspecific", stimulus.nonspecific_input_record_id),
            ("thalamic_matrix", stimulus.matrix_input_record_id),
        ):
            population = sector.populations[name]
            port = next(p for p in population.compiled.external_input_ports if p.record_id == record_id)
            group = population.group
            setattr(group, f"v_{port.compartment}", -50 * brian.mV)
            leak_mV = float(getattr(group, f"e_l_{port.compartment}")[0] / brian.mV)
            g_nS = float(getattr(group, f"g_{port.name}")[0] / brian.nsiemens)
            green_sum = 600 if epoch == "during" else 0
            expected_pA = g_nS * (
                81 * (leak_mV + port.reversal_mV + 50)
                + port.sensitivities_mV[1] * green_sum
            )
            assert getattr(group, f"{port.name}_input_source_count")[0] == 81
            assert getattr(group, f"{port.name}_input_green")[0] == green_sum
            assert float(getattr(group, f"i_{port.name}")[0] / brian.pA) == pytest.approx(expected_pA)


@pytest.mark.parametrize("figure", [6, 7])
def test_persistent_topology_installed_before_first_protocol_integration(
    monkeypatch, figure: int,
) -> None:
    from smart_robustness.validation import figure6, figure7

    module = figure6 if figure == 6 else figure7
    initialized = []

    def capture_initialization(sector, stimulus, **kwargs):
        initialize_convergent_external_input(sector, stimulus, **kwargs)
        initialized.append((sector, stimulus))

    class FirstIntegrationChecked(Exception):
        pass

    def check_first_run(network, *args, **kwargs):
        assert len(initialized) == 1
        sector, stimulus = initialized[0]
        assert network is sector.network
        for name, record_id in (
            ("thalamic_nonspecific", stimulus.nonspecific_input_record_id),
            ("thalamic_matrix", stimulus.matrix_input_record_id),
        ):
            population = sector.populations[name]
            port = next(
                p for p in population.compiled.external_input_ports
                if p.record_id == record_id
            )
            assert getattr(population.group, f"{port.name}_input_source_count")[0] == 81
            assert getattr(population.group, f"{port.name}_input_green")[0] == 0
        # Stop before integration: this checks protocol wiring, not dynamics.
        raise FirstIntegrationChecked

    monkeypatch.setattr(module, "initialize_convergent_external_input", capture_initialization)
    monkeypatch.setattr(brian.Network, "run", check_first_run)
    scope = ConvergentExternalSourceScope.PERSISTENT_FULL_INPUT_GRID
    with pytest.raises(FirstIntegrationChecked):
        if figure == 6:
            figure6.run_figure6_learning(
                protocol=figure6.Figure6LearningProtocol(warmup_ms=0.01),
                convergent_external_source_scope=scope,
                brian=brian,
            )
        else:
            figure7.run_figure7_condition(
                condition=MatchCondition.MATCH,
                top_down_current_pA=600,
                use_paper_constrained_reference=True,
                top_down_cue_lead_ms=0.01,
                convergent_external_source_scope=scope,
                brian=brian,
            )


def test_optional_relay_gains_only_modulate_local_bottom_up_pixels() -> None:
    brian.start_scope()
    sector = build_first_order_intrinsic_sector(brian=brian)
    stimulus = ClassicBarStimulus(BarOrientation.HORIZONTAL)
    gains = np.ones(81)
    gains[40] = 0.5

    apply_bar_stimulus(sector, stimulus, relay_input_gains=gains)

    relay = sector.populations["thalamic_relay"]
    relay_port = next(
        port
        for port in relay.compiled.external_input_ports
        if port.record_id == stimulus.relay_input_record_id
    )
    source = np.asarray(getattr(relay.group, f"{relay_port.name}_input_green")[:])
    assert source[list(stimulus.active_indices)] == pytest.approx((120, 120, 60, 120, 120))
    nonspecific = sector.populations["thalamic_nonspecific"]
    nonspecific_port = next(
        port
        for port in nonspecific.compiled.external_input_ports
        if port.record_id == stimulus.nonspecific_input_record_id
    )
    assert getattr(nonspecific.group, f"{nonspecific_port.name}_input_green")[0] == 600


@pytest.mark.parametrize(
    ("gains", "message"),
    [
        (np.ones(80), "shape"),
        (np.full(81, np.nan), "finite"),
        (np.full(81, 1.01), r"\[0, 1\]"),
    ],
)
def test_relay_input_gains_are_bounded_9x9_fields(
    gains: np.ndarray, message: str
) -> None:
    brian.start_scope()
    sector = build_first_order_intrinsic_sector(brian=brian)
    with pytest.raises(ValueError, match=message):
        apply_bar_stimulus(
            sector,
            ClassicBarStimulus(BarOrientation.HORIZONTAL),
            relay_input_gains=gains,
        )


def test_match_and_mismatch_share_a_horizontal_top_down_category() -> None:
    match = ClassicMatchMismatchCue(MatchCondition.MATCH, top_down_current_pA=100.0)
    mismatch = ClassicMatchMismatchCue(MatchCondition.MISMATCH, top_down_current_pA=100.0)
    assert match.bottom_up_orientation is BarOrientation.HORIZONTAL
    assert mismatch.bottom_up_orientation is BarOrientation.VERTICAL
    assert match.top_down_expectation is BarOrientation.HORIZONTAL
    assert mismatch.top_down_expectation is BarOrientation.HORIZONTAL
    assert not match.bottom_up_stimulus.include_archived_category_pixel
    assert not mismatch.bottom_up_stimulus.include_archived_category_pixel


def test_match_mismatch_cue_targets_one_layer6ii_cell_and_clears() -> None:
    brian.start_scope()
    sector = build_first_order_intrinsic_sector(brian=brian)
    cue = ClassicMatchMismatchCue(MatchCondition.MATCH, top_down_current_pA=100.0)
    apply_match_mismatch_cue(sector, cue, brian=brian)
    drive_pA = np.asarray(
        sector.populations[cue.top_down_population].group.i_drive_soma[:] / brian.pA
    )
    assert np.flatnonzero(drive_pA).tolist() == [40]
    assert drive_pA[40] == pytest.approx(100.0)
    category = sector.populations[cue.top_down_population]
    category_port = next(
        port
        for port in category.compiled.external_input_ports
        if port.record_id == cue.bottom_up_stimulus.category_input_record_id
    )
    assert not np.any(getattr(category.group, f"{category_port.name}_input_blue")[:])
    clear_match_mismatch_cue(sector, cue, brian=brian)
    assert not np.any(
        sector.populations[cue.top_down_population].group.i_drive_soma[:] / brian.pA
    )


def test_methods_layer6ii_somatic_cue_is_separate_from_archived_blue_input() -> None:
    brian.start_scope()
    sector = build_first_order_intrinsic_sector(brian=brian)
    stimulus = ClassicBarStimulus(
        BarOrientation.HORIZONTAL, include_archived_category_pixel=False
    )
    apply_bar_stimulus(sector, stimulus)
    category = sector.populations["layer6ii_excitatory_v1"]
    category_port = next(
        port
        for port in category.compiled.external_input_ports
        if port.record_id == stimulus.category_input_record_id
    )
    assert not np.any(getattr(category.group, f"{category_port.name}_input_blue")[:])
    assert not np.any(stimulus.rgba_grid()[..., 2])

    apply_layer6ii_somatic_cue(sector, current_pA=200.0, brian=brian)
    drive = np.asarray(category.group.i_drive_soma[:] / brian.pA)
    assert np.flatnonzero(drive).tolist() == [40]
    assert drive[40] == pytest.approx(200.0)
    clear_layer6ii_somatic_cue(sector, brian=brian)
    assert not np.any(category.group.i_drive_soma[:] / brian.pA)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"top_down_current_pA": 0}, "positive"),
        ({"top_down_current_pA": 1, "duration_ms": 0}, "duration"),
        ({"top_down_current_pA": 1, "top_down_cell_index": 81}, "9x9"),
    ],
)
def test_match_mismatch_cue_rejects_undocumented_or_invalid_values(
    kwargs: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ClassicMatchMismatchCue(MatchCondition.MATCH, **kwargs)
