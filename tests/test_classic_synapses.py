from __future__ import annotations

import numpy as np
import pytest

brian = pytest.importorskip("brian2")

from smart_robustness.classic_sector import (
    FirstOrderRuntimeConventions,
    build_first_order_chemical_sector,
    build_first_order_connected_sector,
)
from smart_robustness.modeldb_projections import MODELDB_FIRST_ORDER
from smart_robustness.models.currents import biexponential_peak_time_ms
from smart_robustness.projections import SUPPLEMENTARY_TABLE3
from smart_robustness.synapses import (
    GaussianLearningBoundsConvention,
    GaussianWeightConvention,
    ModifiableWeightInitialization,
    connect_modeldb_projection,
    kinness_gap_total_conductance_nS,
    modeldb_topology_pairs,
    topology_pairs,
)


def test_topology_pairs_cover_one_to_one_all_to_one_and_gaussian() -> None:
    one = SUPPLEMENTARY_TABLE3.by_id("l4_e.proximal_dendrite.from_l4_e.ampa")
    pre, post, factor = topology_pairs(one, source_shape=(9, 9), target_shape=(9, 9))
    assert np.array_equal(pre, post)
    assert np.all(factor == 1)

    all_to_one = SUPPLEMENTARY_TABLE3.by_id("l5_e.distal_dendrite.from_matrix.ampa")
    pre, post, factor = topology_pairs(all_to_one, source_shape=(1, 1), target_shape=(9, 9))
    assert len(pre) == 81
    assert set(post) == set(range(81))
    assert np.all(factor == 1)

    gaussian = SUPPLEMENTARY_TABLE3.by_id("l4_i.proximal_dendrite.from_l4_e.ampa")
    pre, post, factor = topology_pairs(gaussian, source_shape=(9, 9), target_shape=(9, 9))
    center = (pre == 40) & (post == 40)
    corner = (pre == 40) & (post == 0)
    assert factor[center][0] > factor[corner][0]


def test_first_order_chemical_sector_builds_all_in_scope_records() -> None:
    brian.start_scope()
    sector = build_first_order_chemical_sector(brian=brian)
    assert len(sector.projections) == 50
    assert all(len(projection) > 0 for projection in sector.projections.values())
    sector.network.run(0 * brian.ms)


def test_modifiable_modeldb_projection_starts_at_source_serialized_weight() -> None:
    brian.start_scope()
    sector = build_first_order_chemical_sector(brian=brian)
    record = next(
        record
        for record in MODELDB_FIRST_ORDER.projections
        if record.modifiable and record.target_population in sector.populations
        and record.source_population in sector.populations
    )
    projection = sector.projections[record.id]
    initial = np.asarray(projection.w[:])
    assert initial.max() == pytest.approx(float(record.weight))
    assert initial.min() >= 0.001
    assert np.asarray(projection.w_maximum[:]) == pytest.approx(float(record.weight))
    assert np.asarray(projection.w_baseline[:]) == pytest.approx(
        float(record.asymptotic_weight)
    )


def test_asymptotic_weight_initialization_remains_an_explicit_audit_alternative() -> None:
    brian.start_scope()
    sector = build_first_order_chemical_sector(
        conventions=FirstOrderRuntimeConventions(
            modifiable_weight_initialization="asymptotic_baseline"
        ),
        brian=brian,
    )
    record = MODELDB_FIRST_ORDER.by_id("modeldb112923.projection.035")
    projection = sector.projections[record.id]
    assert np.asarray(projection.w[:]).max() == pytest.approx(
        float(record.asymptotic_weight)
    )
    assert np.asarray(projection.w_baseline[:]) == pytest.approx(
        float(record.asymptotic_weight)
    )


def test_spatially_scaled_learning_bounds_remain_an_audit_alternative() -> None:
    brian.start_scope()
    sector = build_first_order_chemical_sector(
        conventions=FirstOrderRuntimeConventions(
            gaussian_learning_bounds_convention="spatially_scaled"
        ),
        brian=brian,
    )
    record = MODELDB_FIRST_ORDER.by_id("modeldb112923.projection.035")
    projection = sector.projections[record.id]
    factor = np.asarray(projection.w[:]) / float(record.weight)
    assert np.asarray(projection.w_maximum[:]) == pytest.approx(
        float(record.weight) * factor
    )


def test_fixed_67_mv_spike_coordinate_is_shared_by_plasticity_gate() -> None:
    brian.start_scope()
    sector = build_first_order_chemical_sector(
        conventions=FirstOrderRuntimeConventions(
            spike_event_coordinate="shifted_67_mV"
        ),
        brian=brian,
    )
    projection = sector.projections["modeldb112923.projection.035"]
    assert "v_soma_post+67*mV" in str(projection.equations)


def test_failed_figure6_pathway_candidate_remains_reproducible_but_inactive() -> None:
    brian.start_scope()
    sector = build_first_order_chemical_sector(
        conventions=FirstOrderRuntimeConventions(
            modifiable_weight_initialization=(
                ModifiableWeightInitialization.FIGURE6_PATHWAY_SPECIFIC
            ),
            gaussian_learning_bounds_convention=(
                GaussianLearningBoundsConvention.FIGURE6_PATHWAY_SPECIFIC
            ),
        ),
        brian=brian,
    )
    bottom_record = MODELDB_FIRST_ORDER.by_id("modeldb112923.projection.035")
    bottom = sector.projections[bottom_record.id]
    assert np.asarray(bottom.w_baseline[:]) == pytest.approx(
        float(bottom_record.asymptotic_weight)
    )

    top_record = MODELDB_FIRST_ORDER.by_id("modeldb112923.projection.005")
    top = sector.projections[top_record.id]
    spatial_factor = np.asarray(top.w[:]) / float(top_record.asymptotic_weight)
    assert np.asarray(top.w_baseline[:]) == pytest.approx(
        float(top_record.asymptotic_weight) * spatial_factor
    )
    assert np.asarray(top.w_maximum[:]) == pytest.approx(float(top_record.weight))


def test_first_order_connected_sector_adds_all_gap_junction_records() -> None:
    brian.start_scope()
    sector = build_first_order_connected_sector(brian=brian)
    assert len(sector.projections) == 53
    assert (
        sum(
            len(population.compiled.gap_junction_ports)
            for population in sector.populations.values()
        )
        == 4
    )
    sector.network.run(0 * brian.ms)


def test_modeldb_topology_applies_wrap_and_ring_metadata() -> None:
    wrapped = next(
        record
        for record in MODELDB_FIRST_ORDER.projections
        if record.id == "modeldb112923.projection.000"
    )
    pre, post, factor = modeldb_topology_pairs(wrapped, source_shape=(9, 9), target_shape=(9, 9))
    center_to_left = factor[(pre == 40) & (post == 36)][0]
    center_to_right = factor[(pre == 40) & (post == 44)][0]
    assert center_to_left == pytest.approx(center_to_right)
    assert factor.max() == pytest.approx(1.0)
    assert np.all((factor >= 0) & (factor <= 1))

    ring = next(
        record
        for record in MODELDB_FIRST_ORDER.projections
        if record.kernel is not None and record.kernel.ring
    )
    pre, post, factor = modeldb_topology_pairs(ring, source_shape=(9, 9), target_shape=(9, 9))
    assert not np.any((pre == 40) & (post == 40))


def test_normalized_gaussian_remains_a_paper_figure_audit_alternative() -> None:
    record = MODELDB_FIRST_ORDER.by_id("modeldb112923.projection.035")
    pre, post, factor = modeldb_topology_pairs(
        record,
        source_shape=(9, 9),
        target_shape=(9, 9),
        gaussian_weight_convention=GaussianWeightConvention.NORMALIZED_DENSITY,
    )
    center = factor[(pre == 40) & (post == 40)][0]
    expected = 1 / (2 * np.pi * 0.5**2)
    assert center == pytest.approx(expected)
    assert float(record.weight) * center == pytest.approx(3.819718634)


def test_source_peak_gaussian_is_the_archived_kinness_convention() -> None:
    record = MODELDB_FIRST_ORDER.by_id("modeldb112923.projection.035")
    pre, post, factor = modeldb_topology_pairs(
        record,
        source_shape=(9, 9),
        target_shape=(9, 9),
        gaussian_weight_convention=GaussianWeightConvention.SOURCE_PEAK,
    )
    assert factor[(pre == 40) & (post == 40)][0] == pytest.approx(1.0)


def test_modeldb_gaussian_cuts_resulting_weights_below_legacy_threshold() -> None:
    record = MODELDB_FIRST_ORDER.by_id("modeldb112923.projection.035")
    pre, post, factor = modeldb_topology_pairs(
        record,
        source_shape=(9, 9),
        target_shape=(9, 9),
        gaussian_weight_convention=GaussianWeightConvention.SOURCE_PEAK,
    )
    # At sigma=0.5 and peak Weight=6, a two-cell axial shoulder is 0.0020
    # and survives, whereas the (2,1) diagonal is 0.00027 and is cut.
    assert np.any((pre == 40) & (post == 42))
    assert not np.any((pre == 40) & (post == 51))
    assert np.all(float(record.weight) * factor >= 0.001)


def test_modeldb_projection_accepts_prepartitioned_topology_override() -> None:
    brian.start_scope()
    sector = build_first_order_chemical_sector(brian=brian)
    record = MODELDB_FIRST_ORDER.by_id("modeldb112923.projection.035")
    original = sector.projections[record.id]
    override = connect_modeldb_projection(
        record,
        pre=sector.populations[record.source_population],
        post=sector.populations[record.target_population],
        source_shape=(9, 9),
        target_shape=(9, 9),
        modifiable_weight_initialization="source_serialized_weight",
        gaussian_weight_convention="normalized_density",
        gaussian_learning_bounds_convention="projection_level",
        topology_override=(
            np.asarray(original.i[:3]),
            np.asarray(original.j[:3]),
            np.asarray(original.w[:3]) / float(record.weight),
        ),
        brian=brian,
    )
    assert len(override) == 3
    assert np.asarray(override.w[:]) == pytest.approx(np.asarray(original.w[:3]))


def test_gap_junction_uses_kinness_equation_8_geometry() -> None:
    result = kinness_gap_total_conductance_nS(0.03, diameter_mm=0.001, length_mm=0.005)
    diameter_cm = 0.001 * 0.1
    length_cm = 0.005 * 0.1
    expected = 0.03 * diameter_cm / (4 * length_cm**2) * np.pi * diameter_cm * length_cm * 1e6
    assert result == pytest.approx(expected)


def test_ligand_gate_combines_only_last_two_spikes_and_remains_bounded() -> None:
    brian.start_scope()
    sector = build_first_order_chemical_sector(brian=brian)
    record = MODELDB_FIRST_ORDER.by_id("modeldb112923.projection.035")
    projection = sector.projections[record.id]
    peak = biexponential_peak_time_ms(float(record.rise_ms), float(record.fall_ms))
    projection.last_arrival = -peak * brian.ms
    projection.previous_arrival = -peak * brian.ms
    projection.last_amplitude = 1
    projection.previous_amplitude = 1
    assert np.asarray(projection.last_wave[:]) == pytest.approx(1)
    assert np.asarray(projection.previous_wave[:]) == pytest.approx(1)
    assert np.asarray(projection.pre_signal[:]) == pytest.approx(1)


def test_depleting_projection_scales_ongoing_gate_by_current_resource() -> None:
    brian.start_scope()
    sector = build_first_order_chemical_sector(brian=brian)
    projection = sector.projections["modeldb112923.projection.005"]
    assert "last_amplitude = 1" in projection.pre.code
    updater = projection.summed_updaters["port_005_gate_post"]
    assert "w*pre_signal*transmitter_pre" in updater.abstract_code
    assert float(projection.delay[0] / brian.ms) == pytest.approx(2.0)


def test_depleting_projection_uses_resource_after_delayed_arrival() -> None:
    brian.prefs.codegen.target = "numpy"
    brian.start_scope()
    brian.defaultclock.dt = 0.01 * brian.ms
    sector = build_first_order_chemical_sector(brian=brian)
    source = sector.populations["layer6ii_excitatory_v1"].group
    projection = sector.projections["modeldb112923.projection.005"]
    source_index = 40
    outgoing = np.flatnonzero(np.asarray(projection.i[:]) == source_index)
    assert outgoing.size

    # Put the source directly into Equation 8's armed falling-phase state.
    source.armed[source_index] = 1
    source.v_soma[source_index] = -1 * brian.mV
    sector.network.run(brian.defaultclock.dt)

    assert float(source.transmitter[source_index]) == pytest.approx(0.5)
    assert np.asarray(projection.last_amplitude[:])[outgoing] == pytest.approx(0.0)
    assert np.asarray(projection.pre_signal[:])[outgoing] == pytest.approx(0.0)
    sector.network.run(2 * brian.ms)
    assert np.asarray(projection.last_amplitude[:])[outgoing] == pytest.approx(1.0)
    assert np.asarray(projection.last_arrival[:] / brian.ms)[outgoing] == pytest.approx(2.0)


def test_depleted_resource_continuously_scales_active_ligand_gate() -> None:
    brian.prefs.codegen.target = "numpy"
    brian.start_scope()
    brian.defaultclock.dt = 0.01 * brian.ms
    sector = build_first_order_chemical_sector(brian=brian)
    source = sector.populations["layer6ii_excitatory_v1"].group
    target = sector.populations["thalamic_relay"].group
    projection = sector.projections["modeldb112923.projection.005"]
    projection.last_amplitude = 0
    projection.previous_amplitude = 0
    selected = 0
    source_index = int(projection.i[selected])
    target_index = int(projection.j[selected])
    source.transmitter[source_index] = 0.25
    peak_ms = biexponential_peak_time_ms(2.0, 7.0)
    projection.last_arrival[selected] = -peak_ms * brian.ms
    projection.last_amplitude[selected] = 1
    expected = float(projection.w[selected] * 0.25)
    sector.network.run(brian.defaultclock.dt)
    assert float(target.port_005_gate[target_index]) == pytest.approx(expected, rel=1e-4)


def test_distinct_presynaptic_ligand_currents_sum_per_kinness_equation_16() -> None:
    brian.prefs.codegen.target = "numpy"
    brian.start_scope()
    brian.defaultclock.dt = 0.01 * brian.ms
    sector = build_first_order_chemical_sector(brian=brian)
    projection = sector.projections["modeldb112923.projection.031"]
    target = sector.populations["layer23_excitatory_v1"].group
    projection.last_amplitude = 0
    projection.previous_amplitude = 0
    target_index = 40
    selected = np.flatnonzero(np.asarray(projection.j[:]) == target_index)[:2]
    assert len(np.unique(np.asarray(projection.i[:])[selected])) == 2
    peak_ms = biexponential_peak_time_ms(1.0, 7.0)
    projection.last_arrival[selected] = -peak_ms * brian.ms
    projection.last_amplitude[selected] = 1
    sector.network.run(brian.defaultclock.dt)
    expected = float(np.asarray(projection.w[:])[selected].sum())
    assert float(target.port_000_gate[target_index]) == pytest.approx(expected, rel=1e-4)
