from __future__ import annotations

import warnings
from collections import Counter

import pytest

brian = pytest.importorskip("brian2")

from smart_robustness.classic_sector import build_full_smart_network
from smart_robustness.modeldb_projections import MODELDB_FULL


def test_full_smart_network_builds_all_cells_compartments_and_connections() -> None:
    brian.start_scope()
    brian.prefs.codegen.target = "numpy"
    sector = build_full_smart_network(brian=brian)
    assert len(sector.populations) == 24
    assert sector.cell_count == 1624
    assert sector.compartment_count == 3900
    assert len(sector.projections) == len(MODELDB_FULL.projections) == 118
    assert Counter(record.kind for record in MODELDB_FULL.projections) == {
        "chemical": 109,
        "gap_junction": 9,
    }
    assert sum(
        record.source_population.endswith("_v2")
        != record.target_population.endswith("_v2")
        for record in MODELDB_FULL.projections
    ) == 9
    sector.network.run(0 * brian.ms)


def test_full_smart_network_keeps_v1_and_v2_groups_distinct() -> None:
    brian.start_scope()
    brian.prefs.codegen.target = "numpy"
    sector = build_full_smart_network(brian=brian)
    assert sector.populations["thalamic_relay"].group is not sector.populations[
        "thalamic_relay_v2"
    ].group
    assert int(sector.populations["thalamic_relay_v2"].group.N) == 81
    cross = [
        record
        for record in MODELDB_FULL.projections
        if record.source_population.endswith("_v2")
        != record.target_population.endswith("_v2")
    ]
    for record in cross:
        projection = sector.projections[record.id]
        assert projection.source is sector.populations[record.source_population].group
        assert projection.target is sector.populations[record.target_population].group
    # Mark every assembled Brian object as part of an executable network. This
    # also prevents misleading "never included in a network" teardown warnings.
    sector.network.run(0 * brian.ms)


def test_full_network_projection_selector_is_exact_and_rejects_unknown_ids() -> None:
    brian.start_scope()
    selected = frozenset({"modeldb112923.projection.014", "modeldb112923.projection.016"})
    sector = build_full_smart_network(projection_ids=selected, brian=brian)
    assert sector.projections.keys() == selected
    sector.network.run(0 * brian.ms)

    brian.start_scope()
    with pytest.raises(ValueError, match="unknown full-network projection IDs"):
        build_full_smart_network(projection_ids=frozenset({"missing"}), brian=brian)


def test_inactive_v2_plastic_projection_starts_without_numeric_overflow() -> None:
    brian.start_scope()
    brian.prefs.codegen.target = "numpy"
    sector = build_full_smart_network(
        projection_ids=frozenset({"modeldb112923.projection.085"}), brian=brian
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        sector.network.run(0.01 * brian.ms)
