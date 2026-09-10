from __future__ import annotations

from dataclasses import replace

import pytest

brian = pytest.importorskip("brian2")

from smart_robustness.classic_sector import build_first_order_connected_sector
from smart_robustness.comparison import (
    assert_controlled_substitution,
    snapshot_sector_contract,
)
from smart_robustness.models.adex import (
    create_somatic_adex_population,
    make_somatic_adex_factory,
)
from smart_robustness.models.adex_parameters import LITERATURE_REGULAR_SPIKING
from smart_robustness.models.gif import create_somatic_gif_population


def _contract(factory=None):
    brian.start_scope()
    brian.prefs.codegen.target = "numpy"
    sector = build_first_order_connected_sector(population_factory=factory, brian=brian)
    contract = snapshot_sector_contract(sector)
    sector.network.run(0 * brian.ms)
    return contract


def test_adex_preserves_first_order_topology_ports_delays_and_strengths() -> None:
    classic = _contract()
    adex = _contract(create_somatic_adex_population)
    assert_controlled_substitution(classic, adex)
    assert classic == adex
    assert len(classic.populations) == 12
    assert len(classic.projections) == 53


def test_gif_preserves_first_order_topology_ports_delays_and_strengths() -> None:
    classic = _contract()
    gif = _contract(create_somatic_gif_population)
    assert_controlled_substitution(classic, gif)
    assert classic == gif


def test_controlled_substitution_rejects_a_structural_change() -> None:
    classic = _contract()
    altered = replace(classic, populations=classic.populations[:-1])
    with pytest.raises(ValueError, match="changed the SMART sector contract"):
        assert_controlled_substitution(classic, altered)


def test_per_class_adex_factory_preserves_the_same_contract() -> None:
    classic = _contract()
    adex = _contract(
        make_somatic_adex_factory({"thalamic_relay": LITERATURE_REGULAR_SPIKING})
    )
    assert_controlled_substitution(classic, adex)
