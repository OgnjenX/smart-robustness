from __future__ import annotations

import pytest

brian = pytest.importorskip("brian2")

from smart_robustness import classic_sector
from smart_robustness.protocols import MatchCondition
from smart_robustness.validation import figure10_search_cycle_spread
from smart_robustness.validation.figure6 import run_figure6_learning
from smart_robustness.validation.figure7 import run_figure7_condition
from smart_robustness.validation.figure10 import run_figure10_condition
from smart_robustness.validation.figure10_search_cycle import (
    run_figure10_search_cycle_condition,
)
from smart_robustness.validation.figure10_search_cycle_spread import (
    run_figure10_search_cycle_spread_condition,
)
from smart_robustness.validation.figure14 import run_figure14_condition
from smart_robustness.validation.figure15 import run_figure15_condition
from smart_robustness.validation.higher_order import run_figure16_candidate


class FactoryForwarded(RuntimeError):
    pass


def alternative_population_factory(**kwargs):
    raise AssertionError(f"the marker factory must not execute: {kwargs}")


def assert_forwarded_builder(**kwargs):
    assert kwargs["population_factory"] is alternative_population_factory
    raise FactoryForwarded


def test_figure6_forwards_population_factory(monkeypatch) -> None:
    monkeypatch.setattr(
        classic_sector,
        "build_first_order_connected_sector",
        assert_forwarded_builder,
    )
    with pytest.raises(FactoryForwarded):
        run_figure6_learning(
            population_factory=alternative_population_factory,
            brian=brian,
        )


def test_figure7_forwards_population_factory_to_each_network_scope(monkeypatch) -> None:
    monkeypatch.setattr(
        classic_sector,
        "build_first_order_connected_sector",
        assert_forwarded_builder,
    )
    with pytest.raises(FactoryForwarded):
        run_figure7_condition(
            condition=MatchCondition.MATCH,
            top_down_current_pA=1.0,
            use_paper_constrained_reference=True,
            population_factory=alternative_population_factory,
            brian=brian,
        )

    monkeypatch.setattr(
        classic_sector,
        "build_full_smart_network",
        assert_forwarded_builder,
    )
    with pytest.raises(FactoryForwarded):
        run_figure7_condition(
            condition=MatchCondition.MATCH,
            top_down_current_pA=1.0,
            use_paper_constrained_reference=True,
            include_higher_order_loop=True,
            population_factory=alternative_population_factory,
            brian=brian,
        )

    monkeypatch.setattr(
        classic_sector,
        "build_first_order_voltage_clamp_sector",
        assert_forwarded_builder,
    )
    with pytest.raises(FactoryForwarded):
        run_figure7_condition(
            condition=MatchCondition.MATCH,
            top_down_current_pA=1.0,
            use_paper_constrained_reference=True,
            exact_relay_voltage_clamp=True,
            population_factory=alternative_population_factory,
            brian=brian,
        )


def test_figure10_runners_forward_population_factory(monkeypatch) -> None:
    monkeypatch.setattr(
        classic_sector,
        "build_first_order_connected_sector",
        assert_forwarded_builder,
    )
    with pytest.raises(FactoryForwarded):
        run_figure10_condition(
            top_down_current_pA=1.0,
            pre_match_duration_ms=1.0,
            mismatch_duration_ms=1.0,
            reset_pathway_enabled=True,
            population_factory=alternative_population_factory,
            brian=brian,
        )
    with pytest.raises(FactoryForwarded):
        run_figure10_search_cycle_condition(
            top_down_current_pA=1.0,
            pre_match_duration_ms=1.0,
            mismatch_duration_ms=1.0,
            release_after_mismatch_ms=0.5,
            reset_pathway_enabled=True,
            learned_weights={},
            persistent_projection_weight_scales={},
            population_factory=alternative_population_factory,
            brian=brian,
        )

    monkeypatch.setattr(
        figure10_search_cycle_spread,
        "_build_first_order_connected_sector",
        assert_forwarded_builder,
    )
    with pytest.raises(FactoryForwarded):
        run_figure10_search_cycle_spread_condition(
            top_down_current_pA=1.0,
            pre_match_duration_ms=1.0,
            mismatch_duration_ms=1.0,
            release_after_mismatch_ms=0.5,
            reset_pathway_enabled=True,
            learned_weights={},
            persistent_projection_weight_scales={},
            population_factory=alternative_population_factory,
            brian=brian,
        )


def test_spectral_runners_forward_population_factory(monkeypatch) -> None:
    from smart_robustness.validation import figure14, figure15

    monkeypatch.setattr(figure14, "run_figure7_condition", assert_forwarded_builder)
    with pytest.raises(FactoryForwarded):
        run_figure14_condition(
            condition=MatchCondition.MATCH,
            top_down_current_pA=1.0,
            use_paper_constrained_reference=True,
            population_factory=alternative_population_factory,
            brian=brian,
        )

    monkeypatch.setattr(figure15, "run_figure7_condition", assert_forwarded_builder)
    with pytest.raises(FactoryForwarded):
        run_figure15_condition(
            top_down_current_pA=1.0,
            use_paper_constrained_reference=True,
            population_factory=alternative_population_factory,
            brian=brian,
        )


def test_figure16_forwards_population_factory(monkeypatch) -> None:
    monkeypatch.setattr(
        classic_sector,
        "build_full_smart_network",
        assert_forwarded_builder,
    )
    with pytest.raises(FactoryForwarded):
        run_figure16_candidate(
            use_paper_constrained_reference=True,
            population_factory=alternative_population_factory,
            brian=brian,
        )
