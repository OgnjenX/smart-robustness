from __future__ import annotations

import numpy as np

from smart_robustness.validation.figure6 import (
    MINIMUM_TOP_DOWN_HORIZONTAL_CONTRAST,
    Figure6LearningProtocol,
    Figure6LearningResult,
    Figure6MapSummary,
)


def test_figure6_learning_protocol_is_the_published_horizontal_episode() -> None:
    protocol = Figure6LearningProtocol()
    assert protocol.stimulus_ms == 100.0
    assert protocol.source_value == 120.0
    assert protocol.category_source_value == 70.0
    assert protocol.winning_layer4_index == 40
    assert protocol.active_category_index == 40
    assert protocol.monitored_populations == (
        "thalamic_relay",
        "layer4_excitatory_v1",
        "layer6ii_excitatory_v1",
    )


def test_map_retention_advantage_detects_horizontal_orientation() -> None:
    before = np.ones(81)
    after = np.full(81, 0.5)
    after[[38, 39, 40, 41, 42]] = 0.9
    summary = Figure6MapSummary("projection", "map", tuple(before), tuple(after))
    assert summary.horizontal_retention_advantage > 0
    assert summary.horizontal_orientation_contrast > 0
    assert summary.horizontal_mean > summary.vertical_mean


def test_absolute_map_contrast_is_stable_for_tiny_gaussian_tails() -> None:
    before = np.full(81, 1e-12)
    after = np.full(81, 1e-12)
    after[[38, 39, 41, 42]] = 2e-4
    after[[22, 31, 49, 58]] = 1e-4
    summary = Figure6MapSummary("projection", "map", tuple(before), tuple(after))

    assert np.isclose(summary.horizontal_orientation_contrast, 1e-4)


def test_tiny_positive_top_down_contrast_does_not_count_as_reproduction() -> None:
    before = np.ones(81)
    after = np.ones(81)
    after[[38, 39, 41, 42]] += MINIMUM_TOP_DOWN_HORIZONTAL_CONTRAST / 2
    weak = Figure6MapSummary("topdown", "map", tuple(before), tuple(after))
    bottom = Figure6MapSummary(
        "bottom", "map", tuple(before), tuple(after + np.eye(1, 81, 38).ravel())
    )
    result = Figure6LearningResult("fingerprint", 100.0, {}, bottom, weak, weak)

    assert weak.horizontal_orientation_contrast > 0
    assert not result.top_down_oriented
