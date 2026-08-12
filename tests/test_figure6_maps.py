from __future__ import annotations

import numpy as np

from smart_robustness.validation.figure6 import Figure6MapSummary


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
