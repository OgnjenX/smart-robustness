from __future__ import annotations

import numpy as np

from smart_robustness.validation.figure6 import Figure6MapSummary


def test_map_retention_advantage_detects_horizontal_orientation() -> None:
    before = np.ones(81)
    after = np.full(81, 0.5)
    after[[38, 39, 40, 41, 42]] = 0.9
    summary = Figure6MapSummary("projection", "map", tuple(before), tuple(after))
    assert summary.horizontal_retention_advantage > 0
    assert summary.horizontal_mean > summary.vertical_mean
