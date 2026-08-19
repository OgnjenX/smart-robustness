from __future__ import annotations

import pytest

brian = pytest.importorskip("brian2")

from smart_robustness.validation.figure15 import run_figure15_condition


def test_figure15_runner_requires_published_epoch_before_network_build() -> None:
    with pytest.raises(ValueError, match="1000-ms"):
        run_figure15_condition(
            top_down_current_pA=1000,
            use_paper_constrained_reference=True,
            duration_ms=100,
            brian=brian,
        )
