from __future__ import annotations

import pytest

brian = pytest.importorskip("brian2")

from smart_robustness.protocols import MatchCondition
from smart_robustness.validation.figure14 import run_figure14_condition


def test_figure14_runner_requires_published_epoch_before_network_build() -> None:
    with pytest.raises(ValueError, match="1000-ms"):
        run_figure14_condition(
            condition=MatchCondition.MATCH,
            top_down_current_pA=600,
            use_paper_constrained_reference=True,
            duration_ms=100,
            brian=brian,
        )
