"""Full-sector Figure 15 nearby layer-4 synchrony protocol."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from smart_robustness.analysis.figure15 import (
    Figure15Assessment,
    Figure15Synchrony,
    assess_figure15_synchrony,
    figure15_layer4_synchrony,
)
from smart_robustness.protocols import MatchCondition

from .figure7 import Figure7ConditionResult, run_figure7_condition


@dataclass(frozen=True, slots=True)
class Figure15ConditionResult:
    network_result: Figure7ConditionResult
    synchrony: Figure15Synchrony
    assessment: Figure15Assessment


def run_figure15_condition(
    *,
    top_down_current_pA: float,
    learned_weights: Mapping[str, tuple[float, ...] | np.ndarray] | None = None,
    use_paper_constrained_reference: bool = False,
    conventions=None,
    first_cell_index: int = 39,
    second_cell_index: int = 40,
    duration_ms: float = 1000.0,
    dt_ms: float = 0.01,
    histogram_bin_ms: float = 1.0,
    max_lag_ms: float = 180.0,
    target_hz: float = 44.0,
    tolerance_hz: float = 5.0,
    cpp_standalone_directory: str | Path | None = None,
    brian=None,
) -> Figure15ConditionResult:
    """Run the learned-match trial and score one predeclared adjacent pair."""

    if duration_ms != 1000.0:
        raise ValueError("classic Figure 15 validation requires a 1000-ms synchrony epoch")
    network_result = run_figure7_condition(
        condition=MatchCondition.MATCH,
        top_down_current_pA=top_down_current_pA,
        learned_weights=learned_weights,
        use_paper_constrained_reference=use_paper_constrained_reference,
        conventions=conventions,
        duration_ms=duration_ms,
        dt_ms=dt_ms,
        cpp_standalone_directory=cpp_standalone_directory,
        brian=brian,
    )
    synchrony = figure15_layer4_synchrony(
        network_result.layer4_spike_indices,
        network_result.layer4_spike_times_ms,
        first_cell_index=first_cell_index,
        second_cell_index=second_cell_index,
        duration_ms=duration_ms,
        bin_ms=histogram_bin_ms,
        max_lag_ms=max_lag_ms,
    )
    return Figure15ConditionResult(
        network_result=network_result,
        synchrony=synchrony,
        assessment=assess_figure15_synchrony(
            synchrony,
            target_hz=target_hz,
            tolerance_hz=tolerance_hz,
        ),
    )
