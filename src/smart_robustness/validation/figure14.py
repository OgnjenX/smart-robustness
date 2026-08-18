"""Full-sector Figure 14 match/mismatch spectrum protocol."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from smart_robustness.analysis.figure14 import Figure14Spectrum, figure14_spectrum_from_spikes
from smart_robustness.protocols import MatchCondition

from .figure7 import Figure7ConditionResult, run_figure7_condition


@dataclass(frozen=True, slots=True)
class Figure14ConditionResult:
    condition: MatchCondition
    network_result: Figure7ConditionResult
    spectrum: Figure14Spectrum


def run_figure14_condition(
    *,
    condition: MatchCondition,
    top_down_current_pA: float,
    learned_weights: Mapping[str, tuple[float, ...] | np.ndarray] | None = None,
    use_paper_constrained_reference: bool = False,
    conventions=None,
    duration_ms: float = 1000.0,
    dt_ms: float = 0.01,
    histogram_bin_ms: float = 1.0,
    hamming_window_ms: float = 200.0,
    cpp_standalone_directory: str | Path | None = None,
    brian=None,
) -> Figure14ConditionResult:
    """Run one caption-duration condition and analyze cumulative V1 spikes."""

    if duration_ms != 1000.0:
        raise ValueError("classic Figure 14 validation requires the published 1000-ms epoch")
    network_result = run_figure7_condition(
        condition=condition,
        top_down_current_pA=top_down_current_pA,
        learned_weights=learned_weights,
        use_paper_constrained_reference=use_paper_constrained_reference,
        conventions=conventions,
        duration_ms=duration_ms,
        dt_ms=dt_ms,
        record_v1_cortical_spikes=True,
        cpp_standalone_directory=cpp_standalone_directory,
        brian=brian,
    )
    spectrum = figure14_spectrum_from_spikes(
        network_result.v1_cortical_spike_times_ms,
        duration_ms=duration_ms,
        histogram_bin_ms=histogram_bin_ms,
        hamming_window_ms=hamming_window_ms,
    )
    return Figure14ConditionResult(condition, network_result, spectrum)
