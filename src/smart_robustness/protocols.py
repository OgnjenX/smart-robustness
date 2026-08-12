"""Published stimulation protocols for the classic SMART baseline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from .classic_sector import FirstOrderSector


class BarOrientation(StrEnum):
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"


class MatchCondition(StrEnum):
    MATCH = "match"
    MISMATCH = "mismatch"


@dataclass(frozen=True, slots=True)
class ClassicBarStimulus:
    orientation: BarOrientation
    duration_ms: float = 100.0
    source_channel: str = "green"
    source_value: float = 120.0
    relay_input_record_id: str = "modeldb112923.external.002"
    expected_holding_mV: float = -12.0
    expected_relay_rate_hz: float = 40.0

    def __post_init__(self) -> None:
        if self.duration_ms <= 0:
            raise ValueError("stimulus duration must be positive")

    @property
    def active_indices(self) -> tuple[int, ...]:
        if self.orientation is BarOrientation.HORIZONTAL:
            return tuple(4 * 9 + x for x in range(2, 7))
        return tuple(y * 9 + 4 for y in range(2, 7))

    def source_grid(self) -> np.ndarray:
        grid = np.zeros((9, 9), dtype=float)
        grid.flat[list(self.active_indices)] = self.source_value
        return grid


@dataclass(frozen=True, slots=True)
class ClassicMatchMismatchCue:
    """Published Figure 7 cue applied to an already learned horizontal category.

    Grossberg and Versace specify one stimulated layer-6II category cell but do
    not report the somatic current amplitude. Requiring it here prevents the
    validation harness from silently fitting an undocumented value.
    """

    condition: MatchCondition
    top_down_current_pA: float
    duration_ms: float = 100.0
    top_down_cell_index: int = 40
    top_down_population: str = "layer6ii_excitatory_v1"

    def __post_init__(self) -> None:
        if self.duration_ms <= 0:
            raise ValueError("duration_ms must be positive")
        if self.top_down_current_pA <= 0:
            raise ValueError("top_down_current_pA must be positive")
        if not 0 <= self.top_down_cell_index < 81:
            raise ValueError("top_down_cell_index must address the 9x9 sheet")

    @property
    def bottom_up_orientation(self) -> BarOrientation:
        if self.condition is MatchCondition.MATCH:
            return BarOrientation.HORIZONTAL
        return BarOrientation.VERTICAL

    @property
    def top_down_expectation(self) -> BarOrientation:
        return BarOrientation.HORIZONTAL

    @property
    def bottom_up_stimulus(self) -> ClassicBarStimulus:
        return ClassicBarStimulus(
            self.bottom_up_orientation,
            duration_ms=self.duration_ms,
        )


def apply_bar_stimulus(sector: FirstOrderSector, stimulus: ClassicBarStimulus) -> None:
    """Apply one recovered 9x9 KInNeSS bar to the relay BU input port."""

    relay = sector.populations["thalamic_relay"]
    relay.set_external_input(
        stimulus.relay_input_record_id,
        stimulus.source_channel,
        0.0,
    )
    relay.set_external_input(
        stimulus.relay_input_record_id,
        stimulus.source_channel,
        stimulus.source_value,
        indices=list(stimulus.active_indices),
    )


def clear_bar_stimulus(sector: FirstOrderSector, stimulus: ClassicBarStimulus) -> None:
    relay = sector.populations["thalamic_relay"]
    relay.set_external_input(
        stimulus.relay_input_record_id,
        stimulus.source_channel,
        0.0,
    )


def apply_match_mismatch_cue(
    sector: FirstOrderSector,
    cue: ClassicMatchMismatchCue,
    *,
    brian=None,
) -> None:
    """Apply Figure 7 bottom-up input and the Methods 4.9 layer-6II current cue."""

    if brian is None:
        import brian2 as brian

    apply_bar_stimulus(sector, cue.bottom_up_stimulus)
    layer6ii = sector.populations[cue.top_down_population].group
    layer6ii.i_drive_soma = 0 * brian.pA
    layer6ii.i_drive_soma[cue.top_down_cell_index] = cue.top_down_current_pA * brian.pA


def clear_match_mismatch_cue(
    sector: FirstOrderSector,
    cue: ClassicMatchMismatchCue,
    *,
    brian=None,
) -> None:
    if brian is None:
        import brian2 as brian

    clear_bar_stimulus(sector, cue.bottom_up_stimulus)
    sector.populations[cue.top_down_population].group.i_drive_soma = 0 * brian.pA
