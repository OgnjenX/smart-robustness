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
    category_source_value: float = 70.0
    relay_input_record_id: str = "modeldb112923.external.002"
    category_input_record_id: str = "modeldb112923.external.004"
    nonspecific_input_record_id: str = "modeldb112923.external.009"
    matrix_input_record_id: str = "modeldb112923.external.010"
    expected_holding_mV: float = -12.0
    expected_relay_rate_hz: float = 40.0
    include_archived_category_pixel: bool = True

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

    def rgba_grid(self) -> np.ndarray:
        """Return the exact nonzero channels in the archived 9x9 PNG stimulus."""

        grid = np.zeros((9, 9, 4), dtype=float)
        grid.reshape(-1, 4)[list(self.active_indices), 1] = self.source_value
        if self.include_archived_category_pixel:
            grid[self.center_index // 9, self.center_index % 9, 2] = self.category_source_value
        return grid

    @property
    def center_index(self) -> int:
        return 40


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
    """Apply all nonzero channels of one recovered KInNeSS stimulus PNG."""

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
    sector.populations["layer6ii_excitatory_v1"].set_external_input(
        stimulus.category_input_record_id,
        "blue",
        0.0,
    )
    if stimulus.include_archived_category_pixel:
        sector.populations["layer6ii_excitatory_v1"].set_external_input(
            stimulus.category_input_record_id,
            "blue",
            stimulus.category_source_value,
            indices=[stimulus.center_index],
        )
    # These KInNeSS input gates use connectFromAll onto 1x1 populations, so
    # the five nonzero green source pixels contribute convergently.
    green_sources = np.full(len(stimulus.active_indices), stimulus.source_value)
    sector.populations["thalamic_nonspecific"].set_convergent_external_input(
        stimulus.nonspecific_input_record_id,
        "green",
        green_sources,
    )
    sector.populations["thalamic_matrix"].set_convergent_external_input(
        stimulus.matrix_input_record_id,
        "green",
        green_sources,
    )


def clear_bar_stimulus(sector: FirstOrderSector, stimulus: ClassicBarStimulus) -> None:
    relay = sector.populations["thalamic_relay"]
    relay.set_external_input(
        stimulus.relay_input_record_id,
        stimulus.source_channel,
        0.0,
    )
    sector.populations["layer6ii_excitatory_v1"].set_external_input(
        stimulus.category_input_record_id, "blue", 0.0
    )
    sector.populations["thalamic_nonspecific"].set_external_input(
        stimulus.nonspecific_input_record_id, "green", 0.0
    )
    sector.populations["thalamic_matrix"].set_external_input(
        stimulus.matrix_input_record_id, "green", 0.0
    )


def apply_layer6ii_somatic_cue(
    sector: FirstOrderSector,
    *,
    current_pA: float,
    cell_index: int = 40,
    brian=None,
) -> None:
    """Apply the Methods 4.9 top-down cue to one layer-6II soma."""

    if brian is None:
        import brian2 as brian
    if current_pA <= 0:
        raise ValueError("current_pA must be positive")
    if not 0 <= cell_index < 81:
        raise ValueError("cell_index must address the 9x9 layer-6II sheet")
    group = sector.populations["layer6ii_excitatory_v1"].group
    group.i_drive_soma = 0 * brian.pA
    group.i_drive_soma[cell_index] = current_pA * brian.pA


def clear_layer6ii_somatic_cue(sector: FirstOrderSector, *, brian=None) -> None:
    if brian is None:
        import brian2 as brian
    sector.populations["layer6ii_excitatory_v1"].group.i_drive_soma = 0 * brian.pA


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
