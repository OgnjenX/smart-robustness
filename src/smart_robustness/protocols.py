"""Published stimulation protocols for the classic SMART baseline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from .classic_sector import FirstOrderSector


class BarOrientation(StrEnum):
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"


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
