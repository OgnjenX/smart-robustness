"""Published stimulation protocols for the classic SMART baseline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from .classic_sector import FirstOrderSector
from .partition import population_parts


def _set_mixed_interneuron_image(sector, stimulus, *, clear=False):
    population = sector.populations.get("thalamic_interneuron")
    if population is None:
        return
    record_id = "modeldb112923.projection.042"
    if not any(port.record_id == record_id for _, part in population_parts(population)
               for port in part.compiled.external_input_ports):
        return
    population.set_external_input(record_id, stimulus.source_channel, 0.0)
    if not clear:
        population.set_external_input(record_id, stimulus.source_channel,
                                      stimulus.source_value, indices=list(stimulus.active_indices))


class BarOrientation(StrEnum):
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"


class MatchCondition(StrEnum):
    MATCH = "match"
    MISMATCH = "mismatch"


class ConvergentExternalSourceScope(StrEnum):
    """Which image locations contribute to an all-to-one external-input gate."""

    NONZERO_PIXELS = "nonzero_pixels"
    FULL_INPUT_GRID = "full_input_grid"
    PERSISTENT_FULL_INPUT_GRID = "persistent_full_input_grid"


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
            # Figure 7 supplies its learned category through the explicit
            # layer-6II somatic-current cue below.  The blue pixel belongs to
            # the recovered Figure 6 training PNG and must not be applied as
            # a second, undocumented category input here.
            include_archived_category_pixel=False,
        )


def initialize_convergent_external_input(
    sector: FirstOrderSector,
    stimulus: ClassicBarStimulus,
    *,
    convergent_source_scope: ConvergentExternalSourceScope | str = (
        ConvergentExternalSourceScope.NONZERO_PIXELS
    ),
) -> None:
    """Install the selected blank input topology before any integration.

    Historical epoch-only modes retain their initialized one-source state.
    The persistent alternative connects every pixel, including black pixels.
    """
    scope = ConvergentExternalSourceScope(convergent_source_scope)
    if scope is ConvergentExternalSourceScope.PERSISTENT_FULL_INPUT_GRID:
        for name, record_id in (
            ("thalamic_nonspecific", stimulus.nonspecific_input_record_id),
            ("thalamic_matrix", stimulus.matrix_input_record_id),
        ):
            sector.populations[name].set_convergent_external_input(
                record_id, "green", np.zeros(stimulus.source_grid().size)
            )


def apply_bar_stimulus(
    sector: FirstOrderSector,
    stimulus: ClassicBarStimulus,
    *,
    apply_relay_input: bool = True,
    relay_input_gains: np.ndarray | None = None,
    convergent_source_scope: ConvergentExternalSourceScope | str = (
        ConvergentExternalSourceScope.NONZERO_PIXELS
    ),
) -> None:
    """Apply stimulus values using the requested convergent source scope.

    ``full_input_grid`` does not initialize prestimulus connectivity. Clearing
    this legacy protocol restores one source, so this option alone must not be
    interpreted as a persistent 81-location input topology.
    For ``persistent_full_input_grid``, callers must initialize connectivity
    before any integration and pass the same scope when clearing the stimulus.
    """

    source_scope = ConvergentExternalSourceScope(convergent_source_scope)
    _set_mixed_interneuron_image(sector, stimulus)

    gains = None
    if relay_input_gains is not None:
        gains = np.asarray(relay_input_gains, dtype=float)
        if gains.shape != (81,):
            raise ValueError("relay_input_gains must have shape (81,)")
        if not np.all(np.isfinite(gains)) or np.any((gains < 0) | (gains > 1)):
            raise ValueError("relay_input_gains must be finite and lie in [0, 1]")

    relay = sector.populations["thalamic_relay"]
    relay.set_external_input(stimulus.relay_input_record_id, stimulus.source_channel, 0.0)
    if apply_relay_input and gains is None:
        relay.set_external_input(
            stimulus.relay_input_record_id,
            stimulus.source_channel,
            stimulus.source_value,
            indices=list(stimulus.active_indices),
        )
    elif apply_relay_input:
        for index in stimulus.active_indices:
            relay.set_external_input(
                stimulus.relay_input_record_id,
                stimulus.source_channel,
                stimulus.source_value * gains[index],
                indices=[index],
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
    green_sources = (
        stimulus.source_grid().ravel()
        if source_scope in {
            ConvergentExternalSourceScope.FULL_INPUT_GRID,
            ConvergentExternalSourceScope.PERSISTENT_FULL_INPUT_GRID,
        }
        else np.full(len(stimulus.active_indices), stimulus.source_value)
    )
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


def clear_bar_stimulus(
    sector: FirstOrderSector,
    stimulus: ClassicBarStimulus,
    *,
    convergent_source_scope: ConvergentExternalSourceScope | str = (
        ConvergentExternalSourceScope.NONZERO_PIXELS
    ),
) -> None:
    scope = ConvergentExternalSourceScope(convergent_source_scope)
    _set_mixed_interneuron_image(sector, stimulus, clear=True)
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
    initialize_convergent_external_input(
        sector, stimulus, convergent_source_scope=scope
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
    apply_relay_input: bool = True,
    relay_input_gains: np.ndarray | None = None,
    convergent_source_scope: ConvergentExternalSourceScope | str = (
        ConvergentExternalSourceScope.NONZERO_PIXELS
    ),
    brian=None,
) -> None:
    """Apply Figure 7 bottom-up input and the Methods 4.9 layer-6II current cue."""

    if brian is None:
        import brian2 as brian

    apply_bar_stimulus(
        sector,
        cue.bottom_up_stimulus,
        apply_relay_input=apply_relay_input,
        relay_input_gains=relay_input_gains,
        convergent_source_scope=convergent_source_scope,
    )
    layer6ii = sector.populations[cue.top_down_population].group
    layer6ii.i_drive_soma = 0 * brian.pA
    layer6ii.i_drive_soma[cue.top_down_cell_index] = cue.top_down_current_pA * brian.pA


def clear_match_mismatch_cue(
    sector: FirstOrderSector,
    cue: ClassicMatchMismatchCue,
    *,
    convergent_source_scope: ConvergentExternalSourceScope | str = (
        ConvergentExternalSourceScope.NONZERO_PIXELS
    ),
    brian=None,
) -> None:
    if brian is None:
        import brian2 as brian

    clear_bar_stimulus(
        sector, cue.bottom_up_stimulus,
        convergent_source_scope=convergent_source_scope,
    )
    sector.populations[cue.top_down_population].group.i_drive_soma = 0 * brian.pA
