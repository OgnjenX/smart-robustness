"""Published higher-order V1-pulvinar-V2 protocol helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from smart_robustness.analysis.lfp import Figure16CorticalField, figure16_cortical_field
from smart_robustness.protocols import (
    BarOrientation,
    ClassicBarStimulus,
    apply_bar_stimulus,
    clear_bar_stimulus,
)

FIGURE16_FEEDFORWARD_PROJECTION_ID = "modeldb112923.projection.089"
FIGURE16_INTER_AREA_DELAY_MS = 10.0
FIGURE16_CORTICAL_CLASSES = (
    "layer23_excitatory",
    "layer23_inhibitory",
    "layer4_excitatory",
    "layer4_inhibitory",
    "layer5_excitatory",
    "layer6i_excitatory",
    "layer6ii_excitatory",
)


@dataclass(frozen=True, slots=True)
class Figure16Protocol:
    """Figure 16 caption facts plus explicit source-unreported runtime steps."""

    prestimulus_ms: float = 1000.0
    recording_ms: float = 1000.0
    inter_area_delay_ms: float = FIGURE16_INTER_AREA_DELAY_MS
    integration_dt_ms: float = 0.01
    recording_sample_ms: float = 1.0
    frequency_bands_hz: tuple[tuple[float, float], ...] = (
        (2.0, 4.0),
        (4.0, 8.0),
        (8.0, 12.0),
        (12.0, 20.0),
        (20.0, 100.0),
    )

    def __post_init__(self) -> None:
        if self.prestimulus_ms < 0 or self.recording_ms <= 0:
            raise ValueError("Figure 16 durations must be nonnegative and positive")
        if self.inter_area_delay_ms <= 0:
            raise ValueError("Figure 16 inter-area delay must be positive")
        if self.integration_dt_ms <= 0:
            raise ValueError("Figure 16 integration step must be positive")
        if self.recording_sample_ms <= 0 or self.recording_sample_ms > 5.0:
            raise ValueError("Figure 16 sampling must be positive and resolve 100 Hz")
        if self.recording_sample_ms < self.integration_dt_ms:
            raise ValueError("Figure 16 sampling cannot be faster than integration")
        if any(low < 0 or high <= low for low, high in self.frequency_bands_hz):
            raise ValueError("Figure 16 frequency bands must be ordered and nonnegative")


@dataclass(frozen=True)
class Figure16CandidateResult:
    """Unscored output of the exact-duration two-area candidate run."""

    protocol: Figure16Protocol
    learned_state_provenance: str
    sample_times_ms: tuple[float, ...]
    v1_field: Figure16CorticalField
    v2_field: Figure16CorticalField


def apply_figure16_inter_area_delay(
    sector,
    *,
    protocol: Figure16Protocol | None = None,
    brian=None,
) -> None:
    """Apply the caption's 10-ms V1 layer-2/3 to V2 layer-4 delay.

    The recovered ``SMART.nml`` record serializes 5 ms. This function is a
    protocol-level override and deliberately does not mutate the source
    catalog or any other cross-area pathway.
    """

    if brian is None:
        import brian2 as brian
    protocol = protocol or Figure16Protocol()
    try:
        projection = sector.projections[FIGURE16_FEEDFORWARD_PROJECTION_ID]
    except KeyError as error:
        raise ValueError("Figure 16 feedforward projection is absent from the sector") from error
    projection.delay = protocol.inter_area_delay_ms * brian.ms


def create_figure16_current_monitors(
    sector,
    *,
    protocol: Figure16Protocol | None = None,
    brian=None,
) -> dict[str, object]:
    """Create 1-ms transmembrane-current monitors for both cortical sheets."""

    if brian is None:
        import brian2 as brian
    protocol = protocol or Figure16Protocol()
    monitors: dict[str, object] = {}
    for area in ("v1", "v2"):
        for cortical_class in FIGURE16_CORTICAL_CLASSES:
            name = f"{cortical_class}_{area}"
            try:
                population = sector.populations[name]
            except KeyError as error:
                raise ValueError(f"Figure 16 cortical population {name!r} is absent") from error
            variables = tuple(
                f"i_transmembrane_paper_{compartment}"
                for compartment in population.compartments
            )
            monitors[name] = brian.StateMonitor(
                population.group,
                variables,
                record=True,
                dt=protocol.recording_sample_ms * brian.ms,
                name=f"figure16_currents_{name}",
            )
    return monitors


def figure16_cortical_field_from_monitors(
    sector,
    monitors: dict[str, object],
    *,
    area: str,
    seed: int,
    brian=None,
) -> Figure16CorticalField:
    """Convert one area's Brian current monitors to the full cortical field."""

    if brian is None:
        import brian2 as brian
    if area not in {"v1", "v2"}:
        raise ValueError("area must be 'v1' or 'v2'")
    populations = {}
    for cortical_class in FIGURE16_CORTICAL_CLASSES:
        name = f"{cortical_class}_{area}"
        try:
            population = sector.populations[name]
            monitor = monitors[name]
        except KeyError as error:
            raise ValueError(f"Figure 16 monitor for {name!r} is absent") from error
        currents = {
            compartment: getattr(monitor, f"i_transmembrane_paper_{compartment}") / brian.pA
            for compartment in population.compartments
        }
        populations[name] = (
            population.cell_spec,
            currents,
            len(population.group) // 2,
        )
    return figure16_cortical_field(populations, seed=seed)


def run_figure16_candidate(
    *,
    learned_weights: Mapping[str, tuple[float, ...] | np.ndarray] | None = None,
    use_paper_constrained_reference: bool = False,
    protocol: Figure16Protocol | None = None,
    geometry_seed: int = 16,
    conventions=None,
    brian=None,
) -> Figure16CandidateResult:
    """Run the caption's learned-stimulus prestimulus and recording epochs.

    This function intentionally returns regional tip matrices without scoring
    them: the publication does not preserve the post-learning weight arrays or
    the reduction from regional electrode tips to its plotted curves. Callers
    must supply a learned snapshot or explicitly opt into the labeled
    paper-constrained Figure 6 approximation.
    """

    if learned_weights is not None and use_paper_constrained_reference:
        raise ValueError("pass learned_weights or request the paper-constrained reference, not both")
    if learned_weights is None and not use_paper_constrained_reference:
        raise ValueError("Figure 16 requires an explicit learned expectation state")
    if isinstance(geometry_seed, bool) or not isinstance(geometry_seed, int):
        raise TypeError("geometry_seed must be an integer")
    if brian is None:
        import brian2 as brian
    from smart_robustness.classic_sector import (
        build_full_smart_network,
        figure6_runtime_conventions,
    )
    from smart_robustness.validation.figure7 import (
        apply_figure7_learned_state,
        paper_constrained_figure6_expectation,
    )

    protocol = protocol or Figure16Protocol()
    conventions = conventions or figure6_runtime_conventions()
    brian.start_scope()
    brian.defaultclock.dt = protocol.integration_dt_ms * brian.ms
    sector = build_full_smart_network(conventions=conventions, brian=brian)
    apply_figure16_inter_area_delay(sector, protocol=protocol, brian=brian)
    if use_paper_constrained_reference:
        learned_weights = paper_constrained_figure6_expectation(sector.projections)
        provenance = "paper-constrained-figure6c-reference"
    else:
        provenance = "simulated-learned-weight-snapshot"
    assert learned_weights is not None
    apply_figure7_learned_state(sector.projections, learned_weights)

    stimulus = ClassicBarStimulus(
        BarOrientation.HORIZONTAL,
        duration_ms=protocol.prestimulus_ms + protocol.recording_ms,
    )
    apply_bar_stimulus(sector, stimulus)
    monitors = create_figure16_current_monitors(sector, protocol=protocol, brian=brian)
    for monitor in monitors.values():
        monitor.active = False
    sector.network.add(*monitors.values())
    if protocol.prestimulus_ms:
        sector.network.run(protocol.prestimulus_ms * brian.ms)
    for monitor in monitors.values():
        monitor.active = True
    sector.network.run(protocol.recording_ms * brian.ms)
    clear_bar_stimulus(sector, stimulus)

    first_monitor = monitors[min(monitors)]
    sample_times_ms = tuple(float(value) for value in np.asarray(first_monitor.t / brian.ms))
    v1_field = figure16_cortical_field_from_monitors(
        sector, monitors, area="v1", seed=geometry_seed, brian=brian
    )
    v2_field = figure16_cortical_field_from_monitors(
        sector, monitors, area="v2", seed=geometry_seed + 1, brian=brian
    )
    return Figure16CandidateResult(
        protocol=protocol,
        learned_state_provenance=provenance,
        sample_times_ms=sample_times_ms,
        v1_field=v1_field,
        v2_field=v2_field,
    )
