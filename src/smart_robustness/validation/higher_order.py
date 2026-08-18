"""Published higher-order V1-pulvinar-V2 protocol helpers."""

from __future__ import annotations

from dataclasses import dataclass

from smart_robustness.analysis.lfp import Figure16CorticalField, figure16_cortical_field

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
    """Exact protocol facts stated in the Figure 16 caption."""

    prestimulus_ms: float = 1000.0
    recording_ms: float = 1000.0
    inter_area_delay_ms: float = FIGURE16_INTER_AREA_DELAY_MS
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
        if self.recording_sample_ms <= 0 or self.recording_sample_ms > 5.0:
            raise ValueError("Figure 16 sampling must be positive and resolve 100 Hz")
        if any(low < 0 or high <= low for low, high in self.frequency_bands_hz):
            raise ValueError("Figure 16 frequency bands must be ordered and nonnegative")


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
