"""Published higher-order V1-pulvinar-V2 protocol helpers."""

from __future__ import annotations

from dataclasses import dataclass

FIGURE16_FEEDFORWARD_PROJECTION_ID = "modeldb112923.projection.089"
FIGURE16_INTER_AREA_DELAY_MS = 10.0


@dataclass(frozen=True, slots=True)
class Figure16Protocol:
    """Exact protocol facts stated in the Figure 16 caption."""

    prestimulus_ms: float = 1000.0
    recording_ms: float = 1000.0
    inter_area_delay_ms: float = FIGURE16_INTER_AREA_DELAY_MS
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
