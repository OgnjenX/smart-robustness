from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

brian = pytest.importorskip("brian2")

from smart_robustness.classic_sector import build_full_smart_network
from smart_robustness.validation.higher_order import (
    FIGURE16_CORTICAL_CLASSES,
    FIGURE16_FEEDFORWARD_PROJECTION_ID,
    Figure16Protocol,
    apply_figure16_inter_area_delay,
    create_figure16_current_monitors,
    figure16_cortical_field_from_monitors,
)


def test_figure16_protocol_encodes_caption_values() -> None:
    protocol = Figure16Protocol()
    assert protocol.prestimulus_ms == 1000
    assert protocol.recording_ms == 1000
    assert protocol.inter_area_delay_ms == 10
    assert protocol.recording_sample_ms == 1
    assert protocol.frequency_bands_hz[-1] == (20, 100)


def test_figure16_delay_override_changes_only_named_feedforward_pathway() -> None:
    brian.start_scope()
    sector = build_full_smart_network(
        projection_ids=frozenset(
            {FIGURE16_FEEDFORWARD_PROJECTION_ID, "modeldb112923.projection.081"}
        ),
        brian=brian,
    )
    feedforward = sector.projections[FIGURE16_FEEDFORWARD_PROJECTION_ID]
    control = sector.projections["modeldb112923.projection.081"]
    assert float(feedforward.delay[0] / brian.ms) == pytest.approx(5.0)
    assert float(control.delay[0] / brian.ms) == pytest.approx(1.0)

    apply_figure16_inter_area_delay(sector, brian=brian)
    monitors = create_figure16_current_monitors(sector, brian=brian)
    sector.network.add(*monitors.values())

    assert float(feedforward.delay[0] / brian.ms) == pytest.approx(10.0)
    assert float(control.delay[0] / brian.ms) == pytest.approx(1.0)
    assert set(monitors) == {
        f"{cortical_class}_{area}"
        for cortical_class in FIGURE16_CORTICAL_CLASSES
        for area in ("v1", "v2")
    }
    synthetic_monitors = {}
    for name, population in sector.populations.items():
        if name not in monitors:
            continue
        synthetic_monitors[name] = SimpleNamespace(
            **{
                f"i_transmembrane_paper_{compartment}": (
                    np.zeros((len(population.group), 2)) * brian.pA
                )
                for compartment in population.compartments
            }
        )
    field = figure16_cortical_field_from_monitors(
        sector, synthetic_monitors, area="v1", seed=16, brian=brian
    )
    assert field.potential_uV.shape == (54, 2)
    assert len(field.population_fields) == len(FIGURE16_CORTICAL_CLASSES)
    sector.network.run(0 * brian.ms)


def test_figure16_delay_override_requires_feedforward_pathway() -> None:
    brian.start_scope()
    sector = build_full_smart_network(
        projection_ids=frozenset({"modeldb112923.projection.081"}), brian=brian
    )
    with pytest.raises(ValueError, match="feedforward projection"):
        apply_figure16_inter_area_delay(sector, brian=brian)
    sector.network.run(0 * brian.ms)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"recording_ms": 0},
        {"prestimulus_ms": -1},
        {"inter_area_delay_ms": 0},
        {"frequency_bands_hz": ((8.0, 4.0),)},
        {"recording_sample_ms": 5.1},
    ),
)
def test_figure16_protocol_rejects_invalid_values(kwargs) -> None:
    with pytest.raises(ValueError):
        Figure16Protocol(**kwargs)
