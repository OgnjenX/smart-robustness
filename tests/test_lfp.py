from __future__ import annotations

import numpy as np
import pytest

from smart_robustness.analysis.lfp import (
    current_source_density_uV_per_um2,
    extracellular_potential_uV,
)


def test_equation31_single_source_has_expected_si_conversion() -> None:
    # 1 pA at 1 um with sigma=15 mS/cm=1.5 S/m gives
    # 1/(4*pi*1.5) microvolts.
    result = extracellular_potential_uV(np.asarray([[1.0]]), np.asarray([[1.0]]))
    assert result.shape == (1, 1)
    assert result[0, 0] == pytest.approx(1.0 / (4 * np.pi * 1.5))


def test_equation31_sums_signed_sources_and_sinks_by_inverse_distance() -> None:
    currents = np.asarray([[2.0, 4.0], [-1.0, -2.0]])
    distances = np.asarray([[10.0, 20.0], [20.0, 10.0]])
    result = extracellular_potential_uV(currents, distances)
    scale = 1.0 / (4 * np.pi * 1.5)
    assert result[0] == pytest.approx(scale * np.asarray([0.15, 0.30]))
    assert result[1] == pytest.approx(scale * np.asarray([0.0, 0.0]))


def test_equation33_recovers_constant_second_difference() -> None:
    positions = np.arange(5, dtype=float)
    potential = (positions**2)[:, None] * np.asarray([[1.0, 2.0]])
    csd = current_source_density_uV_per_um2(potential, tip_spacing_um=1.0)
    assert csd.shape == (3, 2)
    assert csd == pytest.approx(np.tile([2.0, 4.0], (3, 1)))


@pytest.mark.parametrize(
    "currents,distances",
    (
        (np.arange(3), np.ones((1, 3))),
        (np.ones((2, 3)), np.ones((1, 3))),
        (np.ones((2, 3)), np.asarray([[1.0, 0.0]])),
    ),
)
def test_lfp_rejects_invalid_shapes_and_distances(currents, distances) -> None:
    with pytest.raises(ValueError):
        extracellular_potential_uV(currents, distances)


def test_csd_rejects_too_few_tips_and_invalid_spacing() -> None:
    with pytest.raises(ValueError, match="three"):
        current_source_density_uV_per_um2(np.ones((2, 10)), 1.0)
    with pytest.raises(ValueError, match="spacing"):
        current_source_density_uV_per_um2(np.ones((3, 10)), 0.0)
