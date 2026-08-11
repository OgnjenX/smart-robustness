from __future__ import annotations

import numpy as np
import pytest

brian = pytest.importorskip("brian2")

from smart_robustness.classic_sector import build_first_order_connected_sector


def test_connected_first_order_sector_integrates_finitely_at_rest() -> None:
    brian.start_scope()
    brian.prefs.codegen.target = "numpy"
    brian.defaultclock.dt = 0.01 * brian.ms
    sector = build_first_order_connected_sector(brian=brian)
    adaptive = {
        record_id: projection
        for record_id, projection in sector.projections.items()
        if hasattr(projection, "modifiable")
        and np.any(np.asarray(projection.modifiable[:]) > 0)
    }
    initial_weights = {
        record_id: np.asarray(projection.w[:]).copy()
        for record_id, projection in adaptive.items()
    }

    sector.network.run(0.1 * brian.ms)

    soma_mV = np.concatenate(
        [
            np.asarray(population.group.v_soma[:] / brian.mV)
            for population in sector.populations.values()
        ]
    )
    assert np.isfinite(soma_mV).all()
    assert -100 < soma_mV.min() < soma_mV.max() < 50
    assert len(adaptive) == 3
    for record_id, projection in adaptive.items():
        weight = np.asarray(projection.w[:])
        assert np.array_equal(weight, initial_weights[record_id])
        assert np.isfinite(weight).all()
        assert np.all(weight >= 0)
        assert np.all(weight <= np.asarray(projection.w_maximum[:]))
