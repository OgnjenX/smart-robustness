from __future__ import annotations

import numpy as np
import pytest

brian = pytest.importorskip("brian2")

from smart_robustness.classic_sector import build_first_order_connected_sector


def test_connected_first_order_sector_integrates_and_adapts_finitely() -> None:
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

    # Exercise the actual Brian2 Equation 25/28 state on the bottom-up
    # relay->layer-4 postsynaptically gated projection. A positive post-spike
    # lobe potentiates; the subsequent negative lobe depotentiates.
    brian.defaultclock.dt = 0.001 * brian.ms
    projection = sector.projections["modeldb112923.projection.035"]
    synapse_index = 0
    target_index = int(projection.j[synapse_index])
    target = sector.populations["layer4_excitatory_v1"].group
    projection.x_learning_rise[synapse_index] = 0
    projection.x_learning_fall[synapse_index] = 1
    target.v_soma[target_index] = -10 * brian.mV
    before_potentiation = float(projection.w[synapse_index])
    sector.network.run(0.001 * brian.ms)
    after_potentiation = float(projection.w[synapse_index])
    assert after_potentiation > before_potentiation

    target.v_soma[target_index] = -65 * brian.mV
    projection.last_post_spike[synapse_index] = sector.network.t - 0.1 * brian.ms
    projection.x_learning_rise[synapse_index] = 0
    projection.x_learning_fall[synapse_index] = 1
    sector.network.run(0.001 * brian.ms)
    after_depotentiation = float(projection.w[synapse_index])
    assert after_depotentiation < after_potentiation
    assert 0 <= after_depotentiation <= float(projection.w_maximum[synapse_index])
