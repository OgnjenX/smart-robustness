import pytest

brian = pytest.importorskip("brian2")

from smart_robustness.models.classic_hh import (
    _dual_exponential_normalizer,
    create_classic_hh_population,
)


def test_dual_exponential_is_normalized() -> None:
    import numpy as np

    rise, decay = 0.5, 5.0
    norm = _dual_exponential_normalizer(rise, decay)
    time = np.linspace(0, 20, 100_000)
    kernel = norm * (np.exp(-time / decay) - np.exp(-time / rise))
    assert kernel.max() == pytest.approx(1.0, rel=1e-6)


def test_classic_hh_population_runs() -> None:
    brian.start_scope()
    brian.defaultclock.dt = 0.05 * brian.ms
    group = create_classic_hh_population(name="smoke", size=2, params={}, brian=brian)
    monitor = brian.StateMonitor(group, "v", record=True)
    brian.run(1 * brian.ms)
    assert monitor.v.shape[1] > 0
    assert brian.numpy_.isfinite(monitor.v[:]).all()


def test_table3_cell_class_populates_geometry_derived_soma_values() -> None:
    brian.start_scope()
    group = create_classic_hh_population(
        name="layer4_table3", size=1, params={"cell_class": "layer4_excitatory"}, brian=brian
    )
    assert group.C_m[0] / brian.pfarad == pytest.approx(78.5398163)
    assert group.g_na[0] / brian.nsiemens == pytest.approx(3926.990817)
    assert group.g_k[0] / brian.nsiemens == pytest.approx(2356.19449)
    assert group.e_l[0] / brian.mV == pytest.approx(-65)
