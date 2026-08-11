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
