from __future__ import annotations

import pytest

brian = pytest.importorskip("brian2")

from smart_robustness.standalone import build_and_run_cpp_standalone


def test_cpp_standalone_helper_runs_and_reloads_monitor_data(tmp_path) -> None:
    brian.device.reinit()
    brian.set_device("cpp_standalone", directory=str(tmp_path), build_on_run=False)
    brian.defaultclock.dt = 0.1 * brian.ms
    group = brian.NeuronGroup(1, "dv/dt=2/ms : 1", threshold="v >= 1", reset="v = 0")
    monitor = brian.SpikeMonitor(group)
    network = brian.Network(group, monitor)
    network.run(0.6 * brian.ms)
    build_and_run_cpp_standalone(brian, tmp_path, jobs=2)
    assert tuple(float(value) for value in monitor.t / brian.ms) == pytest.approx((0.5,))
    brian.device.reinit()
    brian.set_device("runtime")
