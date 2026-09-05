from hashlib import sha256

import numpy as np
import pytest

brian = pytest.importorskip("brian2")

from smart_robustness.protocols import MatchCondition
from smart_robustness.validation.figure7 import run_figure7_condition
from smart_robustness.validation.relay_trace import write_relay_trace


def test_trace_preserves_units_cell_order_and_pre_stimulus_samples(tmp_path):
    brian.start_scope()
    brian.prefs.codegen.target = "numpy"
    group = brian.NeuronGroup(3, "v_soma : volt\ni_ca_distal_dendrite : amp\nh_ca_distal_dendrite : 1")
    group.v_soma = [-60, -50, -40] * brian.mV
    group.i_ca_distal_dendrite = [1, 2, 3] * brian.pA
    group.h_ca_distal_dendrite = [0.1, 0.2, 0.3]
    monitor = brian.StateMonitor(
        group, ("v_soma", "i_ca_distal_dendrite", "h_ca_distal_dendrite"),
        record=[2, 0], dt=0.1 * brian.ms,
    )
    brian.Network(group, monitor).run(0.3 * brian.ms)
    path = tmp_path / "trace.npz"
    fingerprint = write_relay_trace(
        monitor, path, stimulus_start_ms=0.1, condition="mismatch",
        fingerprint="test-runtime", brian=brian,
    )
    assert fingerprint == sha256(path.read_bytes()).hexdigest()
    with np.load(path, allow_pickle=False) as trace:
        np.testing.assert_allclose(trace["time_ms"], [-0.1, 0, 0.1], atol=1e-12)
        np.testing.assert_array_equal(trace["cell_indices"], [2, 0])
        np.testing.assert_allclose(trace["v_soma"][:, 0], [-40, -60])
        np.testing.assert_allclose(trace["i_ca_distal_dendrite"][:, 0], [3, 1])
        np.testing.assert_allclose(trace["h_ca_distal_dendrite"][:, 0], [0.3, 0.1])
        assert trace["variable_units"].tolist() == ["mV", "pA", "dimensionless"]
        assert str(trace["monitor_when"]) == "start"
    with pytest.raises(FileExistsError):
        write_relay_trace(monitor, path, stimulus_start_ms=0.1,
                          condition="mismatch", fingerprint="test-runtime", brian=brian)
    assert fingerprint == sha256(path.read_bytes()).hexdigest()


def test_trace_requires_diagnostics_before_network_construction(tmp_path):
    with pytest.raises(ValueError, match="requires relay diagnostics"):
        run_figure7_condition(
            condition=MatchCondition.MATCH, top_down_current_pA=600,
            use_paper_constrained_reference=True,
            relay_trace_output=tmp_path / "trace.npz",
        )


def test_trace_mode_records_calcium_gates_in_actual_relay_monitor(tmp_path, monkeypatch):
    class MonitorChecked(Exception):
        pass

    def inspect_monitor(network, *args, **kwargs):
        monitor = next(o for o in network.objects if o.name == "figure7_match_relay_pathway_state")
        assert {
            "m_ca_distal_dendrite", "h_ca_distal_dendrite",
            "m_ca_proximal_dendrite", "h_ca_proximal_dendrite",
            "m_ca_soma", "h_ca_soma", "i_ca_soma",
        } <= set(monitor.record_variables)
        raise MonitorChecked

    monkeypatch.setattr(brian.Network, "run", inspect_monitor)
    with pytest.raises(MonitorChecked):
        run_figure7_condition(
            condition=MatchCondition.MATCH, top_down_current_pA=600,
            use_paper_constrained_reference=True, record_relay_diagnostics=True,
            relay_trace_output=tmp_path / "trace.npz",
        )
