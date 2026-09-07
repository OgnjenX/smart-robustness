from dataclasses import replace
from hashlib import sha256

import numpy as np
import pytest

brian = pytest.importorskip("brian2")

from smart_robustness.classic_sector import FirstOrderRuntimeConventions
from smart_robustness.protocols import MatchCondition
from smart_robustness.validation.figure7 import run_figure7_condition
from smart_robustness.validation.relay_trace import write_relay_trace
from smart_robustness.validation.nonspecific_replay import (
    NONSPECIFIC_REPLAY_MONITOR_VARIABLES,
)


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
        fingerprint="test-runtime", brian=brian, population="test_population",
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
        assert str(trace["population"]) == "test_population"
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


def test_nonspecific_replay_trace_monitor_uses_integrator_visible_gates(
    tmp_path, monkeypatch
):
    class MonitorChecked(Exception):
        pass

    def inspect_monitor(network, *args, **kwargs):
        monitor = next(
            obj
            for obj in network.objects
            if obj.name == "figure7_match_nonspecific_replay_state"
        )
        assert tuple(monitor.record_variables) == NONSPECIFIC_REPLAY_MONITOR_VARIABLES
        assert monitor.when == "thresholds"
        assert monitor.order == 0
        np.testing.assert_array_equal(monitor.record, [0])
        raise MonitorChecked

    monkeypatch.setattr(brian.Network, "run", inspect_monitor)
    with pytest.raises(MonitorChecked):
        run_figure7_condition(
            condition=MatchCondition.MATCH,
            top_down_current_pA=600,
            use_paper_constrained_reference=True,
            conventions=replace(
                FirstOrderRuntimeConventions(),
                spike_event_rule="falling_threshold_crossing",
            ),
            duration_ms=0.01,
            nonspecific_replay_trace_output=tmp_path / "nonspecific.npz",
        )


def test_nonspecific_replay_trace_never_overwrites_or_shares_a_path(tmp_path):
    existing = tmp_path / "existing.npz"
    existing.write_bytes(b"owned")
    with pytest.raises(FileExistsError):
        run_figure7_condition(
            condition=MatchCondition.MATCH,
            top_down_current_pA=600,
            use_paper_constrained_reference=True,
            duration_ms=0.01,
            nonspecific_replay_trace_output=existing,
        )
    with pytest.raises(ValueError, match="distinct paths"):
        run_figure7_condition(
            condition=MatchCondition.MATCH,
            top_down_current_pA=600,
            use_paper_constrained_reference=True,
            duration_ms=0.01,
            record_relay_diagnostics=True,
            relay_trace_output=tmp_path / "same.npz",
            nonspecific_replay_trace_output=tmp_path / "same.npz",
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


def test_interneuron_trace_mode_records_declared_input_path(tmp_path, monkeypatch):
    class MonitorChecked(Exception):
        pass

    def inspect_monitor(network, *args, **kwargs):
        spike_monitor = next(
            o for o in network.objects if o.name == "figure7_match_interneuron_spikes"
        )
        state_monitor = next(
            o for o in network.objects if o.name == "figure7_match_interneuron_state"
        )
        assert spike_monitor.source.name == "smart_v1_thalamic_interneuron"
        assert {
            "v_soma",
            "v_proximal_dendrite",
            "armed",
            "i_na_soma",
            "i_k_soma",
            "i_axial_inward_soma",
            "i_port_001",
            "i_port_002",
            "i_port_003",
            "port_001_gate",
            "port_002_gate",
            "port_003_gate",
            "i_external_mixed_input",
            "external_mixed_input_input_green",
            "external_mixed_input_input_source_count",
        } == set(state_monitor.record_variables)
        np.testing.assert_array_equal(
            state_monitor.record, [22, 31, 38, 39, 40, 41, 42, 49, 58]
        )
        raise MonitorChecked

    monkeypatch.setattr(brian.Network, "run", inspect_monitor)
    with pytest.raises(MonitorChecked):
        run_figure7_condition(
            condition=MatchCondition.MATCH,
            top_down_current_pA=800,
            use_paper_constrained_reference=True,
            conventions=replace(
                FirstOrderRuntimeConventions(),
                mixed_input_gate_convention="declared_external_input",
            ),
            interneuron_trace_output=tmp_path / "interneuron.npz",
        )


def test_interneuron_trace_requires_declared_input_and_distinct_path(tmp_path):
    with pytest.raises(ValueError, match="declared external input"):
        run_figure7_condition(
            condition=MatchCondition.MATCH,
            top_down_current_pA=800,
            use_paper_constrained_reference=True,
            interneuron_trace_output=tmp_path / "interneuron.npz",
        )
    with pytest.raises(ValueError, match="distinct paths"):
        run_figure7_condition(
            condition=MatchCondition.MATCH,
            top_down_current_pA=800,
            use_paper_constrained_reference=True,
            record_relay_diagnostics=True,
            relay_trace_output=tmp_path / "same.npz",
            interneuron_trace_output=tmp_path / "same.npz",
        )


def test_interneuron_spike_only_mode_does_not_require_declared_input(monkeypatch):
    class MonitorChecked(Exception):
        pass

    def inspect_monitor(network, *args, **kwargs):
        assert any(
            o.name == "figure7_match_interneuron_spikes" for o in network.objects
        )
        assert not any(
            o.name == "figure7_match_interneuron_state" for o in network.objects
        )
        raise MonitorChecked

    monkeypatch.setattr(brian.Network, "run", inspect_monitor)
    with pytest.raises(MonitorChecked):
        run_figure7_condition(
            condition=MatchCondition.MATCH,
            top_down_current_pA=800,
            use_paper_constrained_reference=True,
            record_interneuron_spikes=True,
        )
