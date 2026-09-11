import pytest

from smart_robustness.validation.thalamic_matching import (
    ConductanceReboundCondition,
    ConductanceReboundProtocol,
    classic_rebound_valid,
)


def _condition(**overrides):
    values = {
        "inhibition_scale": 1.0,
        "baseline_spike_count": 4,
        "inhibition_spike_count": 1,
        "release_spike_count": 5,
        "early_release_spike_count": 3,
        "first_release_spike_latency_ms": 3.0,
        "first_release_isi_ms": 5.0,
        "baseline_mean_voltage_mV": -60.0,
        "inhibition_mean_voltage_mV": -69.0,
        "release_t_current_inward_charge_pA_ms": 20.0,
        "baseline_t_current_inward_charge_pA_ms": 10.0,
        "finite": True,
    }
    return ConductanceReboundCondition(**(values | overrides))


def test_classic_conductance_rebound_gate_requires_suppression_burst_and_t_current() -> None:
    protocol = ConductanceReboundProtocol()
    assert classic_rebound_valid(_condition(), protocol=protocol)
    assert not classic_rebound_valid(
        _condition(inhibition_spike_count=3), protocol=protocol
    )
    assert not classic_rebound_valid(
        _condition(early_release_spike_count=1), protocol=protocol
    )
    assert not classic_rebound_valid(
        _condition(release_t_current_inward_charge_pA_ms=5.0), protocol=protocol
    )


def test_conductance_rebound_protocol_rejects_invalid_windows() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        ConductanceReboundProtocol(early_release_ms=101.0)
