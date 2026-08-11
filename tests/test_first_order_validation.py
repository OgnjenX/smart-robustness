from __future__ import annotations

import pytest

from smart_robustness.classic_sector import FirstOrderRuntimeConventions
from smart_robustness.validation.first_order import (
    FirstOrderBarProtocol,
    FirstOrderBarResult,
    IsolatedRelayInputResult,
    assess_first_order_bar,
)


def _result(active: tuple[int, ...], inactive: int = 0) -> FirstOrderBarResult:
    return FirstOrderBarResult(
        conventions=FirstOrderRuntimeConventions(),
        protocol=FirstOrderBarProtocol(stimulus_ms=100.0),
        warmup_spikes={},
        stimulus_spikes={},
        active_relay_spikes=active,
        inactive_relay_spikes=inactive,
    )


def test_bar_assessment_accepts_five_selective_40hz_relays() -> None:
    assessment = assess_first_order_bar(_result((4, 4, 4, 4, 4)))
    assert assessment.reproduced_drive
    assert assessment.observed_rates_hz == (40.0,) * 5


def test_bar_result_document_contains_exact_profile_and_fingerprint() -> None:
    result = _result((4, 4, 4, 4, 4))
    document = result.as_document()
    assert document["convention_fingerprint"] == result.conventions.fingerprint
    assert document["conventions"]["voltage_coordinate"] == "relative_to_table3_leak"
    assert document["protocol"]["orientation"] == "horizontal"


def test_isolated_relay_rate_uses_declared_duration() -> None:
    result = IsolatedRelayInputResult(
        conventions=FirstOrderRuntimeConventions(),
        duration_ms=100.0,
        source_value=120.0,
        spike_times_ms=(10.0, 35.0, 60.0, 85.0),
        maximum_soma_voltage_mV=40.0,
        final_soma_voltage_mV=-60.0,
    )
    assert result.rate_hz == 40.0
    assert result.numerically_valid


def test_bar_assessment_rejects_rate_and_selectivity_failures() -> None:
    assessment = assess_first_order_bar(_result((0, 4, 4, 4, 4), inactive=1))
    assert not assessment.relay_rate_pass
    assert not assessment.selectivity_pass
    assert not assessment.reproduced_drive


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"warmup_ms": -1}, "warmup_ms"),
        ({"stimulus_ms": 0}, "must be positive"),
        ({"dt_ms": 0}, "must be positive"),
    ],
)
def test_bar_protocol_rejects_invalid_timing(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        FirstOrderBarProtocol(**kwargs)
