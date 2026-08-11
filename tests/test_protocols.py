from __future__ import annotations

import numpy as np
import pytest

brian = pytest.importorskip("brian2")

from smart_robustness.classic_sector import build_first_order_intrinsic_sector
from smart_robustness.protocols import (
    BarOrientation,
    ClassicBarStimulus,
    apply_bar_stimulus,
    clear_bar_stimulus,
)


def test_recovered_bar_patterns_are_centered_five_cell_stimuli() -> None:
    horizontal = ClassicBarStimulus(BarOrientation.HORIZONTAL)
    vertical = ClassicBarStimulus(BarOrientation.VERTICAL)
    assert horizontal.active_indices == (38, 39, 40, 41, 42)
    assert vertical.active_indices == (22, 31, 40, 49, 58)
    assert np.count_nonzero(horizontal.source_grid()) == 5
    assert np.count_nonzero(vertical.source_grid()) == 5
    assert np.all(horizontal.source_grid()[4, 2:7] == 120)
    assert np.all(vertical.source_grid()[2:7, 4] == 120)


def test_bar_input_reconstructs_published_minus_12mV_drive() -> None:
    brian.start_scope()
    sector = build_first_order_intrinsic_sector(brian=brian)
    stimulus = ClassicBarStimulus(BarOrientation.HORIZONTAL)
    apply_bar_stimulus(sector, stimulus)
    relay = sector.populations["thalamic_relay"]
    port = next(
        port
        for port in relay.compiled.external_input_ports
        if port.record_id == stimulus.relay_input_record_id
    )
    source = np.asarray(getattr(relay.group, f"{port.name}_input_green")[:])
    effective_mV = np.asarray(getattr(relay.group, f"e_{port.name}_effective")[:] / brian.mV)
    assert set(np.flatnonzero(source)) == set(stimulus.active_indices)
    assert effective_mV[list(stimulus.active_indices)] == pytest.approx(-12.0)
    assert effective_mV[0] == pytest.approx(-60.0)
    clear_bar_stimulus(sector, stimulus)
    assert np.count_nonzero(getattr(relay.group, f"{port.name}_input_green")[:]) == 0
