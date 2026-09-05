import runpy
from dataclasses import asdict
from pathlib import Path

import pytest

from smart_robustness.protocols import MatchCondition
from smart_robustness.validation.figure7 import (
    Figure7ConditionResult,
    assess_figure7_reproduction,
)


@pytest.mark.parametrize("script", [
    "run_figure7_aligned_on_center_mismatch.py",
    "run_trn_gaba_transfer_figure7_pair.py",
    "run_figure7_top_down_current_mismatch.py",
    "run_figure7_top5_comparator_mismatch.py",
    "run_figure7_top5_nonspecific_gaba_transfer_mismatch.py",
])
@pytest.mark.parametrize("intervention", [
    {"comparator_relay_floor": 0.0},
    {"relay_calcium_ablated_at_stimulus": True},
    {"comparator_transform": "top_k_binary", "comparator_target_count": 5,
     "comparator_source_index": 40},
])
def test_scoring_loaders_cannot_erase_comparator_intervention(script, intervention):
    loader = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / script))["_scoring_result"]
    match = Figure7ConditionResult(
        condition=MatchCondition.MATCH, duration_ms=100,
        nonspecific_spike_times_ms=(10, 30, 50, 70),
        relay_spike_indices=(38, 39, 40, 41, 42),
        relay_spike_times_ms=(10, 10, 10, 10, 10),
        trn_spike_indices=(1, 2), trn_spike_times_ms=(10, 10),
        **intervention,
    )
    mismatch = Figure7ConditionResult(
        condition=MatchCondition.MISMATCH, duration_ms=100,
        nonspecific_spike_times_ms=(10, 20, 30, 40, 50, 60, 70),
        relay_spike_indices=(40,), relay_spike_times_ms=(10,),
        trn_spike_indices=(1,), trn_spike_times_ms=(10,),
    )
    loaded = loader(asdict(match))
    for key, value in intervention.items():
        assert getattr(loaded, key) == value
    assessment = assess_figure7_reproduction(loaded, loader(asdict(mismatch)))
    assert assessment.behavioral_targets_pass
    assert assessment.reconstructed_comparator_present or assessment.relay_calcium_ablation_present
    assert not assessment.reproduced
