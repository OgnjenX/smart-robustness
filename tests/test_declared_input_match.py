import runpy
from dataclasses import replace
from pathlib import Path

import pytest

from smart_robustness.protocols import MatchCondition
from smart_robustness.validation.figure7 import Figure7ConditionResult


@pytest.fixture
def runner(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    return runpy.run_path(str(scripts / "run_declared_input_match.py"))


def test_match_gates_reject_changed_rate_comparator_and_missing_cycles(runner):
    result = Figure7ConditionResult(
        condition=MatchCondition.MATCH, duration_ms=100,
        nonspecific_spike_times_ms=(10, 30, 50, 70),
        relay_spike_indices=(38, 39, 40, 41, 42) * 3,
        relay_spike_times_ms=(10,) * 15,
        trn_spike_indices=(40,), trn_spike_times_ms=(10,),
        trn_detector_threshold_upcrossings_by_index=((40, 1),),
        trn_detector_arm_transitions_by_index=((40, 1),),
        trn_detector_release_transitions_by_index=((40, 1),),
        cue_lead_category_spike_indices=(40,), cue_lead_category_spike_times_ms=(2,),
        top_down_current_termination_time_ms=2,
    )
    score = runner["score_match"]
    assert all(score(result).values())
    for changed in (
        replace(result, nonspecific_spike_times_ms=(10, 30, 50)),
        replace(result, comparator_relay_floor=0),
        replace(result, trn_detector_arm_transitions_by_index=()),
        replace(result, relay_calcium_ablated_at_stimulus=True),
    ):
        assert not all(score(changed).values())


def test_training_handoff_rejects_changed_map(runner):
    record = {k: [] for k in ("convention_fingerprint", "duration_ms", "population_spike_indices",
                              "population_spike_times_ms", "bottom_up", "top_down_wide", "top_down_narrow")}
    runner["verify_training"](record, record)
    changed = dict(record, top_down_wide=[1])
    with pytest.raises(ValueError, match="top_down_wide"):
        runner["verify_training"](changed, record)
