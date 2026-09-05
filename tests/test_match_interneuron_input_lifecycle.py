from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

brian = pytest.importorskip("brian2")

from smart_robustness.protocols import MatchCondition
from smart_robustness.validation.calibration import runtime_conventions_for_candidate
from smart_robustness.validation.figure7 import run_figure7_condition


@pytest.mark.parametrize("condition", [MatchCondition.MATCH, MatchCondition.MISMATCH])
@pytest.mark.parametrize("lead_ms", [0.0, 7.85])
def test_actual_figure7_runner_delivers_interneuron_image_only_at_sensory_onset(monkeypatch, condition, lead_ms):
    root = Path(__file__).resolve().parents[1]
    profile = yaml.safe_load((root / "configs/calibration/figure6_declared_interneuron_input_v1.yaml").read_text())
    base = yaml.safe_load((root / profile["base_profile"]).read_text())
    conventions = replace(runtime_conventions_for_candidate(base["candidate"]), **profile["runtime_overrides"])
    runs = []

    class Checked(Exception):
        pass

    def inspect_run(network, duration, *args, **kwargs):
        group = next(o for o in network.objects if o.name == "smart_v1_thalamic_interneuron")
        cue = lead_ms > 0 and not runs
        expected = np.zeros(81)
        if not cue:
            indices = [38, 39, 40, 41, 42] if condition is MatchCondition.MATCH else [22, 31, 40, 49, 58]
            expected[indices] = 120
        np.testing.assert_array_equal(group.external_mixed_input_input_green[:], expected)
        assert np.all(group.external_mixed_input_input_source_count[:] == 1)
        assert np.all(group.g_external_mixed_input[:] > 0 * brian.nsiemens)
        runs.append(float(duration / brian.ms))
        if not cue:
            raise Checked

    monkeypatch.setattr(brian.Network, "run", inspect_run)
    with pytest.raises(Checked):
        run_figure7_condition(condition=condition, conventions=conventions,
                              use_paper_constrained_reference=True, top_down_current_pA=800,
                              top_down_cue_lead_ms=lead_ms, brian=brian)
    assert runs == pytest.approx([lead_ms, 100] if lead_ms else [100])
