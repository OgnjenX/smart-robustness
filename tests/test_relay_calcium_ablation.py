import numpy as np
import pytest

brian = pytest.importorskip("brian2")

from smart_robustness.protocols import MatchCondition
from smart_robustness.validation.figure7 import run_figure7_condition


@pytest.mark.parametrize("lead_ms", [0.0, 7.85])
@pytest.mark.parametrize("scope", ["none", "dendrites_only", "all_relay_compartments"])
def test_ablation_is_relay_only_and_begins_after_cue(monkeypatch, lead_ms, scope):
    runs = []

    class StimulusChecked(Exception):
        pass

    def inspect_run(network, duration, *args, **kwargs):
        relay = next(o for o in network.objects if o.name == "smart_v1_thalamic_relay")
        nonspecific = next(o for o in network.objects if o.name == "smart_v1_thalamic_nonspecific")
        is_cue = lead_ms > 0 and not runs
        # This legacy diagnostic is dendritic-only, not whole-relay ablation.
        if scope == "all_relay_compartments" and not is_cue:
            assert np.all(relay.g_ca_soma[:] == 0 * brian.nsiemens)
        else:
            assert np.all(relay.g_ca_soma[:] > 0 * brian.nsiemens)
        for compartment in ("distal_dendrite", "proximal_dendrite"):
            conductance = np.asarray(getattr(relay, f"g_ca_{compartment}") / brian.nsiemens)
            if scope != "none" and not is_cue:
                assert np.all(conductance == 0)
            else:
                assert np.all(conductance > 0)
            assert np.all(getattr(nonspecific, f"g_ca_{compartment}")[:] > 0 * brian.nsiemens)
        runs.append(float(duration / brian.ms))
        if not is_cue:
            raise StimulusChecked

    monkeypatch.setattr(brian.Network, "run", inspect_run)
    with pytest.raises(StimulusChecked):
        run_figure7_condition(
            condition=MatchCondition.MISMATCH, top_down_current_pA=800,
            use_paper_constrained_reference=True, top_down_cue_lead_ms=lead_ms,
            ablate_relay_calcium_at_stimulus=scope == "dendrites_only",
            ablate_all_relay_calcium_at_stimulus=scope == "all_relay_compartments",
        )
    assert runs == pytest.approx([lead_ms, 100.0] if lead_ms else [100.0])
