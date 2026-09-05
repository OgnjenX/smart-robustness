"""Fixed comparator-free match with verified, unexpanded Figure 6 weights."""

import argparse
from dataclasses import asdict, replace
from pathlib import Path

import yaml
from run_figure6_nonspecific_distal_gaba_source import _plain

from smart_robustness.protocols import MatchCondition
from smart_robustness.validation.calibration import runtime_conventions_for_candidate
from smart_robustness.validation.figure6 import Figure6LearningProtocol, run_figure6_learning
from smart_robustness.validation.figure7 import TopDownCurrentMode, run_figure7_condition


def verify_training(result, reference):
    actual = _plain(result)
    for field in ("convention_fingerprint", "duration_ms", "population_spike_indices",
                  "population_spike_times_ms", "bottom_up", "top_down_wide", "top_down_narrow"):
        if actual[field] != reference[field]:
            raise ValueError(f"fresh Figure 6 handoff differs: {field}")


def score_match(result):
    expected = {38, 39, 40, 41, 42}
    events = [t for i, t in zip(result.cue_lead_category_spike_indices,
                               result.cue_lead_category_spike_times_ms, strict=True) if i == 40]
    counts = {i: result.relay_spike_indices.count(i) for i in expected}
    up = dict(result.trn_detector_threshold_upcrossings_by_index)
    arms = dict(result.trn_detector_arm_transitions_by_index)
    releases = dict(result.trn_detector_release_transitions_by_index)
    cycles = bool(arms) and any(result.trn_spike_indices.count(i) for i in arms) and all(
        result.trn_spike_indices.count(i) == up.get(i) == arms[i] == releases.get(i) for i in arms)
    return {
        "one_selected_category_event_during_lead": len(events) == 1,
        "no_off_source_category_events_during_lead": all(i == 40 for i in result.cue_lead_category_spike_indices),
        "current_terminated_on_selected_event": len(events) == 1 and result.top_down_current_termination_time_ms == events[0],
        "no_relay_events_during_lead": not result.cue_lead_relay_spike_times_ms,
        "relay_active_indices": set(result.relay_spike_indices) == expected,
        "minimum_relay_events_per_active_index": all(n >= 3 for n in counts.values()),
        "trn_events": bool(result.trn_spike_times_ms),
        "sampled_trn_events_have_fresh_cycles": cycles,
        "figure7_target_duration": result.duration_ms == 100,
        "nonspecific_events": len(result.nonspecific_spike_times_ms) == 4,
        "nonspecific_40_hz": result.nonspecific_rate_hz == 40,
        "no_reconstructed_comparator": result.comparator_transform in (None, "none") and result.comparator_relay_floor is None and result.comparator_target_count is None,
        "no_calcium_ablation": not result.relay_calcium_ablated_at_stimulus,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    registration = yaml.safe_load(Path(args.registration).read_text())
    profile = yaml.safe_load(Path(registration["training_profile"]).read_text())
    reference = yaml.safe_load(Path(registration["training_result"]).read_text())
    if not reference["pass"] or not all(reference["gates"].values()):
        raise ValueError("Figure 6 prerequisites failed")
    base = yaml.safe_load(Path(profile["base_profile"]).read_text())
    conventions = replace(runtime_conventions_for_candidate(base["candidate"]), **profile["runtime_overrides"])
    if conventions.fingerprint != registration["runtime_fingerprint"]:
        raise ValueError("registered runtime differs")
    import brian2 as brian
    brian.prefs.codegen.target = "numpy"
    scales = profile["trn_to_relay_gaba"]["scales"]
    training = run_figure6_learning(
        conventions=conventions,
        protocol=Figure6LearningProtocol(monitored_populations=tuple(profile["monitored_populations"])),
        projection_weight_scales=scales, brian=brian)
    verify_training(training.result, reference["result"])
    result = run_figure7_condition(
        condition=MatchCondition.MATCH, learned_weights=training.learned_weights,
        conventions=conventions, persistent_projection_weight_scales=scales,
        top_down_current_pA=800, top_down_current_mode=TopDownCurrentMode.UNTIL_CUED_CELL_FIRST_EVENT,
        top_down_cue_lead_ms=7.85, duration_ms=100, dt_ms=0.01, equilibration_ms=0,
        record_relay_diagnostics=True, brian=brian)
    gates = score_match(result)
    artifact = {"schema_version": 1, "registration": args.registration,
                "runtime_fingerprint": conventions.fingerprint,
                "runtime_conventions": asdict(conventions),
                "training_repeat_verified": True, "training_result": training.result,
                "weight_handoff": "actual_figure6_weights_no_expansion",
                "applied_common_weight_factor": 1.0,
                "match_result": result, "gates": gates, "match_prerequisites_pass": all(gates.values()),
                "original_smart_reproduced": False, "baseline_promoted": False}
    with output.open("x") as stream:
        yaml.safe_dump(_plain(artifact), stream, sort_keys=False)
    print(f"match relay={len(result.relay_spike_indices)} trn={len(result.trn_spike_indices)} nonspecific={len(result.nonspecific_spike_times_ms)} pass={all(gates.values())}", flush=True)


if __name__ == "__main__":
    main()
