"""Screen uniform relay input strength for endogenous Figure 7 comparison."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path

import yaml
from run_figure6_nonspecific_distal_gaba_source import _plain

from smart_robustness.protocols import MatchCondition
from smart_robustness.validation.calibration import runtime_conventions_for_candidate
from smart_robustness.validation.figure6 import Figure6LearningProtocol, run_figure6_learning
from smart_robustness.validation.figure7 import TopDownCurrentMode, run_figure7_condition

MATCH_INDICES = {38, 39, 40, 41, 42}
OVERLAP_INDICES = {40}


def verify_training(result, reference) -> None:
    actual = _plain(result)
    for field in (
        "convention_fingerprint",
        "duration_ms",
        "population_spike_indices",
        "population_spike_times_ms",
        "bottom_up",
        "top_down_wide",
        "top_down_narrow",
    ):
        if actual[field] != reference[field]:
            raise ValueError(f"fresh Figure 6 handoff differs: {field}")


def score_pair(match, mismatch) -> dict[str, bool]:
    match_active = set(match.relay_spike_indices)
    mismatch_active = set(mismatch.relay_spike_indices)
    return {
        "match_horizontal_relay_set": match_active == MATCH_INDICES,
        "mismatch_overlap_only": bool(mismatch_active)
        and mismatch_active <= OVERLAP_INDICES,
        "match_more_active_relay_cells": len(match_active) > len(mismatch_active),
        "match_more_trn_events": len(match.trn_spike_indices)
        > len(mismatch.trn_spike_indices),
        "no_reconstructed_comparator": all(
            result.comparator_transform in (None, "none")
            and result.comparator_relay_floor is None
            and result.comparator_target_count is None
            for result in (match, mismatch)
        ),
        "no_calcium_ablation": not (
            match.relay_calcium_ablated_at_stimulus
            or mismatch.relay_calcium_ablated_at_stimulus
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    registration = yaml.safe_load(Path(args.registration).read_text())
    profile = yaml.safe_load(Path(registration["profile"]).read_text())
    training_profile = yaml.safe_load(Path(registration["training_profile"]).read_text())
    training_reference = yaml.safe_load(Path(registration["training_result"]).read_text())
    base = yaml.safe_load(Path(training_profile["base_profile"]).read_text())
    conventions = replace(
        runtime_conventions_for_candidate(base["candidate"]),
        **training_profile["runtime_overrides"],
    )
    if conventions.fingerprint != registration["runtime_fingerprint"]:
        raise ValueError("registered runtime differs")

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    scales = training_profile["trn_to_relay_gaba"]["scales"]
    training = run_figure6_learning(
        conventions=conventions,
        protocol=Figure6LearningProtocol(
            monitored_populations=tuple(training_profile["monitored_populations"])
        ),
        projection_weight_scales=scales,
        brian=brian,
    )
    verify_training(training.result, training_reference["result"])

    protocol = profile["protocol"]
    outcomes = []
    for gain in protocol["uniform_relay_input_gains"]:
        conditions = {}
        for condition in (MatchCondition.MATCH, MatchCondition.MISMATCH):
            conditions[condition.value] = run_figure7_condition(
                condition=condition,
                learned_weights=training.learned_weights,
                conventions=conventions,
                persistent_projection_weight_scales=scales,
                top_down_current_pA=float(protocol["top_down_current_pA"]),
                top_down_current_mode=TopDownCurrentMode(
                    protocol["top_down_current_mode"]
                ),
                top_down_cue_lead_ms=float(protocol["top_down_cue_lead_ms"]),
                uniform_relay_input_gain=float(gain),
                duration_ms=float(protocol["duration_ms"]),
                dt_ms=float(protocol["dt_ms"]),
                record_interneuron_spikes=True,
                brian=brian,
            )
        gates = score_pair(conditions["match"], conditions["mismatch"])
        outcome = {
            "uniform_relay_input_gain": float(gain),
            "match": asdict(conditions["match"]),
            "mismatch": asdict(conditions["mismatch"]),
            "gates": gates,
            "pass": all(gates.values()),
        }
        outcomes.append(outcome)
        print(
            f"gain={gain:g} match={sorted(set(conditions['match'].relay_spike_indices))} "
            f"mismatch={sorted(set(conditions['mismatch'].relay_spike_indices))} "
            f"trn={len(conditions['match'].trn_spike_indices)}/"
            f"{len(conditions['mismatch'].trn_spike_indices)} pass={outcome['pass']}",
            flush=True,
        )

    survivors = [item for item in outcomes if item["pass"]]
    artifact = {
        "schema_version": 1,
        "id": output.stem,
        "registration": args.registration,
        "classification": profile["classification"],
        "runtime_fingerprint": conventions.fingerprint,
        "training_repeat_verified": True,
        "protocol": protocol,
        "outcomes": outcomes,
        "surviving_gains": [item["uniform_relay_input_gain"] for item in survivors],
        "selected_gain": (
            survivors[0]["uniform_relay_input_gain"] if survivors else None
        ),
        "figure7_reproduced": False,
        "baseline_promoted": False,
    }
    output.open("x").write(yaml.safe_dump(_plain(artifact), sort_keys=False))


if __name__ == "__main__":
    main()
