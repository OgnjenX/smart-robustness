"""Decompose simultaneous receptor priming into on-center and off-surround."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path

import yaml
from run_figure6_nonspecific_distal_gaba_source import _plain
from run_figure7_uniform_relay_input_gain_screen import score_pair, verify_training

from smart_robustness.protocols import MatchCondition
from smart_robustness.validation.calibration import runtime_conventions_for_candidate
from smart_robustness.validation.figure6 import Figure6LearningProtocol, run_figure6_learning
from smart_robustness.validation.figure7 import TopDownCurrentMode, run_figure7_condition


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
    persistent_scales = training_profile["trn_to_relay_gaba"]["scales"]
    training = run_figure6_learning(
        conventions=conventions,
        protocol=Figure6LearningProtocol(
            monitored_populations=tuple(training_profile["monitored_populations"])
        ),
        projection_weight_scales=persistent_scales,
        brian=brian,
    )
    verify_training(training.result, training_reference["result"])

    protocol = profile["protocol"]
    outcomes = []
    for arm_name, arm in profile["arms"].items():
        projection_ids = tuple(arm["projection_ids"])
        conditions = {}
        for condition in (MatchCondition.MATCH, MatchCondition.MISMATCH):
            conditions[condition.value] = run_figure7_condition(
                condition=condition,
                learned_weights=training.learned_weights,
                conventions=conventions,
                persistent_projection_weight_scales=persistent_scales,
                top_down_current_pA=float(protocol["top_down_current_pA"]),
                top_down_current_mode=TopDownCurrentMode(
                    protocol["top_down_current_mode"]
                ),
                top_down_cue_lead_ms=float(protocol["top_down_cue_lead_ms"]),
                uniform_relay_input_gain=float(protocol["uniform_relay_input_gain"]),
                prime_top_down_receptors_at_stimulus=True,
                top_down_receptor_prime_projection_ids=projection_ids,
                duration_ms=float(protocol["duration_ms"]),
                dt_ms=float(protocol["dt_ms"]),
                record_interneuron_spikes=True,
                brian=brian,
            )
        match = conditions["match"]
        mismatch = conditions["mismatch"]
        gates = score_pair(match, mismatch)
        gates["exact_arm_primed"] = all(
            {projection_id for projection_id, _ in result.top_down_receptor_prime_edge_counts}
            == set(projection_ids)
            for result in (match, mismatch)
        )
        outcome = {
            "arm": arm_name,
            "projection_ids": list(projection_ids),
            "match": asdict(match),
            "mismatch": asdict(mismatch),
            "gates": gates,
            "pass": all(gates.values()),
        }
        outcomes.append(outcome)
        print(
            f"arm={arm_name} match={sorted(set(match.relay_spike_indices))} "
            f"mismatch={sorted(set(mismatch.relay_spike_indices))} "
            f"trn={len(match.trn_spike_indices)}/{len(mismatch.trn_spike_indices)} "
            f"pass={outcome['pass']}",
            flush=True,
        )

    artifact = {
        "schema_version": 1,
        "id": output.stem,
        "registration": args.registration,
        "classification": profile["classification"],
        "runtime_fingerprint": conventions.fingerprint,
        "training_repeat_verified": True,
        "fixed_prime": profile["fixed_prime"],
        "protocol": protocol,
        "controls": profile["controls"],
        "outcomes": outcomes,
        "passing_arms": [item["arm"] for item in outcomes if item["pass"]],
        "figure7_reproduced": False,
        "baseline_promoted": False,
    }
    output.open("x").write(yaml.safe_dump(_plain(artifact), sort_keys=False))


if __name__ == "__main__":
    main()
