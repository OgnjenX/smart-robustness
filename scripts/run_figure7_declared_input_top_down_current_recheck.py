"""Recheck bounded top-down current endpoints under declared Relay_INT input."""

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


def _first(values: tuple[float, ...]) -> float | None:
    return min(values) if values else None


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
    control = yaml.safe_load(Path(registration["fixed_control_result"]).read_text())
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
    feedback_delay_ms = float(protocol["learned_feedback_delay_ms"])
    outcomes = []
    for current_p_a in profile["dimension"]["endpoints_to_recheck_pA"]:
        conditions = {}
        for condition in (MatchCondition.MATCH, MatchCondition.MISMATCH):
            conditions[condition.value] = run_figure7_condition(
                condition=condition,
                learned_weights=training.learned_weights,
                conventions=conventions,
                persistent_projection_weight_scales=persistent_scales,
                top_down_current_pA=float(current_p_a),
                top_down_current_mode=TopDownCurrentMode(
                    protocol["top_down_current_mode"]
                ),
                top_down_cue_lead_ms=float(protocol["top_down_cue_lead_ms"]),
                uniform_relay_input_gain=float(protocol["uniform_relay_input_gain"]),
                duration_ms=float(protocol["duration_ms"]),
                dt_ms=float(protocol["dt_ms"]),
                record_interneuron_spikes=True,
                brian=brian,
            )
        match = conditions["match"]
        mismatch = conditions["mismatch"]
        match_category = _first(match.category_spike_times_ms)
        mismatch_category = _first(mismatch.category_spike_times_ms)
        match_relay = _first(match.relay_spike_times_ms)
        mismatch_relay = _first(mismatch.relay_spike_times_ms)
        timing_pass = all(
            category is not None
            and relay is not None
            and category + feedback_delay_ms <= relay
            for category, relay in (
                (match_category, match_relay),
                (mismatch_category, mismatch_relay),
            )
        )
        gates = score_pair(match, mismatch)
        gates["learned_feedback_arrives_before_first_relay_event"] = timing_pass
        outcome = {
            "top_down_current_pA": float(current_p_a),
            "match": asdict(match),
            "mismatch": asdict(mismatch),
            "timing_ms": {
                "match_category_first_event": match_category,
                "match_feedback_first_arrival": (
                    None if match_category is None else match_category + feedback_delay_ms
                ),
                "match_relay_first_event": match_relay,
                "mismatch_category_first_event": mismatch_category,
                "mismatch_feedback_first_arrival": (
                    None
                    if mismatch_category is None
                    else mismatch_category + feedback_delay_ms
                ),
                "mismatch_relay_first_event": mismatch_relay,
            },
            "gates": gates,
            "pass": all(gates.values()),
        }
        outcomes.append(outcome)
        print(
            f"current={current_p_a:g}pA "
            f"match={sorted(set(match.relay_spike_indices))} "
            f"mismatch={sorted(set(mismatch.relay_spike_indices))} "
            f"arrival={outcome['timing_ms']['match_feedback_first_arrival']} "
            f"relay={match_relay} pass={outcome['pass']}",
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
        "fixed_control_800_pA": {
            "source": registration["fixed_control_result"],
            "outcome": next(
                item for item in control["outcomes"]
                if item["projection_002_gain"] == 1.0
            ),
        },
        "protocol": protocol,
        "outcomes": outcomes,
        "surviving_currents_pA": [item["top_down_current_pA"] for item in survivors],
        "selected_current_pA": (
            survivors[0]["top_down_current_pA"] if survivors else None
        ),
        "figure7_reproduced": False,
        "baseline_promoted": False,
    }
    output.open("x").write(yaml.safe_dump(_plain(artifact), sort_keys=False))


if __name__ == "__main__":
    main()
