"""Screen trace-derived sub-unit Relay_INT transfer under the fixed prime endpoint."""

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
from smart_robustness.validation.figure7 import (
    TopDownCurrentMode,
    expand_figure7_source_expectation_toward_bounds,
    run_figure7_condition,
)


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
    source_control = Path(registration["source_control_result"])
    import hashlib

    if hashlib.sha256(source_control.read_bytes()).hexdigest() != registration["source_control_sha256"]:
        raise ValueError("source-control artifact hash differs")
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

    learned_spec = profile["learned_state"]
    learned, factor = expand_figure7_source_expectation_toward_bounds(
        training.learned_weights,
        headroom_fraction=float(learned_spec["headroom_fraction"]),
        source_index=int(learned_spec["source_index"]),
    )
    if factor != float(learned_spec["expected_common_weight_factor"]):
        raise ValueError("maximal selected-row factor differs from registration")

    protocol = profile["protocol"]
    projection_id = profile["pathway"]["projection_id"]
    prime_ids = tuple(profile["receptor_prime"]["projection_ids"])
    outcomes = []
    for gain in profile["pathway"]["gains_in_execution_order"]:
        conditions = {}
        for condition in (MatchCondition.MATCH, MatchCondition.MISMATCH):
            conditions[condition.value] = run_figure7_condition(
                condition=condition,
                learned_weights=learned,
                conventions=conventions,
                persistent_projection_weight_scales=persistent_scales,
                projection_weight_scales={projection_id: float(gain)},
                top_down_current_pA=float(protocol["top_down_current_pA"]),
                top_down_current_mode=TopDownCurrentMode(protocol["top_down_current_mode"]),
                top_down_cue_lead_ms=float(protocol["top_down_cue_lead_ms"]),
                uniform_relay_input_gain=float(protocol["uniform_relay_input_gain"]),
                prime_top_down_receptors_at_stimulus=True,
                top_down_receptor_prime_projection_ids=prime_ids,
                duration_ms=float(protocol["duration_ms"]),
                dt_ms=float(protocol["dt_ms"]),
                record_interneuron_spikes=True,
                brian=brian,
            )
        gates = score_pair(conditions["match"], conditions["mismatch"])
        gates["all_six_expectation_records_primed"] = all(
            result.top_down_receptor_prime_present
            and {item[0] for item in result.top_down_receptor_prime_edge_counts} == set(prime_ids)
            for result in conditions.values()
        )
        outcome = {
            "projection_002_gain": float(gain),
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
        "source_control_result": str(source_control),
        "source_control_sha256": registration["source_control_sha256"],
        "projection_id": projection_id,
        "applied_common_weight_factor": factor,
        "protocol": protocol,
        "outcomes": outcomes,
        "surviving_gains": [item["projection_002_gain"] for item in survivors],
        "selected_gain": survivors[0]["projection_002_gain"] if survivors else None,
        "figure7_reproduced": False,
        "baseline_promoted": False,
    }
    output.open("x").write(yaml.safe_dump(_plain(artifact), sort_keys=False))


if __name__ == "__main__":
    main()
