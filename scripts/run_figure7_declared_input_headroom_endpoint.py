"""Test the sole source-bounded learned-feedback endpoint under declared input."""

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
    learned, applied_factor = expand_figure7_source_expectation_toward_bounds(
        training.learned_weights,
        headroom_fraction=float(profile["learned_state"]["headroom_fraction"]),
        source_index=int(profile["learned_state"]["source_index"]),
    )
    protocol = profile["protocol"]
    conditions = {}
    for condition in (MatchCondition.MATCH, MatchCondition.MISMATCH):
        conditions[condition.value] = run_figure7_condition(
            condition=condition,
            learned_weights=learned,
            conventions=conventions,
            persistent_projection_weight_scales=scales,
            top_down_current_pA=float(protocol["top_down_current_pA"]),
            top_down_current_mode=TopDownCurrentMode(protocol["top_down_current_mode"]),
            top_down_cue_lead_ms=float(protocol["top_down_cue_lead_ms"]),
            uniform_relay_input_gain=float(protocol["uniform_relay_input_gain"]),
            duration_ms=float(protocol["duration_ms"]),
            dt_ms=float(protocol["dt_ms"]),
            record_interneuron_spikes=True,
            brian=brian,
        )
    gates = score_pair(conditions["match"], conditions["mismatch"])
    artifact = {
        "schema_version": 1,
        "id": output.stem,
        "registration": args.registration,
        "classification": profile["classification"],
        "runtime_fingerprint": conventions.fingerprint,
        "training_repeat_verified": True,
        "headroom_fraction": float(profile["learned_state"]["headroom_fraction"]),
        "applied_common_weight_factor": applied_factor,
        "protocol": protocol,
        "match": asdict(conditions["match"]),
        "mismatch": asdict(conditions["mismatch"]),
        "gates": gates,
        "mechanism_screen_pass": all(gates.values()),
        "figure7_reproduced": False,
        "baseline_promoted": False,
    }
    output.open("x").write(yaml.safe_dump(_plain(artifact), sort_keys=False))
    print(
        f"factor={applied_factor:.6f} "
        f"match={sorted(set(conditions['match'].relay_spike_indices))} "
        f"mismatch={sorted(set(conditions['mismatch'].relay_spike_indices))} "
        f"trn={len(conditions['match'].trn_spike_indices)}/"
        f"{len(conditions['mismatch'].trn_spike_indices)} pass={all(gates.values())}",
        flush=True,
    )


if __name__ == "__main__":
    main()
