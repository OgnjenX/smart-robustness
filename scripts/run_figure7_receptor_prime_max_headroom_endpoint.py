"""Cross receptor simultaneity with maximal selected-row learned headroom."""

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
    FIGURE7_TOP_DOWN_EXPECTATION_PROJECTION_IDS,
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
    learned, factor = expand_figure7_source_expectation_toward_bounds(
        training.learned_weights,
        headroom_fraction=float(profile["learned_state"]["headroom_fraction"]),
        source_index=int(profile["learned_state"]["source_index"]),
    )
    expected_factor = float(profile["learned_state"]["expected_common_weight_factor"])
    if factor != expected_factor:
        raise ValueError("maximal selected-row factor differs from registration")

    protocol = profile["protocol"]
    projection_ids = tuple(profile["receptor_prime"]["projection_ids"])
    conditions = {}
    for condition in (MatchCondition.MATCH, MatchCondition.MISMATCH):
        conditions[condition.value] = run_figure7_condition(
            condition=condition,
            learned_weights=learned,
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
    gates["all_six_expectation_records_primed"] = all(
        {projection_id for projection_id, _ in result.top_down_receptor_prime_edge_counts}
        == set(FIGURE7_TOP_DOWN_EXPECTATION_PROJECTION_IDS)
        for result in (match, mismatch)
    )
    passed = all(gates.values())
    artifact = {
        "schema_version": 1,
        "id": output.stem,
        "registration": args.registration,
        "classification": profile["classification"],
        "runtime_fingerprint": conventions.fingerprint,
        "training_repeat_verified": True,
        "headroom_fraction": float(profile["learned_state"]["headroom_fraction"]),
        "applied_common_weight_factor": factor,
        "receptor_prime": profile["receptor_prime"],
        "protocol": protocol,
        "match": asdict(match),
        "mismatch": asdict(mismatch),
        "gates": gates,
        "mechanism_screen_pass": passed,
        "figure7_reproduced": False,
        "baseline_promoted": False,
    }
    print(
        f"factor={factor:.6f} match={sorted(set(match.relay_spike_indices))} "
        f"mismatch={sorted(set(mismatch.relay_spike_indices))} "
        f"trn={len(match.trn_spike_indices)}/{len(mismatch.trn_spike_indices)} "
        f"pass={passed}",
        flush=True,
    )
    output.open("x").write(yaml.safe_dump(_plain(artifact), sort_keys=False))


if __name__ == "__main__":
    main()
