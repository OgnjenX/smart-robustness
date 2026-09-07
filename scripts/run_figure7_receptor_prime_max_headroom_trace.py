"""Trace the rejected maximal-headroom receptor-prime endpoint."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path

import yaml
from run_figure6_nonspecific_distal_gaba_source import _plain
from run_figure7_uniform_relay_input_gain_screen import verify_training

from smart_robustness.protocols import MatchCondition
from smart_robustness.validation.calibration import runtime_conventions_for_candidate
from smart_robustness.validation.figure6 import Figure6LearningProtocol, run_figure6_learning
from smart_robustness.validation.figure7 import (
    TopDownCurrentMode,
    expand_figure7_source_expectation_toward_bounds,
    run_figure7_condition,
)


def _prefix(result, duration_ms: float) -> dict[str, list[float] | list[int]]:
    relay_pairs = [
        (int(index), float(time_ms))
        for index, time_ms in zip(
            result.relay_spike_indices, result.relay_spike_times_ms, strict=True
        )
        if time_ms <= duration_ms
    ]
    return {
        "relay_spike_indices": [index for index, _ in relay_pairs],
        "relay_spike_times_ms": [time_ms for _, time_ms in relay_pairs],
        "trn_spike_indices": [
            int(index)
            for index, time_ms in zip(
                result.trn_spike_indices, result.trn_spike_times_ms, strict=True
            )
            if time_ms <= duration_ms
        ],
        "trn_spike_times_ms": [
            float(time_ms) for time_ms in result.trn_spike_times_ms if time_ms <= duration_ms
        ],
        "nonspecific_spike_times_ms": [
            float(time_ms)
            for time_ms in result.nonspecific_spike_times_ms
            if time_ms <= duration_ms
        ],
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
    trace_profile = yaml.safe_load(Path(registration["profile"]).read_text())
    candidate = yaml.safe_load(Path(registration["candidate_profile"]).read_text())
    candidate_result = yaml.safe_load(Path(registration["candidate_result"]).read_text())
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
        headroom_fraction=float(candidate["learned_state"]["headroom_fraction"]),
        source_index=int(candidate["learned_state"]["source_index"]),
    )
    if factor != float(candidate["learned_state"]["expected_common_weight_factor"]):
        raise ValueError("maximal selected-row factor differs from registration")

    protocol = candidate["protocol"]
    projection_ids = tuple(candidate["receptor_prime"]["projection_ids"])
    duration_ms = float(trace_profile["protocol"]["duration_ms"])
    sample_times_ms = tuple(trace_profile["readout"]["fixed_sample_times_ms"])
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
            duration_ms=duration_ms,
            dt_ms=float(protocol["dt_ms"]),
            record_relay_diagnostics=True,
            relay_fixed_sample_times_ms=sample_times_ms,
            record_interneuron_spikes=True,
            brian=brian,
        )

    prefix_window_ms = float(trace_profile["protocol"]["prefix_identity_window_ms"])
    prefixes = {
        name: _prefix(result, prefix_window_ms) for name, result in conditions.items()
    }
    expected_prefixes = {
        name: {
            key: candidate_result[name][key]
            for key in (
                "relay_spike_indices",
                "relay_spike_times_ms",
                "trn_spike_indices",
                "trn_spike_times_ms",
                "nonspecific_spike_times_ms",
            )
        }
        for name in ("match", "mismatch")
    }
    prefix_identity = prefixes == expected_prefixes
    if not prefix_identity:
        raise ValueError("diagnostic run does not reproduce Artifact 455 prefix")

    artifact = {
        "schema_version": 1,
        "id": output.stem,
        "registration": args.registration,
        "classification": trace_profile["classification"],
        "runtime_fingerprint": conventions.fingerprint,
        "training_repeat_verified": True,
        "applied_common_weight_factor": factor,
        "sample_times_ms": list(sample_times_ms),
        "prefix_identity_window_ms": prefix_window_ms,
        "prefix_identity_verified": prefix_identity,
        "match": asdict(conditions["match"]),
        "mismatch": asdict(conditions["mismatch"]),
        "figure7_reproduced": False,
        "baseline_promoted": False,
    }
    output.open("x").write(yaml.safe_dump(_plain(artifact), sort_keys=False))
    print(
        f"prefix={prefix_identity} match={sorted(set(conditions['match'].relay_spike_indices))} "
        f"mismatch={sorted(set(conditions['mismatch'].relay_spike_indices))}",
        flush=True,
    )


if __name__ == "__main__":
    main()
