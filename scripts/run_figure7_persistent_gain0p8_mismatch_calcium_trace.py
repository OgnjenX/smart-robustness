"""Replay the failed fixed mismatch with lossless relay calcium diagnostics."""

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


IDENTITY_FIELDS = (
    "nonspecific_spike_times_ms",
    "layer4_spike_indices",
    "layer4_spike_times_ms",
    "relay_spike_indices",
    "relay_spike_times_ms",
    "trn_spike_indices",
    "trn_spike_times_ms",
    "category_spike_indices",
    "category_spike_times_ms",
    "equilibration_nonspecific_spike_times_ms",
    "equilibration_layer4_spike_indices",
    "equilibration_layer4_spike_times_ms",
    "equilibration_relay_spike_indices",
    "equilibration_relay_spike_times_ms",
    "equilibration_trn_spike_indices",
    "equilibration_trn_spike_times_ms",
    "equilibration_category_spike_indices",
    "equilibration_category_spike_times_ms",
    "cue_lead_category_spike_indices",
    "cue_lead_category_spike_times_ms",
    "cue_lead_nonspecific_spike_times_ms",
    "cue_lead_trn_spike_indices",
    "cue_lead_trn_spike_times_ms",
    "cue_lead_relay_spike_indices",
    "cue_lead_relay_spike_times_ms",
    "v1_cortical_spike_times_ms",
    "v2_layer4_spike_indices",
    "v2_layer4_spike_times_ms",
    "v2_relay_spike_indices",
    "v2_relay_spike_times_ms",
    "interneuron_spike_indices",
    "interneuron_spike_times_ms",
    "cue_lead_interneuron_spike_indices",
    "cue_lead_interneuron_spike_times_ms",
    "top_down_current_termination_time_ms",
    "top_down_receptor_prime_edge_counts",
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
    pair_profile = yaml.safe_load(Path(profile["base_profile"]).read_text())
    training_profile = yaml.safe_load(Path(registration["training_profile"]).read_text())
    training_reference = yaml.safe_load(Path(registration["training_result"]).read_text())
    reference_pair = yaml.safe_load(Path(registration["reference_pair"]).read_text())
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
        headroom_fraction=float(pair_profile["learned_state"]["headroom_fraction"]),
        source_index=int(pair_profile["learned_state"]["source_index"]),
    )
    expected_factor = float(pair_profile["learned_state"]["expected_common_weight_factor"])
    if factor != expected_factor:
        raise ValueError("maximal selected-row factor differs from registration")

    protocol = pair_profile["protocol"]
    projection_ids = tuple(pair_profile["receptor_prime"]["projection_ids"])
    trace_path = Path(profile["trace"]["output"])
    mismatch = run_figure7_condition(
        condition=MatchCondition.MISMATCH,
        learned_weights=learned,
        conventions=conventions,
        persistent_projection_weight_scales=persistent_scales,
        top_down_current_pA=float(protocol["top_down_current_pA"]),
        top_down_current_mode=TopDownCurrentMode(protocol["top_down_current_mode"]),
        top_down_cue_lead_ms=float(protocol["top_down_cue_lead_ms"]),
        uniform_relay_input_gain=float(protocol["uniform_relay_input_gain"]),
        prime_top_down_receptors_at_stimulus=True,
        top_down_receptor_prime_projection_ids=projection_ids,
        duration_ms=float(protocol["duration_ms"]),
        dt_ms=float(protocol["dt_ms"]),
        record_relay_diagnostics=True,
        record_interneuron_spikes=True,
        relay_trace_output=trace_path,
        brian=brian,
    )

    observed = _plain(asdict(mismatch))
    expected = reference_pair["mismatch"]
    mismatches = {
        field: {"expected": expected[field], "observed": observed[field]}
        for field in IDENTITY_FIELDS
        if observed[field] != expected[field]
    }
    if mismatches:
        raise ValueError(f"event identity failed: {sorted(mismatches)}")

    variables = set()
    import numpy as np

    with np.load(trace_path, allow_pickle=False) as trace:
        variables = {str(value) for value in trace["variable_names"].tolist()}
        recorded_indices = [int(value) for value in trace["cell_indices"].tolist()]
        sample_count = len(trace["time_ms"])
    missing_variables = sorted(set(profile["trace"]["required_variables"]) - variables)
    if missing_variables:
        raise ValueError(f"trace variables missing: {missing_variables}")
    if recorded_indices != profile["trace"]["recorded_relay_indices"]:
        raise ValueError("recorded relay indices differ from registration")

    artifact = {
        "schema_version": 1,
        "id": output.stem,
        "registration": args.registration,
        "classification": profile["classification"],
        "runtime_fingerprint": conventions.fingerprint,
        "training_repeat_verified": True,
        "applied_common_weight_factor": factor,
        "reference_pair_sha256": registration["reference_pair_sha256"],
        "event_identity_fields": list(IDENTITY_FIELDS),
        "event_train_identity_verified": True,
        "trace": {
            "path": mismatch.relay_trace_path,
            "sha256": mismatch.relay_trace_sha256,
            "sample_count": sample_count,
            "recorded_relay_indices": recorded_indices,
            "variable_names": sorted(variables),
            "required_variables_present": True,
        },
        "analysis_contract": profile["analysis_contract"],
        "mismatch": observed,
        "parameter_selected": False,
        "artifact_468_rescored": False,
        "original_smart_reproduced": False,
        "baseline_promoted": False,
    }
    output.open("x").write(yaml.safe_dump(_plain(artifact), sort_keys=False))
    print(
        f"identity=True trace={mismatch.relay_trace_sha256} "
        f"samples={sample_count} relay_events={len(mismatch.relay_spike_indices)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
