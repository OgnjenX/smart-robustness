"""Run the preregistered top-five corticointralaminar causal ablation."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from smart_robustness.protocols import MatchCondition
from smart_robustness.validation.calibration import runtime_conventions_for_candidate
from smart_robustness.validation.figure6 import run_figure6_learning
from smart_robustness.validation.figure7 import (
    TopDownCurrentMode,
    expand_figure7_source_expectation_toward_bounds,
    run_figure7_condition,
)


def _plain(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return _plain(asdict(value))
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Enum):
        return value.value
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        default=(
            "configs/calibration/"
            "figure7_top5_corticointralaminar_ablation_v1.yaml"
        ),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    profile = yaml.safe_load(Path(args.profile).read_text())
    base_profile = yaml.safe_load(Path(profile["base_profile"]).read_text())
    base = runtime_conventions_for_candidate(base_profile["candidate"])
    detector = profile["detector"]
    conventions = replace(
        base,
        trn_spike_event_coordinate="absolute_physical",
        trn_spike_event_threshold_mV=float(detector["arm_mV"]),
        trn_spike_event_release_mV=float(detector["release_mV"]),
        trn_spike_event_proximal_blend_fraction=None,
        **profile["runtime_overrides"],
    )
    scales = {
        str(key): float(value)
        for key, value in profile["trn_to_relay_gaba"]["scales"].items()
    }
    training = run_figure6_learning(
        conventions=conventions,
        projection_weight_scales=scales,
        brian=brian,
    )
    if training.result.population_spikes["thalamic_relay"] != 20:
        raise ValueError("fresh handoff did not reproduce the Figure 6 relay train")
    learned, applied_factor = expand_figure7_source_expectation_toward_bounds(
        training.learned_weights,
        headroom_fraction=float(profile["learned_state"]["headroom_fraction"]),
        source_index=int(profile["learned_state"]["source_index"]),
    )
    protocol = profile["protocol"]
    disabled = tuple(profile["causal_ablation"]["disabled_projection_ids"])
    results = {}
    for condition_name in protocol["conditions"]:
        condition = MatchCondition(condition_name)
        results[condition] = run_figure7_condition(
            condition=condition,
            top_down_current_pA=float(protocol["top_down_current_pA"]),
            learned_weights=learned,
            conventions=conventions,
            duration_ms=float(protocol["duration_ms"]),
            dt_ms=float(protocol["dt_ms"]),
            record_relay_diagnostics=True,
            persistent_projection_weight_scales=scales,
            disabled_projection_ids=disabled,
            comparator_top_k_targets=int(profile["comparator"]["target_count"]),
            comparator_source_index=int(profile["learned_state"]["source_index"]),
            top_down_current_mode=TopDownCurrentMode(
                protocol["top_down_current_mode"]
            ),
            top_down_cue_lead_ms=float(protocol["top_down_cue_lead_ms"]),
            equilibration_ms=float(protocol["equilibration_ms"]),
            brian=brian,
        )

    match = results[MatchCondition.MATCH]
    mismatch = results[MatchCondition.MISMATCH]
    match_events = len(match.nonspecific_spike_times_ms)
    mismatch_events = len(mismatch.nonspecific_spike_times_ms)
    if mismatch_events > match_events:
        causal_classification = "latent_trn_disinhibition_has_correct_sign"
    elif mismatch_events == match_events:
        causal_classification = "trn_disinhibition_has_insufficient_rate_capacity"
    else:
        causal_classification = "remaining_pathway_has_wrong_arousal_sign"
    artifact = {
        "schema_version": 1,
        "id": Path(args.output).stem,
        "date": datetime.now(tz=UTC).date().isoformat(),
        "status": "complete-causal-diagnostic-not-candidate",
        "classification": "calibrated-reconstruction-causal-ablation",
        "profile": args.profile,
        "registration_artifact": profile["registration_artifact"],
        "base_candidate_fingerprint": base_profile["candidate_fingerprint"],
        "runtime_fingerprint": conventions.fingerprint,
        "handoff_figure6_population_spikes": training.result.population_spikes,
        "applied_common_weight_factor": applied_factor,
        "disabled_projection_ids": disabled,
        "match_result": match,
        "mismatch_result": mismatch,
        "causal_readout": {
            "match_nonspecific_events": match_events,
            "mismatch_nonspecific_events": mismatch_events,
            "event_count_delta_mismatch_minus_match": (
                mismatch_events - match_events
            ),
            "match_trn_events": len(match.trn_spike_times_ms),
            "mismatch_trn_events": len(mismatch.trn_spike_times_ms),
            "match_trn_gaba_integral_ms": match.nonspecific_trn_gaba_integral_ms,
            "mismatch_trn_gaba_integral_ms": (
                mismatch.nonspecific_trn_gaba_integral_ms
            ),
            "corticointralaminar_gates_are_zero": all(
                value == 0.0
                for value in (
                    match.nonspecific_layer6ii_ampa_peak,
                    match.nonspecific_layer6ii_nmda_peak,
                    mismatch.nonspecific_layer6ii_ampa_peak,
                    mismatch.nonspecific_layer6ii_nmda_peak,
                )
            ),
            "classification": causal_classification,
        },
        "promotable": False,
        "reproduced": False,
        "next_gate": profile["next_gate"],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(_plain(artifact), sort_keys=False))
    print(
        "ablated match relay/trn/ns="
        f"{len(match.relay_spike_times_ms)}/"
        f"{len(match.trn_spike_times_ms)}/{match_events}; "
        "mismatch="
        f"{len(mismatch.relay_spike_times_ms)}/"
        f"{len(mismatch.trn_spike_times_ms)}/{mismatch_events}; "
        f"classification={causal_classification}",
        flush=True,
    )


if __name__ == "__main__":
    main()
