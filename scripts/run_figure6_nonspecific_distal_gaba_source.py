"""Test the discrete supplement tuple for projection 049 at Figure 6."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from smart_robustness.validation.calibration import runtime_conventions_for_candidate
from smart_robustness.validation.figure6 import (
    HORIZONTAL_INDICES,
    Figure6LearningProtocol,
    assess_figure6_cortical_recruitment,
    assess_figure6_top_down_timing,
    run_figure6_learning,
)

FIGURE6_MONITORED_POPULATIONS = (
    "thalamic_relay",
    "layer4_excitatory_v1",
    "layer23_excitatory_v1",
    "layer5_excitatory_v1",
    "layer6i_excitatory_v1",
    "layer6ii_excitatory_v1",
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
        default="configs/calibration/figure6_nonspecific_distal_gaba_supplement_v1.yaml",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    profile = yaml.safe_load(Path(args.profile).read_text())
    base_profile = yaml.safe_load(Path(profile["base_profile"]).read_text())
    base = runtime_conventions_for_candidate(base_profile["candidate"])
    conventions = replace(base, **profile["runtime_overrides"])
    scales = {
        str(key): float(value)
        for key, value in profile["trn_to_relay_gaba"]["scales"].items()
    }
    run = run_figure6_learning(
        conventions=conventions,
        protocol=Figure6LearningProtocol(
            monitored_populations=FIGURE6_MONITORED_POPULATIONS
        ),
        projection_weight_scales=scales,
        brian=brian,
    )
    result = run.result
    relay_indices = result.population_spike_indices["thalamic_relay"]
    relay_counts = {index: relay_indices.count(index) for index in HORIZONTAL_INDICES}
    recruitment = assess_figure6_cortical_recruitment(result)
    timing = assess_figure6_top_down_timing(result)
    gates = {
        "relay_active_indices": set(relay_indices) == set(HORIZONTAL_INDICES),
        "relay_events_per_active_index": set(relay_counts.values()) == {4},
        "relay_events": len(relay_indices) == 20,
        "cortical_chain_complete": recruitment.feedforward_chain_complete,
        "causal_learning_pair": timing.causal_pair_in_learning_window,
        "top_down_horizontal_contrast": (
            result.top_down_combined.horizontal_orientation_contrast > 0
        ),
    }
    passed = all(gates.values())
    artifact = {
        "schema_version": 1,
        "id": Path(args.output).stem,
        "date": datetime.now(tz=UTC).date().isoformat(),
        "status": "figure6-passed" if passed else "figure6-failed",
        "profile": args.profile,
        "registration_artifact": profile["registration_artifact"],
        "base_candidate_fingerprint": base_profile["candidate_fingerprint"],
        "runtime_fingerprint": conventions.fingerprint,
        "source_alternative": profile["source_conflict"],
        "result": result,
        "relay_event_counts_by_index": relay_counts,
        "cortical_recruitment": recruitment,
        "top_down_timing": timing,
        "gates": gates,
        "pass": passed,
        "assessment": {
            "advance_to_figure7_match": passed,
            "figure7_holdouts_remain_locked": not passed,
        },
        "next_gate": profile["next_gate"],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(_plain(artifact), sort_keys=False))
    print(
        f"relay={len(relay_indices)} "
        f"layer4={result.population_spikes['layer4_excitatory_v1']} "
        f"layer6ii={result.population_spikes['layer6ii_excitatory_v1']} "
        f"contrast={result.top_down_combined.horizontal_orientation_contrast:.6g} "
        f"pass={passed}",
        flush=True,
    )


if __name__ == "__main__":
    main()
