"""Evaluate a frozen calibration profile on the Figure 6/7 training targets."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from smart_robustness.protocols import MatchCondition
from smart_robustness.validation.calibration import runtime_conventions_for_candidate
from smart_robustness.validation.figure6 import (
    Figure6LearningProtocol,
    assess_figure6_cortical_recruitment,
    assess_figure6_top_down_timing,
    run_figure6_learning,
)
from smart_robustness.validation.figure7 import (
    assess_figure7_reproduction,
    run_figure7_condition,
)

MONITORED_FIGURE6_POPULATIONS = (
    "thalamic_relay",
    "layer4_excitatory_v1",
    "layer23_inhibitory_v1",
    "layer23_excitatory_v1",
    "layer5_excitatory_v1",
    "layer6i_excitatory_v1",
    "layer6ii_excitatory_v1",
    "trn",
    "thalamic_nonspecific",
)


def _plain(value: Any) -> Any:
    """Convert dataclass/NumPy values to stable YAML primitives."""

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
    if isinstance(value, Path):
        return str(value)
    return value


def _map_summary(summary) -> dict[str, Any]:
    delta = np.asarray(summary.delta, dtype=float)
    return {
        "projection_id": summary.projection_id,
        "map_role": summary.map_role,
        "minimum_delta": float(np.min(delta)),
        "maximum_delta": float(np.max(delta)),
        "horizontal_orientation_contrast": summary.horizontal_orientation_contrast,
        "horizontal_retention_advantage": summary.horizontal_retention_advantage,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile", default="configs/calibration/trn_stage_a_survivor_v1.yaml"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--run-figure7",
        action="store_true",
        help="Run Figure 7 even if the Figure 6 learned-state gates fail.",
    )
    args = parser.parse_args()

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    profile = yaml.safe_load(Path(args.profile).read_text())
    conventions = runtime_conventions_for_candidate(profile["candidate"])
    protocol = Figure6LearningProtocol(
        monitored_populations=MONITORED_FIGURE6_POPULATIONS
    )
    learning = run_figure6_learning(
        conventions=conventions, protocol=protocol, brian=brian
    )
    recruitment = assess_figure6_cortical_recruitment(learning.result)
    timing = assess_figure6_top_down_timing(learning.result)
    first_spikes = {
        name: min(times) if times else None
        for name, times in (learning.result.population_spike_times_ms or {}).items()
    }
    figure6 = {
        "protocol": asdict(protocol),
        "population_spikes": learning.result.population_spikes,
        "first_spike_ms": first_spikes,
        "recruitment": asdict(recruitment),
        "feedforward_chain_complete": recruitment.feedforward_chain_complete,
        "top_down_timing": asdict(timing),
        "causal_pair_in_learning_window": timing.causal_pair_in_learning_window,
        "bottom_up": _map_summary(learning.result.bottom_up),
        "top_down_wide": _map_summary(learning.result.top_down_wide),
        "top_down_narrow": _map_summary(learning.result.top_down_narrow),
        "bottom_up_oriented": learning.result.bottom_up_oriented,
        "top_down_oriented": learning.result.top_down_oriented,
    }
    figure6_reproduced = bool(
        recruitment.feedforward_chain_complete
        and timing.causal_pair_in_learning_window
        and learning.result.bottom_up_oriented
        and learning.result.top_down_oriented
    )

    artifact: dict[str, Any] = {
        "schema_version": 1,
        "id": Path(args.output).stem,
        "date": datetime.now(tz=UTC).date().isoformat(),
        "profile": str(args.profile),
        "candidate_fingerprint": profile["candidate_fingerprint"],
        "runtime_fingerprint": conventions.fingerprint,
        "holdouts_consulted": False,
        "status": "passing-figure6-training-target" if figure6_reproduced else "failed-figure6-training-target",
        "figure6": figure6,
        "assessment": {
            "figure6_reproduced": figure6_reproduced,
            "figure7_eligible": figure6_reproduced,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(_plain(artifact), sort_keys=False))
    print(
        "Figure 6: "
        f"chain={recruitment.feedforward_chain_complete} "
        f"bottom_up={learning.result.bottom_up_oriented} "
        f"top_down={learning.result.top_down_oriented}",
        flush=True,
    )

    if args.run_figure7:
        top_down_current = float(profile["candidate"]["top_down_current_pA"])
        conditions = {
            condition.value: run_figure7_condition(
                condition=condition,
                top_down_current_pA=top_down_current,
                learned_weights=learning.learned_weights,
                conventions=conventions,
                record_relay_diagnostics=True,
                brian=brian,
            )
            for condition in (MatchCondition.MATCH, MatchCondition.MISMATCH)
        }
        assessment = assess_figure7_reproduction(
            conditions[MatchCondition.MATCH.value],
            conditions[MatchCondition.MISMATCH.value],
        )
        artifact["figure7"] = {
            "conditions": conditions,
            "assessment": assessment,
            "reproduced": assessment.reproduced,
        }
        output.write_text(yaml.safe_dump(_plain(artifact), sort_keys=False))
        print(
            "Figure 7: "
            f"match={conditions['match'].nonspecific_rate_hz:g} Hz "
            f"mismatch={conditions['mismatch'].nonspecific_rate_hz:g} Hz "
            f"reproduced={assessment.reproduced}",
            flush=True,
        )


if __name__ == "__main__":
    main()
