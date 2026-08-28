"""Run a frozen source candidate on the paired canonical Figure 7 protocol."""

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
from smart_robustness.validation.figure7 import (
    assess_figure7_reproduction,
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
        default="configs/calibration/figure7_trn_calcium_reversal_v1.yaml",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--duration-ms", type=float, default=100.0)
    parser.add_argument("--equilibration-ms", type=float, default=20.0)
    parser.add_argument("--cue-lead-ms", type=float, default=0.0)
    args = parser.parse_args()
    if args.duration_ms <= 45.0:
        raise ValueError("pathway diagnostics require --duration-ms greater than 45")

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    profile = yaml.safe_load(Path(args.profile).read_text())
    conventions = runtime_conventions_for_candidate(profile["candidate"])
    if conventions.fingerprint != profile["runtime_fingerprint"]:
        raise ValueError("profile runtime fingerprint does not match its candidate")
    top_down_current_pA = float(profile["candidate"]["top_down_current_pA"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    conditions = {}
    for condition in (MatchCondition.MATCH, MatchCondition.MISMATCH):
        result = run_figure7_condition(
            condition=condition,
            top_down_current_pA=top_down_current_pA,
            use_paper_constrained_reference=True,
            conventions=conventions,
            duration_ms=args.duration_ms,
            top_down_cue_lead_ms=args.cue_lead_ms,
            equilibration_ms=args.equilibration_ms,
            record_relay_diagnostics=True,
            brian=brian,
        )
        conditions[condition.value] = result
        checkpoint = {
            "schema_version": 1,
            "id": output.stem,
            "date": datetime.now(tz=UTC).date().isoformat(),
            "status": "partial-condition-checkpoint",
            "profile": args.profile,
            "candidate_fingerprint": profile["candidate_fingerprint"],
            "runtime_fingerprint": conventions.fingerprint,
            "completed_conditions": tuple(conditions),
            "conditions": conditions,
        }
        output.write_text(yaml.safe_dump(_plain(checkpoint), sort_keys=False))
        print(
            f"{condition.value}: relay={len(result.relay_spike_times_ms)} "
            f"trn={len(result.trn_spike_times_ms)} "
            f"nonspecific={result.nonspecific_rate_hz:g} Hz",
            flush=True,
        )
    assessment = assess_figure7_reproduction(
        conditions[MatchCondition.MATCH.value],
        conditions[MatchCondition.MISMATCH.value],
    )
    artifact = {
        "schema_version": 1,
        "id": Path(args.output).stem,
        "date": datetime.now(tz=UTC).date().isoformat(),
        "status": "passing-figure7-directional-contract" if assessment.reproduced else "failed-figure7-directional-contract",
        "profile": args.profile,
        "candidate_fingerprint": profile["candidate_fingerprint"],
        "runtime_fingerprint": conventions.fingerprint,
        "holdouts_consulted": True,
        "protocol": {
            "duration_ms": args.duration_ms,
            "equilibration_ms": args.equilibration_ms,
            "top_down_cue_lead_ms": args.cue_lead_ms,
            "top_down_current_pA": top_down_current_pA,
            "learned_state": "paper-constrained Figure 6c reference",
            "timing_source": (
                "Grossberg and Versace 2008 Figure 7 text: simultaneous "
                "bottom-up and top-down excitatory inputs"
            ),
        },
        "conditions": conditions,
        "assessment": assessment,
        "reproduced": assessment.reproduced,
        "next_gate": (
            "Retain the identical candidate through the complete Figure 6 prerequisite "
            "before promotion."
        ),
    }
    output.write_text(yaml.safe_dump(_plain(artifact), sort_keys=False))


if __name__ == "__main__":
    main()
