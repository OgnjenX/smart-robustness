"""Screen a registered TRN soma-sodium density by shunting-drive grid."""

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
from smart_robustness.validation.isolated_cells import (
    TrnRecruitmentProtocol,
    run_trn_recruitment_condition,
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
        default="configs/calibration/trn_soma_sodium_behavior_grid_v1.yaml",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    profile = yaml.safe_load(Path(args.profile).read_text())
    base_profile = yaml.safe_load(Path(profile["base_profile"]).read_text())
    base = runtime_conventions_for_candidate(base_profile["candidate"])
    fixed = profile["fixed_choices"]
    protocol_values = profile["stage_1_protocol"]
    base_protocol = TrnRecruitmentProtocol(
        pre_drive_ms=float(protocol_values["pre_drive_ms"]),
        drive_ms=float(protocol_values["drive_ms"]),
        dt_ms=float(protocol_values["dt_ms"]),
        relay_ampa_gate=float(protocol_values["relay_ampa_gate"]),
        layer6ii_ampa_gate=float(protocol_values["layer6ii_ampa_gate"]),
        layer6ii_nmda_gate=float(protocol_values["layer6ii_nmda_gate"]),
    )
    multipliers = [
        float(value) for value in protocol_values["drive_multipliers"]
    ]
    outcomes = []
    for density in (float(value) for value in profile["dimension"]["grid"]):
        conventions = replace(
            base,
            trn_potassium_convention="selected_source",
            trn_soma_sodium_density_mS_cm2=density,
            trn_soma_potassium_density_mS_cm2=float(
                fixed["trn_soma_potassium_density_mS_cm2"]
            ),
            trn_calcium_source_convention=str(
                fixed["trn_calcium_source_convention"]
            ),
            trn_dendritic_calcium_density_mS_cm2=float(
                fixed["trn_dendritic_calcium_density_mS_cm2"]
            ),
            trn_spike_event_proximal_blend_fraction=None,
        )
        control = run_trn_recruitment_condition(
            driven=False,
            conventions=conventions,
            protocol=base_protocol,
            brian=brian,
        )
        driven_outcomes = []
        for multiplier in multipliers:
            protocol = replace(base_protocol, drive_multiplier=multiplier)
            driven = run_trn_recruitment_condition(
                driven=True,
                conventions=conventions,
                protocol=protocol,
                brian=brian,
            )
            driven_pass = bool(driven.finite and driven.post_drive_spike_times_ms)
            driven_outcomes.append(
                {
                    "drive_multiplier": multiplier,
                    "result": driven,
                    "driven_pass": driven_pass,
                }
            )
            print(
                f"density={density:g} multiplier={multiplier:g}: "
                f"events={driven.post_drive_spike_count} "
                f"soma_max={driven.soma_voltage_range_mV[1]:.3f} "
                f"pass={driven_pass}",
                flush=True,
            )
        stage_1_pass = bool(
            control.finite
            and not control.post_drive_spike_times_ms
            and any(item["driven_pass"] for item in driven_outcomes)
        )
        outcomes.append(
            {
                "trn_soma_sodium_density_mS_cm2": density,
                "runtime_fingerprint": conventions.fingerprint,
                "control": control,
                "driven_outcomes": driven_outcomes,
                "stage_1_pass": stage_1_pass,
            }
        )
    survivors = [item for item in outcomes if item["stage_1_pass"]]
    finite_driven = [
        item["result"]
        for outcome in outcomes
        for item in outcome["driven_outcomes"]
        if item["result"].finite
    ]
    artifact = {
        "schema_version": 1,
        "id": Path(args.output).stem,
        "date": datetime.now(tz=UTC).date().isoformat(),
        "status": "stage-1-survivors-found" if survivors else "no-stage-1-survivor",
        "profile": args.profile,
        "base_candidate_fingerprint": base_profile["candidate_fingerprint"],
        "protocol": base_protocol,
        "drive_multipliers": multipliers,
        "stage_1_survivor_densities_mS_cm2": [
            item["trn_soma_sodium_density_mS_cm2"] for item in survivors
        ],
        "outcomes": outcomes,
        "assessment": {
            "stage_1_survivor_count": len(survivors),
            "all_controls_post_drive_quiet": all(
                not item["control"].post_drive_spike_times_ms for item in outcomes
            ),
            "finite_driven_trial_count": len(finite_driven),
            "total_driven_trial_count": len(outcomes) * len(multipliers),
            "best_finite_driven_soma_peak_mV": max(
                item.soma_voltage_range_mV[1] for item in finite_driven
            ),
            "sodium_density_sufficient_in_registered_assay": bool(survivors),
            "advance_to_connected_match": bool(survivors),
        },
        "next_gate": profile["next_gate"],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(_plain(artifact), sort_keys=False))


if __name__ == "__main__":
    main()
