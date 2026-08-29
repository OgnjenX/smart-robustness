"""Screen a registered behavioral TRN soma-potassium density grid."""

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
        default="configs/calibration/trn_soma_potassium_behavior_grid_v1.yaml",
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
    protocol = TrnRecruitmentProtocol(
        pre_drive_ms=float(protocol_values["pre_drive_ms"]),
        drive_ms=float(protocol_values["drive_ms"]),
        dt_ms=float(protocol_values["dt_ms"]),
        relay_ampa_gate=float(protocol_values["relay_ampa_gate"]),
        layer6ii_ampa_gate=float(protocol_values["layer6ii_ampa_gate"]),
        layer6ii_nmda_gate=float(protocol_values["layer6ii_nmda_gate"]),
    )
    outcomes = []
    for density in (float(value) for value in profile["dimension"]["grid"]):
        conventions = replace(
            base,
            trn_potassium_convention="selected_source",
            trn_soma_potassium_density_mS_cm2=density,
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
            protocol=protocol,
            brian=brian,
        )
        driven = run_trn_recruitment_condition(
            driven=True,
            conventions=conventions,
            protocol=protocol,
            brian=brian,
        )
        stage_1_pass = bool(
            control.finite
            and driven.finite
            and not control.post_drive_spike_times_ms
            and driven.post_drive_spike_times_ms
        )
        outcomes.append(
            {
                "trn_soma_potassium_density_mS_cm2": density,
                "runtime_fingerprint": conventions.fingerprint,
                "control": control,
                "driven": driven,
                "stage_1_pass": stage_1_pass,
            }
        )
        print(
            f"density={density:g}: control={control.post_drive_spike_count} "
            f"driven={driven.post_drive_spike_count} "
            f"soma_max={driven.soma_voltage_range_mV[1]:.3f} "
            f"stage_1={stage_1_pass}",
            flush=True,
        )
    survivors = [item for item in outcomes if item["stage_1_pass"]]
    driven_soma_peaks = [item["driven"].soma_voltage_range_mV[1] for item in outcomes]
    artifact = {
        "schema_version": 1,
        "id": Path(args.output).stem,
        "date": datetime.now(tz=UTC).date().isoformat(),
        "status": "stage-1-survivors-found" if survivors else "no-stage-1-survivor",
        "profile": args.profile,
        "base_candidate_fingerprint": base_profile["candidate_fingerprint"],
        "protocol": protocol,
        "stage_1_survivor_densities_mS_cm2": [
            item["trn_soma_potassium_density_mS_cm2"] for item in survivors
        ],
        "outcomes": outcomes,
        "assessment": {
            "stage_1_survivor_count": len(survivors),
            "all_controls_post_drive_quiet": all(
                not item["control"].post_drive_spike_times_ms for item in outcomes
            ),
            "all_driven_trials_finite": all(item["driven"].finite for item in outcomes),
            "best_driven_soma_peak_mV": max(driven_soma_peaks),
            "published_source_range_mS_cm2": [80.0, 100.0],
            "lower_potassium_sufficient_for_recruitment": bool(survivors),
            "advance_to_connected_match": bool(survivors),
            "interpretation": (
                "candidate-density-survives-isolated-gate"
                if survivors
                else "lower-somatic-potassium-does-not-restore-evoked-trn-output"
            ),
        },
        "next_gate": profile["next_gate"],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(_plain(artifact), sort_keys=False))


if __name__ == "__main__":
    main()
