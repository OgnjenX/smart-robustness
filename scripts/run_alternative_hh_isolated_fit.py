"""Fit cortical alternative-HH somata without consulting SMART network outcomes."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml

from smart_robustness.baseline import load_frozen_classic_baseline
from smart_robustness.classic_sector import first_order_population_parameters
from smart_robustness.models.compartmental_hh import create_compartmental_hh_population
from smart_robustness.models.modeldb112923 import first_order_population_facts
from smart_robustness.validation.alternative_hh_isolated_cell_matching import (
    AlternativeHHFit,
    alternative_hh_phenotype_loss,
    assess_alternative_hh_match,
    generate_alternative_hh_sobol_candidates,
    run_alternative_hh_candidate_batch,
    run_selected_alternative_hh,
    select_alternative_hh_candidate,
)
from smart_robustness.validation.isolated_cell_matching import (
    CurrentStepProtocol,
    passive_normalized_currents_pA,
    run_current_step_protocol,
    select_interleaved_current_levels,
    subset_phenotype,
)


def _assessment_dict(assessment) -> dict[str, object]:
    return asdict(assessment) | {"promoted": assessment.promoted}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        default="configs/baselines/classic_smart_calibrated_v1.yaml",
    )
    parser.add_argument(
        "--protocol",
        default="configs/models/alternative_hh_isolated_match_protocol_v1.yaml",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    baseline = load_frozen_classic_baseline(args.baseline)
    raw = yaml.safe_load(Path(args.protocol).read_text())
    scan_config = raw["classic_target_scan"]
    selection_config = raw["sealed_level_selection"]
    candidate_config = raw["candidate_generation"]
    gate_config = raw["promotion_gates"]
    loss_config = raw["selection"]
    protocol = CurrentStepProtocol(
        training_currents_pA=(0.0,),
        holdout_currents_pA=(1.0,),
        pre_ms=float(scan_config["pre_ms"]),
        step_ms=float(scan_config["step_ms"]),
        post_ms=float(scan_config["post_ms"]),
        dt_ms=float(scan_config["dt_ms"]),
    )
    lower, upper = (float(value) for value in scan_config["passive_voltage_offset_range_mV"])
    increment = float(scan_config["passive_voltage_offset_increment_mV"])
    offsets = tuple(float(value) for value in np.arange(lower, upper + increment / 2, increment))
    bounds = {
        str(name): tuple(float(value) for value in values)
        for name, values in candidate_config["dimensions"].items()
    }
    candidates = generate_alternative_hh_sobol_candidates(
        bounds,
        count=int(candidate_config["candidates_per_cell_class"]),
    )
    training_positions = tuple(int(value) for value in selection_config["training_positions"])
    holdout_positions = tuple(
        int(value) for value in selection_config["sealed_holdout_positions"]
    )
    target_classes = frozenset(str(value) for value in raw["scope"]["cell_classes"])

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    conventions = baseline.runtime_conventions()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    class_results: dict[str, object] = {}
    selected_parameters: dict[str, object] = {}
    every_class_promoted = True
    for facts in first_order_population_facts():
        if facts.canonical_name not in target_classes:
            continue
        params = first_order_population_parameters(facts, conventions=conventions)
        scan_currents = passive_normalized_currents_pA(params, offsets)
        classic_scan = run_current_step_protocol(
            population_factory=create_compartmental_hh_population,
            population_params=params,
            currents_pA=scan_currents,
            protocol=protocol,
            brian=brian,
        )
        selected_scan_indices = select_interleaved_current_levels(
            classic_scan,
            level_count=int(selection_config["levels_per_cell_class"]),
        )
        target_levels = subset_phenotype(classic_scan, selected_scan_indices)
        training_target = subset_phenotype(target_levels, training_positions)
        training_candidates = run_alternative_hh_candidate_batch(
            candidates=candidates,
            population_params=params,
            currents_pA=training_target.currents_pA,
            protocol=protocol,
            brian=brian,
        )
        scored = tuple(
            AlternativeHHFit(
                parameters=parameters,
                training_loss=alternative_hh_phenotype_loss(
                    training_target,
                    phenotype,
                    resting_scale_mV=float(
                        loss_config["normalization"]["resting_voltage_mV"]
                    ),
                    firing_rate_scale_hz=float(
                        loss_config["normalization"]["firing_rate_hz"]
                    ),
                    latency_scale_ms=float(
                        loss_config["normalization"]["first_spike_latency_ms"]
                    ),
                    adaptation_scale=float(
                        loss_config["normalization"]["adaptation_ratio"]
                    ),
                    missing_penalty=float(loss_config["missing_required_feature_penalty"]),
                ),
                training_phenotype=phenotype,
            )
            for parameters, phenotype in zip(candidates, training_candidates, strict=True)
        )
        selected = select_alternative_hh_candidate(scored)
        training_assessment = assess_alternative_hh_match(
            training_target,
            selected.training_phenotype,
            resting_voltage_error_mV_max=float(
                gate_config["resting_voltage_absolute_error_mV_max"]
            ),
            spike_count_error_max=int(gate_config["spike_count_absolute_error_max"]),
            latency_error_ms_max=float(
                gate_config["first_spike_latency_absolute_error_ms_max"]
            ),
            adaptation_error_max=float(
                gate_config["adaptation_ratio_absolute_error_max"]
            ),
            adaptation_minimum_target_spikes=int(
                gate_config["adaptation_gate_applies_when_classic_has_at_least_spikes"]
            ),
        )
        holdout_target = subset_phenotype(target_levels, holdout_positions)
        selected_holdout = run_selected_alternative_hh(
            parameters=selected.parameters,
            population_params=params,
            currents_pA=holdout_target.currents_pA,
            protocol=protocol,
            brian=brian,
        )
        holdout_assessment = assess_alternative_hh_match(
            holdout_target,
            selected_holdout,
            resting_voltage_error_mV_max=float(
                gate_config["resting_voltage_absolute_error_mV_max"]
            ),
            spike_count_error_max=int(gate_config["spike_count_absolute_error_max"]),
            latency_error_ms_max=float(
                gate_config["first_spike_latency_absolute_error_ms_max"]
            ),
            adaptation_error_max=float(
                gate_config["adaptation_ratio_absolute_error_max"]
            ),
            adaptation_minimum_target_spikes=int(
                gate_config["adaptation_gate_applies_when_classic_has_at_least_spikes"]
            ),
        )
        promoted = bool(training_assessment.promoted and holdout_assessment.promoted)
        every_class_promoted &= promoted
        selected_parameters[facts.canonical_name] = selected.parameters.as_dict()
        class_results[facts.canonical_name] = {
            "scan_offsets_mV": offsets,
            "scan_currents_pA": scan_currents,
            "selected_scan_indices": selected_scan_indices,
            "training_positions": training_positions,
            "holdout_positions": holdout_positions,
            "classic_selected_target": asdict(target_levels),
            "selected_parameters": selected.parameters.as_dict(),
            "selected_training_loss": selected.training_loss,
            "selected_training_phenotype": asdict(selected.training_phenotype),
            "selected_holdout_phenotype": asdict(selected_holdout),
            "training_assessment": _assessment_dict(training_assessment),
            "holdout_assessment": _assessment_dict(holdout_assessment),
            "isolated_promoted": promoted,
        }
        checkpoint = {
            "schema_version": 1,
            "status": "running-network-blind-alternative-hh-isolated-fit",
            "baseline_manifest": args.baseline,
            "baseline_manifest_fingerprint": baseline.manifest_fingerprint,
            "runtime_fingerprint": baseline.runtime_fingerprint,
            "protocol": args.protocol,
            "candidate_count_per_class": len(candidates),
            "network_outcomes_used": False,
            "holdout_used_for_selection": False,
            "completed_cell_classes": list(class_results),
            "selected_parameters": selected_parameters,
            "cell_classes": class_results,
        }
        output_path.write_text(yaml.safe_dump(checkpoint, sort_keys=False))

    output_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "status": "completed-network-blind-alternative-hh-isolated-fit",
                "baseline_manifest": args.baseline,
                "baseline_manifest_fingerprint": baseline.manifest_fingerprint,
                "runtime_fingerprint": baseline.runtime_fingerprint,
                "protocol": args.protocol,
                "candidate_count_per_class": len(candidates),
                "network_outcomes_used": False,
                "holdout_used_for_selection": False,
                "all_cortical_classes_promoted": every_class_promoted,
                "selected_parameters": selected_parameters,
                "cell_classes": class_results,
            },
            sort_keys=False,
        )
    )


if __name__ == "__main__":
    main()
