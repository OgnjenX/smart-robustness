"""Fit all SMART cell classes using only preregistered isolated-cell outputs."""

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
from smart_robustness.validation.isolated_cell_matching import (
    CandidateFit,
    CurrentStepProtocol,
    generate_adex_sobol_candidates,
    passive_normalized_currents_pA,
    phenotype_loss,
    run_adex_candidate_batch,
    run_current_step_protocol,
    select_adex_candidate,
    select_interleaved_current_levels,
    subset_phenotype,
)


def _positions(values) -> tuple[int, ...]:
    return tuple(int(value) for value in values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        default="configs/baselines/classic_smart_calibrated_v1.yaml",
    )
    parser.add_argument(
        "--protocol",
        default="configs/models/adex_isolated_match_protocol_v3.yaml",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    baseline = load_frozen_classic_baseline(args.baseline)
    raw = yaml.safe_load(Path(args.protocol).read_text())
    scan_config = raw["classic_target_scan"]
    selection_config = raw["sealed_level_selection"]
    candidate_config = raw["candidate_generation"]
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
        str(name): (float(values[0]), float(values[1]))
        for name, values in candidate_config["bounds"].items()
    }
    candidates = generate_adex_sobol_candidates(
        bounds,
        count=int(candidate_config["candidates_per_cell_class"]),
        include_literature=bool(candidate_config["include_literature_candidate"]),
    )
    training_positions = _positions(selection_config["training_positions"])
    holdout_positions = _positions(selection_config["sealed_holdout_positions"])

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    conventions = baseline.runtime_conventions()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    class_results: dict[str, object] = {}
    selected_parameters: dict[str, object] = {}
    current_step_promoted = True
    for facts in first_order_population_facts():
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
        selected_currents = target_levels.currents_pA
        candidate_levels = run_adex_candidate_batch(
            candidates=candidates,
            population_params=params,
            currents_pA=selected_currents,
            protocol=protocol,
            brian=brian,
        )
        training_target = subset_phenotype(target_levels, training_positions)
        holdout_target = subset_phenotype(target_levels, holdout_positions)
        scored = []
        for candidate, phenotype in zip(candidates, candidate_levels, strict=True):
            training = subset_phenotype(phenotype, training_positions)
            holdout = subset_phenotype(phenotype, holdout_positions)
            scored.append(
                CandidateFit(
                    parameters=candidate,
                    training_loss=phenotype_loss(training_target, training),
                    holdout_loss=phenotype_loss(holdout_target, holdout),
                    training_phenotype=training,
                    holdout_phenotype=holdout,
                )
            )
        selected = select_adex_candidate(scored)
        literature = scored[-1]
        promoted = bool(
            selected.training_phenotype.finite
            and selected.holdout_phenotype.finite
            and selected.training_loss <= literature.training_loss
            and selected.holdout_loss <= literature.holdout_loss
        )
        current_step_promoted &= promoted
        selected_parameters[facts.canonical_name] = selected.parameters.as_dict()
        class_results[facts.canonical_name] = {
            "scan_offsets_mV": offsets,
            "scan_currents_pA": scan_currents,
            "classic_scan_firing_rates_hz": classic_scan.firing_rates_hz,
            "selected_scan_indices": selected_scan_indices,
            "training_positions": training_positions,
            "holdout_positions": holdout_positions,
            "classic_selected_target": asdict(target_levels),
            "selected_parameters": selected.parameters.as_dict(),
            "selected_training_loss": selected.training_loss,
            "selected_holdout_loss": selected.holdout_loss,
            "literature_training_loss": literature.training_loss,
            "literature_holdout_loss": literature.holdout_loss,
            "current_step_promoted": promoted,
        }

        checkpoint = {
            "schema_version": 1,
            "status": "running-network-blind-isolated-current-fit",
            "baseline_manifest": args.baseline,
            "baseline_manifest_fingerprint": baseline.manifest_fingerprint,
            "runtime_fingerprint": baseline.runtime_fingerprint,
            "protocol": args.protocol,
            "candidate_count_per_class": len(candidates),
            "network_outcomes_used": False,
            "completed_cell_classes": list(class_results),
            "selected_parameters": selected_parameters,
            "cell_classes": class_results,
        }
        output_path.write_text(yaml.safe_dump(checkpoint, sort_keys=False))

    output = yaml.safe_dump(
        {
            "schema_version": 1,
            "status": "completed-network-blind-isolated-current-fit",
            "baseline_manifest": args.baseline,
            "baseline_manifest_fingerprint": baseline.manifest_fingerprint,
            "runtime_fingerprint": baseline.runtime_fingerprint,
            "protocol": args.protocol,
            "candidate_count_per_class": len(candidates),
            "network_outcomes_used": False,
            "current_step_all_classes_promoted": current_step_promoted,
            "rebound_and_ahp_ach_promotion_pending": True,
            "selected_parameters": selected_parameters,
            "cell_classes": class_results,
        },
        sort_keys=False,
    )
    output_path.write_text(output)


if __name__ == "__main__":
    main()
