"""Refit the three unresolved AdEx classes using generation-2 sealed levels."""

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

BOUNDS = {
    "threshold_offset_mV": (10.0, 30.0),
    "slope_factor_mV": (0.5, 5.0),
    "reset_offset_mV": (-15.0, 5.0),
    "subthreshold_adaptation_nS": (0.0, 10.0),
    "spike_adaptation_pA": (0.0, 150.0),
    "adaptation_time_constant_ms": (40.0, 400.0),
}


def _score(
    candidates,
    candidate_levels,
    target_levels,
    training_positions,
    holdout_positions,
):
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
    return scored


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        default="configs/baselines/classic_smart_calibrated_v1.yaml",
    )
    parser.add_argument(
        "--protocol",
        default="configs/models/adex_isolated_match_protocol_generation2_v1.yaml",
    )
    parser.add_argument("--generation1", default="results/adex-isolated-fit-v3-890.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    baseline = load_frozen_classic_baseline(args.baseline)
    raw = yaml.safe_load(Path(args.protocol).read_text())
    generation1 = yaml.safe_load(Path(args.generation1).read_text())
    common = raw["common"]
    protocol = CurrentStepProtocol(
        training_currents_pA=(0.0,),
        holdout_currents_pA=(1.0,),
        pre_ms=float(common["pre_ms"]),
        step_ms=float(common["step_ms"]),
        post_ms=float(common["post_ms"]),
        dt_ms=float(common["dt_ms"]),
    )
    candidates = generate_adex_sobol_candidates(BOUNDS, count=32, include_literature=True)
    refit_names = set(raw["scope"]["refit"])
    selected_parameters = dict(generation1["selected_parameters"])

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    conventions = baseline.runtime_conventions()
    facts_by_name = {fact.canonical_name: fact for fact in first_order_population_facts()}
    results = {}
    all_promoted = True
    for name in raw["scope"]["refit"]:
        facts = facts_by_name[name]
        params = first_order_population_parameters(facts, conventions=conventions)
        if name == "layer5_excitatory_v1":
            layer5 = raw["layer5"]
            lower, upper = (float(value) for value in layer5["generation1_scan_range_mV"])
            increment = float(layer5["generation1_scan_increment_mV"])
            offsets = tuple(
                float(value) for value in np.arange(lower, upper + increment / 2, increment)
            )
            all_indices = tuple(layer5["training_scan_indices"]) + tuple(
                layer5["new_sealed_holdout_scan_indices"]
            )
            selected_offsets = tuple(offsets[int(index)] for index in all_indices)
            training_positions = tuple(range(len(layer5["training_scan_indices"])))
            holdout_positions = tuple(range(len(training_positions), len(all_indices)))
            currents = passive_normalized_currents_pA(params, selected_offsets)
            target_levels = run_current_step_protocol(
                population_factory=create_compartmental_hh_population,
                population_params=params,
                currents_pA=currents,
                protocol=protocol,
                brian=brian,
            )
            level_metadata = {"selected_offsets_mV": selected_offsets}
        else:
            silent = raw["previously_silent_classes"]
            lower, upper = (float(value) for value in silent["new_scan_offset_range_mV"])
            increment = float(silent["new_scan_increment_mV"])
            offsets = tuple(
                float(value) for value in np.arange(lower, upper + increment / 2, increment)
            )
            scan_currents = passive_normalized_currents_pA(params, offsets)
            classic_scan = run_current_step_protocol(
                population_factory=create_compartmental_hh_population,
                population_params=params,
                currents_pA=scan_currents,
                protocol=protocol,
                brian=brian,
            )
            indices = select_interleaved_current_levels(
                classic_scan, level_count=int(silent["selected_levels"])
            )
            target_levels = subset_phenotype(classic_scan, indices)
            currents = target_levels.currents_pA
            training_positions = tuple(int(value) for value in silent["training_positions"])
            holdout_positions = tuple(
                int(value) for value in silent["new_sealed_holdout_positions"]
            )
            level_metadata = {
                "scan_offsets_mV": offsets,
                "scan_rate_range_hz": [
                    min(classic_scan.firing_rates_hz),
                    max(classic_scan.firing_rates_hz),
                ],
                "selected_scan_indices": indices,
            }

        candidate_levels = run_adex_candidate_batch(
            candidates=candidates,
            population_params=params,
            currents_pA=currents,
            protocol=protocol,
            brian=brian,
        )
        scored = _score(
            candidates,
            candidate_levels,
            target_levels,
            training_positions,
            holdout_positions,
        )
        selected = select_adex_candidate(scored)
        literature = scored[-1]
        promoted = bool(
            selected.training_phenotype.finite
            and selected.holdout_phenotype.finite
            and selected.training_loss <= literature.training_loss
            and selected.holdout_loss <= literature.holdout_loss
        )
        all_promoted &= promoted
        selected_parameters[name] = selected.parameters.as_dict()
        results[name] = {
            **level_metadata,
            "currents_pA": currents,
            "training_positions": training_positions,
            "holdout_positions": holdout_positions,
            "classic_target": asdict(target_levels),
            "selected_parameters": selected.parameters.as_dict(),
            "selected_training_loss": selected.training_loss,
            "selected_holdout_loss": selected.holdout_loss,
            "literature_training_loss": literature.training_loss,
            "literature_holdout_loss": literature.holdout_loss,
            "promoted": promoted,
        }

    output = {
        "schema_version": 1,
        "status": "completed-network-blind-isolated-generation2-refit",
        "baseline_manifest": args.baseline,
        "baseline_manifest_fingerprint": baseline.manifest_fingerprint,
        "runtime_fingerprint": baseline.runtime_fingerprint,
        "protocol": args.protocol,
        "generation1": args.generation1,
        "network_outcomes_used_for_selection": False,
        "refit_classes": sorted(refit_names),
        "all_refit_classes_promoted": all_promoted,
        "rebound_and_ahp_ach_promotion_pending": True,
        "selected_parameters": selected_parameters,
        "class_results": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(output, sort_keys=False))


if __name__ == "__main__":
    main()
