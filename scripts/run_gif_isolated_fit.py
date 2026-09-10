"""Fit GIF parameters using only registered isolated-cell targets and seeds."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml

from smart_robustness.baseline import load_frozen_classic_baseline
from smart_robustness.classic_sector import first_order_population_parameters
from smart_robustness.models.compartmental_hh import create_compartmental_hh_population
from smart_robustness.models.gif_parameters import literature_gif_parameters
from smart_robustness.models.modeldb112923 import first_order_population_facts
from smart_robustness.validation.gif_isolated_cell_matching import (
    GIFCandidateFit,
    generate_gif_sobol_candidates,
    run_gif_candidate_batch,
    select_gif_candidate,
)
from smart_robustness.validation.isolated_cell_matching import (
    CurrentStepProtocol,
    passive_normalized_currents_pA,
    phenotype_loss,
    run_current_step_protocol,
    select_interleaved_current_levels,
    subset_phenotype,
)


def _positions(values) -> tuple[int, ...]:
    return tuple(int(value) for value in values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline", default="configs/baselines/classic_smart_calibrated_v1.yaml"
    )
    parser.add_argument(
        "--protocol", default="configs/models/gif_isolated_match_protocol_v1.yaml"
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    baseline = load_frozen_classic_baseline(args.baseline)
    raw = yaml.safe_load(Path(args.protocol).read_text())
    scan_config = raw["classic_target_scan"]
    level_config = raw["sealed_level_selection"]
    stochastic = raw["stochastic_repetitions"]
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
        for name, values in candidate_config["fitted_parameters"].items()
    }
    training_positions = _positions(level_config["training_positions"])
    holdout_positions = _positions(level_config["sealed_current_holdout_positions"])
    training_seeds = _positions(stochastic["training_seeds"])
    seed_holdout = _positions(stochastic["sealed_seed_holdout"])

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    conventions = baseline.runtime_conventions()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    class_results: dict[str, object] = {}
    selected_parameters: dict[str, object] = {}
    all_promoted = True

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
            classic_scan, level_count=int(level_config["levels_per_cell_class"])
        )
        target_levels = subset_phenotype(classic_scan, selected_scan_indices)
        training_target = subset_phenotype(target_levels, training_positions)
        holdout_target = subset_phenotype(target_levels, holdout_positions)
        base = literature_gif_parameters(facts.canonical_name)
        candidates = generate_gif_sobol_candidates(
            bounds,
            base=base,
            count=int(candidate_config["candidates_per_cell_class"]),
            include_literature=bool(candidate_config["include_literature_candidate"]),
        )

        training_losses: list[list[float]] = [[] for _ in candidates]
        for seed in training_seeds:
            phenotypes = run_gif_candidate_batch(
                candidates=candidates,
                population_params=params,
                currents_pA=target_levels.currents_pA,
                protocol=protocol,
                seed=seed,
                brian=brian,
            )
            for index, phenotype in enumerate(phenotypes):
                training_losses[index].append(
                    phenotype_loss(
                        training_target, subset_phenotype(phenotype, training_positions)
                    )
                )

        provisional = [
            GIFCandidateFit(
                parameters=candidate,
                mean_training_loss=float(np.mean(losses)),
                mean_seed_holdout_loss=float("nan"),
                training_losses=tuple(losses),
                seed_holdout_losses=(),
            )
            for candidate, losses in zip(candidates, training_losses, strict=True)
        ]
        selected_index = provisional.index(select_gif_candidate(provisional))
        literature_index = candidates.index(base)
        holdout_candidates = (candidates[selected_index], candidates[literature_index])
        holdout_losses: list[list[float]] = [[], []]
        holdout_phenotypes: list[dict[str, object]] = []
        for seed in seed_holdout:
            phenotypes = run_gif_candidate_batch(
                candidates=holdout_candidates,
                population_params=params,
                currents_pA=target_levels.currents_pA,
                protocol=protocol,
                seed=seed,
                brian=brian,
            )
            for index, phenotype in enumerate(phenotypes):
                holdout_losses[index].append(
                    phenotype_loss(
                        holdout_target, subset_phenotype(phenotype, holdout_positions)
                    )
                )
            holdout_phenotypes.append(
                {
                    "seed": seed,
                    "selected": asdict(phenotypes[0]),
                    "literature": asdict(phenotypes[1]),
                }
            )

        selected_training = float(np.mean(training_losses[selected_index]))
        literature_training = float(np.mean(training_losses[literature_index]))
        selected_holdout = float(np.mean(holdout_losses[0]))
        literature_holdout = float(np.mean(holdout_losses[1]))
        promoted = bool(
            np.isfinite(selected_training)
            and np.isfinite(selected_holdout)
            and selected_training <= literature_training
            and selected_holdout <= literature_holdout
        )
        all_promoted &= promoted
        selected_parameters[facts.canonical_name] = candidates[selected_index].as_dict()
        class_results[facts.canonical_name] = {
            "selected_scan_indices": selected_scan_indices,
            "selected_currents_pA": target_levels.currents_pA,
            "classic_target": asdict(target_levels),
            "selected_parameters": candidates[selected_index].as_dict(),
            "selected_training_losses": training_losses[selected_index],
            "literature_training_losses": training_losses[literature_index],
            "selected_mean_training_loss": selected_training,
            "literature_mean_training_loss": literature_training,
            "selected_seed_holdout_losses": holdout_losses[0],
            "literature_seed_holdout_losses": holdout_losses[1],
            "selected_mean_seed_holdout_loss": selected_holdout,
            "literature_mean_seed_holdout_loss": literature_holdout,
            "holdout_phenotypes": holdout_phenotypes,
            "promoted": promoted,
        }
        checkpoint = {
            "schema_version": 1,
            "status": "running-network-blind-stochastic-gif-fit",
            "baseline_manifest": args.baseline,
            "baseline_manifest_fingerprint": baseline.manifest_fingerprint,
            "runtime_fingerprint": baseline.runtime_fingerprint,
            "protocol": args.protocol,
            "network_outcomes_used": False,
            "completed_cell_classes": list(class_results),
            "selected_parameters": selected_parameters,
            "cell_classes": class_results,
        }
        output_path.write_text(yaml.safe_dump(checkpoint, sort_keys=False))

    output_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "status": "completed-network-blind-stochastic-gif-fit",
                "baseline_manifest": args.baseline,
                "baseline_manifest_fingerprint": baseline.manifest_fingerprint,
                "runtime_fingerprint": baseline.runtime_fingerprint,
                "protocol": args.protocol,
                "network_outcomes_used": False,
                "all_classes_promoted": all_promoted,
                "selected_parameters": selected_parameters,
                "cell_classes": class_results,
            },
            sort_keys=False,
        )
    )


if __name__ == "__main__":
    main()
