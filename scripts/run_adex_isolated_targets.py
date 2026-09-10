"""Generate classic targets and literature-AdEx scores without network outcomes."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import yaml

from smart_robustness.baseline import load_frozen_classic_baseline
from smart_robustness.classic_sector import first_order_population_parameters
from smart_robustness.models.adex import create_somatic_adex_population
from smart_robustness.models.compartmental_hh import create_compartmental_hh_population
from smart_robustness.models.modeldb112923 import first_order_population_facts
from smart_robustness.validation.isolated_cell_matching import (
    CurrentStepProtocol,
    passive_normalized_currents_pA,
    phenotype_loss,
    run_current_step_protocol,
    subset_phenotype,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        default="configs/baselines/classic_smart_calibrated_v1.yaml",
    )
    parser.add_argument(
        "--protocol",
        default="configs/models/adex_isolated_match_protocol_v2.yaml",
    )
    parser.add_argument("--output")
    args = parser.parse_args()

    baseline = load_frozen_classic_baseline(args.baseline)
    raw_protocol = yaml.safe_load(Path(args.protocol).read_text())
    step = raw_protocol["current_steps"]
    protocol = CurrentStepProtocol(
        training_currents_pA=(0.0,),
        holdout_currents_pA=(1.0,),
        pre_ms=float(step["pre_ms"]),
        step_ms=float(step["step_ms"]),
        post_ms=float(step["post_ms"]),
        dt_ms=float(step["dt_ms"]),
    )
    train_offsets = tuple(float(value) for value in step["training_passive_voltage_offsets_mV"])
    holdout_offsets = tuple(
        float(value) for value in step["sealed_holdout_passive_voltage_offsets_mV"]
    )

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    conventions = baseline.runtime_conventions()
    results: dict[str, object] = {}
    for facts in first_order_population_facts():
        params = first_order_population_parameters(facts, conventions=conventions)
        train_currents = passive_normalized_currents_pA(params, train_offsets)
        holdout_currents = passive_normalized_currents_pA(params, holdout_offsets)
        combined_currents = train_currents + holdout_currents
        classic_combined = run_current_step_protocol(
            population_factory=create_compartmental_hh_population,
            population_params=params,
            currents_pA=combined_currents,
            protocol=protocol,
            brian=brian,
        )
        classic_train = subset_phenotype(
            classic_combined, range(len(train_currents))
        )
        classic_holdout = subset_phenotype(
            classic_combined, range(len(train_currents), len(combined_currents))
        )
        literature_combined = run_current_step_protocol(
            population_factory=create_somatic_adex_population,
            population_params=params,
            currents_pA=combined_currents,
            protocol=protocol,
            brian=brian,
        )
        literature_train = subset_phenotype(
            literature_combined, range(len(train_currents))
        )
        literature_holdout = subset_phenotype(
            literature_combined, range(len(train_currents), len(combined_currents))
        )
        results[facts.canonical_name] = {
            "training_currents_pA": train_currents,
            "holdout_currents_pA": holdout_currents,
            "classic_training_target": asdict(classic_train),
            "classic_sealed_holdout": asdict(classic_holdout),
            "literature_adex_training": asdict(literature_train),
            "literature_adex_sealed_holdout": asdict(literature_holdout),
            "literature_training_loss": phenotype_loss(classic_train, literature_train),
            "literature_holdout_loss": phenotype_loss(classic_holdout, literature_holdout),
        }

    output = yaml.safe_dump(
        {
            "schema_version": 1,
            "status": "completed-network-blind-isolated-targets",
            "baseline_manifest": args.baseline,
            "baseline_manifest_fingerprint": baseline.manifest_fingerprint,
            "runtime_fingerprint": baseline.runtime_fingerprint,
            "protocol": args.protocol,
            "network_outcomes_used": False,
            "cell_classes": results,
        },
        sort_keys=False,
    )
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output)
    else:
        print(output)


if __name__ == "__main__":
    main()
