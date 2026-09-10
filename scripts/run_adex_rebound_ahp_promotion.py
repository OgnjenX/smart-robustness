"""Run final network-blind rebound and AHP/ACh gates for matched AdEx."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import yaml

from smart_robustness.baseline import load_frozen_classic_baseline
from smart_robustness.classic_sector import first_order_population_parameters
from smart_robustness.models.adex import make_somatic_adex_factory
from smart_robustness.models.adex_parameters import AdExParameters
from smart_robustness.models.compartmental_hh import create_compartmental_hh_population
from smart_robustness.models.modeldb112923 import first_order_population_facts
from smart_robustness.validation.isolated_cell_matching import (
    ReboundProtocol,
    assess_rebound_match,
    run_rebound_protocol,
)
from smart_robustness.validation.isolated_cells import (
    Figure19Protocol,
    assess_figure19_kernel,
    run_figure19_kernel_condition,
)


def _figure19(factory, protocol, brian):
    traces = {
        "control": run_figure19_kernel_condition(
            spike_count=0,
            acetylcholine=False,
            protocol=protocol,
            population_factory=factory,
            brian=brian,
        ),
        "one": run_figure19_kernel_condition(
            spike_count=1,
            acetylcholine=False,
            protocol=protocol,
            population_factory=factory,
            brian=brian,
        ),
        "two": run_figure19_kernel_condition(
            spike_count=2,
            acetylcholine=False,
            protocol=protocol,
            population_factory=factory,
            brian=brian,
        ),
        "two_ach": run_figure19_kernel_condition(
            spike_count=2,
            acetylcholine=True,
            protocol=protocol,
            population_factory=factory,
            brian=brian,
        ),
    }
    return assess_figure19_kernel(
        traces["control"], traces["one"], traces["two"], traces["two_ach"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        default="configs/baselines/classic_smart_calibrated_v1.yaml",
    )
    parser.add_argument(
        "--protocol",
        default="configs/models/adex_rebound_ahp_promotion_v1.yaml",
    )
    parser.add_argument("--parameters", default="results/adex-isolated-refit-generation2-893.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    baseline = load_frozen_classic_baseline(args.baseline)
    raw_protocol = yaml.safe_load(Path(args.protocol).read_text())
    parameter_result = yaml.safe_load(Path(args.parameters).read_text())
    parameter_map = {
        name: AdExParameters.from_mapping(values)
        for name, values in parameter_result["selected_parameters"].items()
    }
    rebound_raw = raw_protocol["thalamic_rebound"]
    rebound_protocol = ReboundProtocol(
        baseline_ms=float(rebound_raw["baseline_ms"]),
        hyperpolarization_ms=float(rebound_raw["hyperpolarization_ms"]),
        release_ms=float(rebound_raw["release_ms"]),
        hyperpolarizing_passive_offset_mV=float(
            rebound_raw["hyperpolarizing_passive_offset_mV"]
        ),
        dt_ms=float(rebound_raw["dt_ms"]),
    )

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    conventions = baseline.runtime_conventions()
    facts = {fact.canonical_name: fact for fact in first_order_population_facts()}
    rebound_results = {}
    all_rebound = True
    for name in rebound_raw["cell_classes"]:
        params = first_order_population_parameters(facts[name], conventions=conventions)
        classic = run_rebound_protocol(
            population_factory=create_compartmental_hh_population,
            population_params=params,
            protocol=rebound_protocol,
            brian=brian,
        )
        alternative = run_rebound_protocol(
            population_factory=make_somatic_adex_factory(parameter_map[name]),
            population_params=params,
            protocol=rebound_protocol,
            brian=brian,
        )
        assessment = assess_rebound_match(classic, alternative)
        all_rebound &= assessment.promoted
        rebound_results[name] = {
            "classic": asdict(classic),
            "adex": asdict(alternative),
            "assessment": asdict(assessment),
            "promoted": assessment.promoted,
        }

    figure19_protocol = Figure19Protocol()
    classic_ahp = _figure19(create_compartmental_hh_population, figure19_protocol, brian)
    adex_ahp = _figure19(
        make_somatic_adex_factory(parameter_map["layer5_excitatory_v1"]),
        figure19_protocol,
        brian,
    )
    ahp_promoted = classic_ahp.reproduced and adex_ahp.reproduced
    output = {
        "schema_version": 1,
        "status": "completed-final-isolated-adex-promotion",
        "baseline_manifest": args.baseline,
        "baseline_manifest_fingerprint": baseline.manifest_fingerprint,
        "protocol": args.protocol,
        "parameter_source": args.parameters,
        "network_outcomes_used_for_selection": False,
        "rebound": rebound_results,
        "all_rebound_classes_promoted": all_rebound,
        "classic_ahp_ach": asdict(classic_ahp),
        "adex_ahp_ach": asdict(adex_ahp),
        "ahp_ach_promoted": ahp_promoted,
        "matched_parameter_map_promoted": all_rebound and ahp_promoted,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(output, sort_keys=False))


if __name__ == "__main__":
    main()
