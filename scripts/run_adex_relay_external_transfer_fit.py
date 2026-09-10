"""Fit relay AdEx parameters to isolated current and external-input transfer."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import yaml

from smart_robustness.baseline import load_frozen_classic_baseline
from smart_robustness.classic_sector import first_order_population_parameters
from smart_robustness.models.adex import load_adex_parameter_map
from smart_robustness.models.adex_parameters import AdExParameters
from smart_robustness.models.compartmental_hh import create_compartmental_hh_population
from smart_robustness.models.modeldb112923 import first_order_population_facts
from smart_robustness.validation.isolated_cell_matching import (
    CurrentStepProtocol,
    ExternalInputProtocol,
    RelayInhibitionProtocol,
    StepPhenotype,
    external_input_phenotype_loss,
    generate_adex_sobol_candidates,
    generate_extended_adex_sobol_candidates,
    generate_full_adex_sobol_candidates,
    phenotype_loss,
    relay_inhibition_phenotype_loss,
    run_adex_candidate_batch,
    run_adex_external_input_candidate_batch,
    run_adex_relay_inhibition_candidate_batch,
    run_external_input_protocol,
    run_relay_inhibition_protocol,
    subset_external_input_phenotype,
    subset_phenotype,
    subset_relay_inhibition_phenotype,
)


def _step_phenotype(values: dict) -> StepPhenotype:
    return StepPhenotype(**{key: tuple(value) if isinstance(value, list) else value for key, value in values.items()})


def _parameter_tuple(parameters: AdExParameters) -> tuple[float, ...]:
    return tuple(float(value) for value in parameters.as_dict().values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline", default="configs/baselines/classic_smart_calibrated_v1.yaml"
    )
    parser.add_argument(
        "--protocol",
        default="configs/models/adex_relay_external_transfer_protocol_v1.yaml",
    )
    parser.add_argument(
        "--current-targets", default="results/adex-isolated-fit-v3-890.yaml"
    )
    parser.add_argument(
        "--predecessor", default="configs/models/adex_current_step_matched_v1.yaml"
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    baseline = load_frozen_classic_baseline(args.baseline)
    raw = yaml.safe_load(Path(args.protocol).read_text())
    scope = raw["scope"]
    candidate_config = raw["candidate_generation"]
    external_protocol = ExternalInputProtocol(
        pre_ms=float(scope["pre_ms"]),
        step_ms=float(scope["step_ms"]),
        post_ms=float(scope["post_ms"]),
        dt_ms=float(scope["dt_ms"]),
    )
    source_values = tuple(float(value) for value in scope["source_values"])
    external_training_positions = tuple(int(i) for i in scope["training_positions"])
    external_holdout_positions = tuple(
        int(i) for i in scope["sealed_holdout_positions"]
    )
    current_config = raw["retained_current_step_target"]
    current_training_positions = tuple(
        int(i) for i in current_config["training_positions"]
    )
    current_holdout_positions = tuple(
        int(i) for i in current_config["sealed_holdout_positions"]
    )
    bounds = {
        name: (float(values[0]), float(values[1]))
        for name, values in candidate_config["bounds"].items()
    }
    if "somatic_capacitance_scale" in bounds:
        generator = generate_full_adex_sobol_candidates
    elif "effective_leak_offset_mV" in bounds:
        generator = generate_extended_adex_sobol_candidates
    else:
        generator = generate_adex_sobol_candidates
    candidates = list(
        generator(
            bounds,
            count=int(candidate_config["candidates"]),
            include_literature=bool(candidate_config["append_literature_candidate"]),
        )
    )
    predecessor = load_adex_parameter_map(args.predecessor)["thalamic_relay"]
    if predecessor not in candidates:
        candidates.append(predecessor)

    facts = next(
        item
        for item in first_order_population_facts()
        if item.canonical_name == "thalamic_relay"
    )
    params = first_order_population_parameters(
        facts, conventions=baseline.runtime_conventions()
    )
    current_raw = yaml.safe_load(Path(args.current_targets).read_text())
    current_target = _step_phenotype(
        current_raw["cell_classes"]["thalamic_relay"]["classic_selected_target"]
    )
    current_protocol_raw = yaml.safe_load(
        Path(current_raw["protocol"]).read_text()
    )["classic_target_scan"]
    current_protocol = CurrentStepProtocol(
        training_currents_pA=(0.0,),
        holdout_currents_pA=(1.0,),
        pre_ms=float(current_protocol_raw["pre_ms"]),
        step_ms=float(current_protocol_raw["step_ms"]),
        post_ms=float(current_protocol_raw["post_ms"]),
        dt_ms=float(current_protocol_raw["dt_ms"]),
    )

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    external_target = run_external_input_protocol(
        population_factory=create_compartmental_hh_population,
        population_params=params,
        source_values=source_values,
        record_id=str(scope["input_record_id"]),
        channel=str(scope["channel"]),
        protocol=external_protocol,
        brian=brian,
    )
    candidate_external = run_adex_external_input_candidate_batch(
        candidates=candidates,
        population_params=params,
        source_values=source_values,
        record_id=str(scope["input_record_id"]),
        channel=str(scope["channel"]),
        protocol=external_protocol,
        brian=brian,
    )
    candidate_current = run_adex_candidate_batch(
        candidates=candidates,
        population_params=params,
        currents_pA=current_target.currents_pA,
        protocol=current_protocol,
        brian=brian,
    )
    context_target = None
    candidate_context = None
    context_training_positions: tuple[int, ...] = ()
    context_holdout_positions: tuple[int, ...] = ()
    if "inhibitory_context" in raw:
        context_raw = raw["inhibitory_context"]
        condition_sources = tuple(
            float(item["source_value"]) for item in context_raw["conditions"]
        )
        condition_scales = tuple(
            float(item["inhibition_scale"]) for item in context_raw["conditions"]
        )
        gate_baselines = {
            str(record_id): float(value)
            for record_id, value in context_raw["gate_baselines"].items()
        }
        context_protocol = RelayInhibitionProtocol(
            pre_ms=float(context_raw["pre_ms"]),
            step_ms=float(context_raw["step_ms"]),
            post_ms=float(context_raw["post_ms"]),
            dt_ms=float(context_raw["dt_ms"]),
        )
        context_training_positions = tuple(
            int(index) for index in context_raw["training_positions"]
        )
        context_holdout_positions = tuple(
            int(index) for index in context_raw["sealed_holdout_positions"]
        )
        context_target = run_relay_inhibition_protocol(
            population_factory=create_compartmental_hh_population,
            population_params=params,
            source_values=condition_sources,
            inhibition_scales=condition_scales,
            external_record_id=str(scope["input_record_id"]),
            channel=str(scope["channel"]),
            inhibitory_gate_baselines=gate_baselines,
            protocol=context_protocol,
            brian=brian,
        )
        candidate_context = run_adex_relay_inhibition_candidate_batch(
            candidates=candidates,
            population_params=params,
            source_values=condition_sources,
            inhibition_scales=condition_scales,
            external_record_id=str(scope["input_record_id"]),
            channel=str(scope["channel"]),
            inhibitory_gate_baselines=gate_baselines,
            protocol=context_protocol,
            brian=brian,
        )

    current_training_target = subset_phenotype(
        current_target, current_training_positions
    )
    current_holdout_target = subset_phenotype(current_target, current_holdout_positions)
    external_training_target = subset_external_input_phenotype(
        external_target, external_training_positions
    )
    external_holdout_target = subset_external_input_phenotype(
        external_target, external_holdout_positions
    )
    scored = []
    context_iterator = candidate_context or (None,) * len(candidates)
    for parameters, current, external, context in zip(
        candidates,
        candidate_current,
        candidate_external,
        context_iterator,
        strict=True,
    ):
        current_training = phenotype_loss(
            current_training_target,
            subset_phenotype(current, current_training_positions),
        )
        current_holdout = phenotype_loss(
            current_holdout_target,
            subset_phenotype(current, current_holdout_positions),
        )
        external_training = external_input_phenotype_loss(
            external_training_target,
            subset_external_input_phenotype(external, external_training_positions),
            include_model_specific_peak=bool(
                raw["selection"].get("include_model_specific_peak", True)
            ),
        )
        external_holdout = external_input_phenotype_loss(
            external_holdout_target,
            subset_external_input_phenotype(external, external_holdout_positions),
            include_model_specific_peak=bool(
                raw["selection"].get("include_model_specific_peak", True)
            ),
        )
        context_training = 0.0
        context_holdout = 0.0
        if context_target is not None and context is not None:
            context_training = relay_inhibition_phenotype_loss(
                subset_relay_inhibition_phenotype(
                    context_target, context_training_positions
                ),
                subset_relay_inhibition_phenotype(
                    context, context_training_positions
                ),
            )
            context_holdout = relay_inhibition_phenotype_loss(
                subset_relay_inhibition_phenotype(
                    context_target, context_holdout_positions
                ),
                subset_relay_inhibition_phenotype(
                    context, context_holdout_positions
                ),
            )
        scored.append(
            {
                "parameters": parameters,
                "current": current,
                "external": external,
                "current_training_loss": current_training,
                "current_holdout_loss": current_holdout,
                "external_training_loss": external_training,
                "external_holdout_loss": external_holdout,
                "context": context,
                "context_training_loss": context_training,
                "context_holdout_loss": context_holdout,
                "combined_training_loss": (
                    current_training + external_training + context_training
                ),
                "combined_holdout_loss": (
                    current_holdout + external_holdout + context_holdout
                ),
            }
        )
    selected = min(
        scored,
        key=lambda item: (
            item["combined_training_loss"],
            _parameter_tuple(item["parameters"]),
        ),
    )
    predecessor_result = next(
        item for item in scored if item["parameters"] == predecessor
    )
    promoted = bool(
        selected["current"].finite
        and selected["external"].finite
        and selected["combined_training_loss"]
        < predecessor_result["combined_training_loss"]
        and selected["combined_holdout_loss"]
        < predecessor_result["combined_holdout_loss"]
    )
    output = {
        "schema_version": 1,
        "status": "completed-network-blind-relay-transfer-fit",
        "baseline_manifest": args.baseline,
        "baseline_manifest_fingerprint": baseline.manifest_fingerprint,
        "protocol": args.protocol,
        "network_outcomes_used_for_selection": False,
        "candidate_count": len(candidates),
        "classic_external_target": asdict(external_target),
        "selected_parameters": selected["parameters"].as_dict(),
        "selected_current_phenotype": asdict(selected["current"]),
        "selected_external_phenotype": asdict(selected["external"]),
        "classic_inhibitory_context_target": (
            None if context_target is None else asdict(context_target)
        ),
        "selected_inhibitory_context_phenotype": (
            None if selected["context"] is None else asdict(selected["context"])
        ),
        "selected_losses": {
            key: value for key, value in selected.items() if key.endswith("_loss")
        },
        "predecessor_losses": {
            key: value
            for key, value in predecessor_result.items()
            if key.endswith("_loss")
        },
        "sealed_holdout_promoted": promoted,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(output, sort_keys=False))


if __name__ == "__main__":
    main()
