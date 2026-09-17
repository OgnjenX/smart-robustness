"""Run sealed structural checks for the direct visual-thalamus L5 module."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from smart_robustness.baseline import load_frozen_classic_baseline
from smart_robustness.modeldb_projections import MODELDB_FIRST_ORDER
from smart_robustness.models.direct_visual_thalamus_layer5 import (
    DIRECT_VISUAL_THALAMUS_LAYER5_ID,
    DIRECT_VISUAL_THALAMUS_LAYER5_TARGET,
    DIRECT_VISUAL_THALAMUS_LAYER5_TEMPLATE_ID,
    REGISTERED_DIRECT_VISUAL_THALAMUS_LAYER5_RATIOS,
    direct_visual_thalamus_layer5_record,
    make_direct_visual_thalamus_layer5_sector_builder,
)
from smart_robustness.validation.figure10_search_cycle_spread import (
    build_projection036_variance_sector,
)


def _finite(sector, brian) -> bool:
    return all(
        np.isfinite(
            np.asarray(
                getattr(population.group, f"v_{compartment}")[:] / brian.mV
            )
        ).all()
        for population in sector.populations.values()
        for compartment in population.compartments
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        default="configs/baselines/classic_smart_calibrated_v1.yaml",
    )
    parser.add_argument(
        "--study",
        default="configs/robustness/direct_visual_thalamus_layer5_v1.yaml",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        raise FileExistsError("refusing to overwrite an existing precheck result")
    study = yaml.safe_load(Path(args.study).read_text())
    ratios = tuple(float(value) for value in study["independent_variable"]["values"])
    if ratios != REGISTERED_DIRECT_VISUAL_THALAMUS_LAYER5_RATIOS:
        raise ValueError("study ratio grid differs from the sealed implementation")
    baseline = load_frozen_classic_baseline(args.baseline)
    conventions = baseline.runtime_conventions()
    template = MODELDB_FIRST_ORDER.by_id(DIRECT_VISUAL_THALAMUS_LAYER5_TEMPLATE_ID)

    record_checks = []
    for ratio in ratios[1:]:
        record = direct_visual_thalamus_layer5_record(ratio)
        copied_fields = (
            "kind",
            "source_population",
            "source_compartment",
            "dependency",
            "channel_conductance_mS_cm2",
            "reversal_mV",
            "rise_ms",
            "fall_ms",
            "delay_ms",
            "method",
            "kernel",
            "gate_attributes",
        )
        copied = all(getattr(record, key) == getattr(template, key) for key in copied_fields)
        record_checks.append(
            {
                "ratio": ratio,
                "fixed_peak_weight": record.weight,
                "copied_executable_fields_match": copied,
                "target_population": record.target_population,
                "target_compartment": record.target_compartment,
                "modifiable": record.modifiable,
            }
        )

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    brian.defaultclock.dt = 0.01 * brian.ms
    zero_builder = make_direct_visual_thalamus_layer5_sector_builder(
        strength_ratio=0.0,
        base_builder=build_projection036_variance_sector,
    )
    zero_is_exact_builder = zero_builder is build_projection036_variance_sector
    brian.start_scope()
    control = zero_builder(conventions=conventions, brian=brian)
    control_projection_ids = tuple(sorted(control.projections))
    control_l5_port_ids = tuple(
        port.record_id
        for port in control.populations[
            DIRECT_VISUAL_THALAMUS_LAYER5_TARGET
        ].compiled.synaptic_ports
    )
    control.network.run(0.05 * brian.ms)
    control_finite = _finite(control, brian)

    representative_ratio = 0.5
    intervention_builder = make_direct_visual_thalamus_layer5_sector_builder(
        strength_ratio=representative_ratio,
        base_builder=build_projection036_variance_sector,
    )
    brian.start_scope()
    intervention = intervention_builder(conventions=conventions, brian=brian)
    intervention_projection_ids = tuple(sorted(intervention.projections))
    intervention_l5_port_ids = tuple(
        port.record_id
        for port in intervention.populations[
            DIRECT_VISUAL_THALAMUS_LAYER5_TARGET
        ].compiled.synaptic_ports
    )
    added_projection_ids = tuple(
        sorted(set(intervention_projection_ids) - set(control_projection_ids))
    )
    added_l5_port_ids = tuple(
        item for item in intervention_l5_port_ids if item not in control_l5_port_ids
    )
    projection = intervention.projections[DIRECT_VISUAL_THALAMUS_LAYER5_ID]
    template_projection = intervention.projections[
        DIRECT_VISUAL_THALAMUS_LAYER5_TEMPLATE_ID
    ]
    topology_identical = bool(
        np.array_equal(np.asarray(projection.i), np.asarray(template_projection.i))
        and np.array_equal(np.asarray(projection.j), np.asarray(template_projection.j))
    )
    weight_ratio_exact = bool(
        np.allclose(
            np.asarray(projection.w),
            representative_ratio * np.asarray(template_projection.w),
            rtol=0.0,
            atol=1e-15,
        )
    )
    intervention.network.run(0.05 * brian.ms)
    intervention_finite = _finite(intervention, brian)
    other_population_has_added_port = any(
        DIRECT_VISUAL_THALAMUS_LAYER5_ID
        in {port.record_id for port in population.compiled.synaptic_ports}
        for name, population in intervention.populations.items()
        if name != DIRECT_VISUAL_THALAMUS_LAYER5_TARGET
    )

    all_checks_pass = bool(
        zero_is_exact_builder
        and all(item["copied_executable_fields_match"] for item in record_checks)
        and all(item["modifiable"] is False for item in record_checks)
        and added_projection_ids == (DIRECT_VISUAL_THALAMUS_LAYER5_ID,)
        and added_l5_port_ids == (DIRECT_VISUAL_THALAMUS_LAYER5_ID,)
        and not other_population_has_added_port
        and topology_identical
        and weight_ratio_exact
        and control_finite
        and intervention_finite
    )
    payload = {
        "schema_version": 1,
        "status": "completed-direct-visual-thalamus-layer5-prechecks",
        "baseline_manifest": args.baseline,
        "baseline_manifest_fingerprint": baseline.manifest_fingerprint,
        "runtime_fingerprint": baseline.runtime_fingerprint,
        "study": args.study,
        "registration": (
            "docs/validation-results/"
            "post2008-direct-visual-thalamus-layer5-registration-938.yaml"
        ),
        "network_outcomes_observed": False,
        "ratio_zero_is_exact_frozen_builder": zero_is_exact_builder,
        "record_checks": record_checks,
        "representative_nonzero_ratio": representative_ratio,
        "control_projection_count": len(control_projection_ids),
        "intervention_projection_count": len(intervention_projection_ids),
        "added_projection_ids": added_projection_ids,
        "added_l5_port_ids": added_l5_port_ids,
        "other_population_has_added_port": other_population_has_added_port,
        "topology_indices_identical_to_projection035": topology_identical,
        "projection_weights_exact_registered_ratio": weight_ratio_exact,
        "control_brief_run_finite": control_finite,
        "intervention_brief_run_finite": intervention_finite,
        "all_checks_pass": all_checks_pass,
        "figure6_authorized": all_checks_pass,
        "later_stages_authorized": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(payload, sort_keys=False))


if __name__ == "__main__":
    main()
