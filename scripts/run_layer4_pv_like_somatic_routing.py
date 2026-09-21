"""Run the sealed one-factor L4 PV-like somatic-routing comparison."""

from __future__ import annotations

import argparse
import fcntl
import os
import platform
import tempfile
from pathlib import Path
from typing import Any

import yaml
from run_layer5_distal_nak_stage3 import (
    _condition_summary,
    _figure7_gates,
    _figure10_gates,
    _target_summary,
    _without_repetition,
)

from smart_robustness import classic_sector
from smart_robustness.baseline import load_frozen_classic_baseline
from smart_robustness.classic_sector import first_order_population_parameters
from smart_robustness.models.compartmental_hh import create_compartmental_hh_population
from smart_robustness.models.inhibitory_routing import (
    PROJECTION036_ID,
    make_layer4_projection036_somatic_population_factory,
    retarget_projection036_to_soma,
)
from smart_robustness.models.modeldb112923 import first_order_population_facts
from smart_robustness.protocols import MatchCondition
from smart_robustness.validation import figure7 as figure7_module
from smart_robustness.validation import figure10_search_cycle_spread as figure10_module
from smart_robustness.validation.active_apical_recording import file_sha256
from smart_robustness.validation.figure7 import TopDownCurrentMode
from smart_robustness.validation.figure10_search_cycle_spread import (
    build_projection036_variance_sector,
)

REPETITIONS = 2
DEFAULT_SEAL = Path(
    "docs/validation-results/post2008-layer4-pv-like-somatic-routing-seal-980.yaml"
)


def checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".l4-pv-routing-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            yaml.safe_dump(payload, stream, sort_keys=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def verify_seal(path: Path) -> dict:
    seal = yaml.safe_load(path.read_text())
    if seal["status"] != "sealed-before-network-outcomes":
        raise ValueError("a network execution seal is required")
    for filename, digest in seal["files"].items():
        if file_sha256(Path(filename)) != digest:
            raise ValueError(f"sealed file changed: {filename}")
    if "scripts/run_layer4_pv_like_somatic_routing.py" not in seal["files"]:
        raise ValueError("runner is missing from execution seal")
    return seal


def structural_summary(conventions) -> dict[str, Any]:
    fact = next(
        item
        for item in first_order_population_facts()
        if item.canonical_name == "layer4_excitatory_v1"
    )
    original = first_order_population_parameters(fact, conventions=conventions)
    transformed = retarget_projection036_to_soma(original)
    old_ports = {port.record_id: port for port in original["synaptic_ports"]}
    new_ports = {port.record_id: port for port in transformed["synaptic_ports"]}
    if old_ports.keys() != new_ports.keys():
        raise RuntimeError("PV-like transform changed the port inventory")
    changed = [
        record_id
        for record_id in old_ports
        if old_ports[record_id] != new_ports[record_id]
    ]
    if changed != [PROJECTION036_ID]:
        raise RuntimeError(f"unexpected changed ports: {changed}")
    old = old_ports[PROJECTION036_ID]
    new = new_ports[PROJECTION036_ID]
    old_total = (
        old.conductance_density_mS_cm2
        * fact.cell.compartment(old.compartment).lateral_area_cm2
        * 1e6
    )
    new_total = (
        new.conductance_density_mS_cm2
        * fact.cell.compartment(new.compartment).lateral_area_cm2
        * 1e6
    )
    if old_total != new_total:
        raise RuntimeError("projection 036 total receptor conductance changed")
    unchanged_parameters = all(
        original[key] == transformed[key]
        for key in original
        if key != "synaptic_ports"
    )
    if not unchanged_parameters:
        raise RuntimeError("a non-port population parameter changed")
    return {
        "changed_projection_ids": changed,
        "old_compartment": old.compartment,
        "new_compartment": new.compartment,
        "old_total_port_conductance_nS": old_total,
        "new_total_port_conductance_nS": new_total,
        "total_port_conductance_exact": old_total == new_total,
        "all_non_port_parameters_equal": unchanged_parameters,
    }


def transformed_builder(base_builder):
    population_factory = make_layer4_projection036_somatic_population_factory(
        base_factory=create_compartmental_hh_population
    )

    def builder(*args, **kwargs):
        if args:
            raise TypeError("transformed network builder accepts keyword arguments only")
        if kwargs.get("population_factory") is not None:
            raise ValueError("nested population-factory transformation is forbidden")
        kwargs = dict(kwargs)
        kwargs["population_factory"] = population_factory
        return base_builder(**kwargs)

    return builder


def run_repetition(
    *, learned_weights, conventions, scales, profile, repetition: int, brian
) -> dict[str, object]:
    figure7_protocol = profile["figure7_protocol"]
    comparator = profile["comparator"]
    common7 = {
        "learned_weights": learned_weights,
        "conventions": conventions,
        "persistent_projection_weight_scales": scales,
        "top_down_current_pA": float(figure7_protocol["top_down_current_pA"]),
        "top_down_current_mode": TopDownCurrentMode(
            figure7_protocol["top_down_current_mode"]
        ),
        "top_down_cue_lead_ms": float(figure7_protocol["top_down_cue_lead_ms"]),
        "duration_ms": float(figure7_protocol["duration_ms"]),
        "dt_ms": float(figure7_protocol["dt_ms"]),
        "equilibration_ms": float(figure7_protocol["equilibration_ms"]),
        "comparator_top_k_targets": int(comparator["target_count"]),
        "comparator_source_index": int(comparator["source_index"]),
        "record_relay_diagnostics": True,
        "brian": brian,
    }
    original7 = classic_sector.build_first_order_connected_sector
    try:
        classic_sector.build_first_order_connected_sector = transformed_builder(
            build_projection036_variance_sector
        )
        match_result = figure7_module.run_figure7_condition(
            condition=MatchCondition.MATCH, **common7
        )
        mismatch_result = figure7_module.run_figure7_condition(
            condition=MatchCondition.MISMATCH, **common7
        )
    finally:
        classic_sector.build_first_order_connected_sector = original7
    match = _condition_summary(match_result)
    mismatch = _condition_summary(mismatch_result)
    figure7_gates = _figure7_gates(match, mismatch, profile)
    outcome: dict[str, object] = {
        "repetition": repetition,
        "figure7": {
            "match": match,
            "mismatch": mismatch,
            "gates": figure7_gates,
            "pass": all(figure7_gates.values()),
        },
        "figure10": None,
        "behavioral_pass": False,
    }
    if not all(figure7_gates.values()):
        return outcome

    figure10_protocol = profile["figure10_protocol"]
    release_ms = float(figure10_protocol["release_after_mismatch_ms"])
    common10 = {
        "top_down_current_pA": float(figure10_protocol["top_down_current_pA"]),
        "pre_match_duration_ms": float(figure10_protocol["pre_match_duration_ms"]),
        "mismatch_duration_ms": float(figure10_protocol["mismatch_duration_ms"]),
        "release_after_mismatch_ms": release_ms,
        "learned_weights": learned_weights,
        "persistent_projection_weight_scales": scales,
        "persistent_projection_delays_ms": {},
        "comparator_top_k_targets": int(comparator["target_count"]),
        "comparator_source_index": int(comparator["source_index"]),
        "top_down_current_mode": figure10_protocol["top_down_current_mode"],
        "conventions": conventions,
        "dt_ms": float(figure10_protocol["dt_ms"]),
        "brian": brian,
    }
    original10 = figure10_module._build_first_order_connected_sector
    try:
        figure10_module._build_first_order_connected_sector = transformed_builder(
            original10
        )
        intact = figure10_module.run_figure10_search_cycle_spread_condition(
            reset_pathway_enabled=True, **common10
        )
        control = figure10_module.run_figure10_search_cycle_spread_condition(
            reset_pathway_enabled=False, **common10
        )
    finally:
        figure10_module._build_first_order_connected_sector = original10
    intact_summary = _target_summary(intact, release_ms=release_ms, late_start_ms=75.0)
    control_summary = _target_summary(control, release_ms=release_ms, late_start_ms=75.0)
    figure10_gates = _figure10_gates(
        intact, control, intact_summary, control_summary
    )
    outcome["figure10"] = {
        "intact": intact_summary,
        "disconnected_control": control_summary,
        "gates": figure10_gates,
        "pass": all(figure10_gates.values()),
    }
    outcome["behavioral_pass"] = all(figure10_gates.values())
    return outcome


def recursive_differences(left: Any, right: Any, path: str = "") -> list[dict[str, Any]]:
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            return [{"path": path, "legacy": left, "transformed": right}]
        differences = []
        for key in left:
            differences.extend(
                recursive_differences(left[key], right[key], f"{path}.{key}".lstrip("."))
            )
        return differences
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [{"path": path, "legacy": left, "transformed": right}]
        differences = []
        for index, (old, new) in enumerate(zip(left, right, strict=True)):
            differences.extend(recursive_differences(old, new, f"{path}[{index}]"))
        return differences
    return [] if left == right else [{"path": path, "legacy": left, "transformed": right}]


def load_legacy_reference(path: Path, expected_sha256: str) -> dict[str, Any]:
    if file_sha256(path) != expected_sha256:
        raise ValueError("archived legacy result hash differs")
    payload = yaml.safe_load(path.read_text())
    if payload["status"] != "completed-inhibitory-routing-null":
        raise ValueError("legacy reference is not complete")
    if payload["cross_arm_exact"] is not True:
        raise ValueError("legacy wrapped and unwrapped controls are not exact")
    outcomes = [
        outcome
        for arm in ("unwrapped_control", "wrapped_legacy_aggregate")
        for outcome in payload["arms"][arm]["outcomes"]
    ]
    normalized = [_without_repetition(outcome) for outcome in outcomes]
    if len(normalized) != 4 or any(item != normalized[0] for item in normalized[1:]):
        raise ValueError("legacy reference does not contain four exact outcomes")
    if not all(outcome["behavioral_pass"] for outcome in outcomes):
        raise ValueError("legacy reference contains a behavioral failure")
    return normalized[0]


def execute(output: Path, seal_path: Path) -> None:
    import brian2 as brian

    seal = verify_seal(seal_path)
    baseline = load_frozen_classic_baseline(seal["baseline_manifest"])
    conventions = baseline.runtime_conventions()
    structural = structural_summary(conventions)
    legacy_reference = load_legacy_reference(
        Path(seal["legacy_result"]), seal["legacy_result_sha256"]
    )
    figure6_path = Path(seal["figure6_result"])
    figure6 = yaml.safe_load(figure6_path.read_text())
    source_trials = figure6["arms"]["ratio_0"]["trials"]
    keys = ("bottom_up_weights", "top_down_wide_weights", "top_down_narrow_weights")
    if len(source_trials) != 2 or any(
        source_trials[0][key] != source_trials[1][key] for key in keys
    ):
        raise ValueError("sealed ratio-zero Figure 6 weights are not exact repeats")
    learned_weights = {
        "modeldb112923.projection.035": source_trials[0]["bottom_up_weights"],
        "modeldb112923.projection.005": source_trials[0]["top_down_wide_weights"],
        "modeldb112923.projection.007": source_trials[0]["top_down_narrow_weights"],
    }
    manifest = yaml.safe_load(Path(seal["baseline_manifest"]).read_text())
    profile = yaml.safe_load(
        (baseline.repository_root / manifest["implementation"]["profile"]["path"]).read_text()
    )
    identity = {
        "seal_sha256": file_sha256(seal_path),
        "baseline_manifest_fingerprint": baseline.manifest_fingerprint,
        "runtime_fingerprint": baseline.runtime_fingerprint,
        "legacy_result_sha256": file_sha256(Path(seal["legacy_result"])),
        "figure6_result_sha256": file_sha256(figure6_path),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "brian2": brian.__version__,
    }
    payload = {
        "schema_version": 1,
        "status": "running-layer4-pv-like-somatic-routing",
        "identity": identity,
        "structural": structural,
        "outcomes": [],
        "exact_repeat": None,
        "legacy_exact": None,
        "differences_from_legacy": None,
        "classification": None,
        "network_execution": True,
        "frozen_baseline_modified": False,
    }
    if output.exists():
        payload = yaml.safe_load(output.read_text())
        if payload["identity"] != identity or payload["structural"] != structural:
            raise ValueError("checkpoint identity or structural summary mismatch")
    outcomes = list(payload["outcomes"])
    if len(outcomes) > REPETITIONS:
        raise ValueError("checkpoint has too many repetitions")
    if payload["status"] == "completed-layer4-pv-like-somatic-routing":
        if len(outcomes) != REPETITIONS or payload["classification"] is None:
            raise ValueError("incomplete checkpoint marked complete")
        return

    brian.prefs.codegen.target = "numpy"
    for repetition in range(len(outcomes), REPETITIONS):
        outcomes.append(
            run_repetition(
                learned_weights=learned_weights,
                conventions=conventions,
                scales=dict(baseline.projection_weight_scales),
                profile=profile,
                repetition=repetition,
                brian=brian,
            )
        )
        payload["outcomes"] = outcomes
        checkpoint(output, payload)
    normalized = [_without_repetition(outcome) for outcome in outcomes]
    payload["exact_repeat"] = normalized[0] == normalized[1]
    payload["legacy_exact"] = all(outcome == legacy_reference for outcome in normalized)
    payload["differences_from_legacy"] = recursive_differences(
        legacy_reference, normalized[0]
    )
    all_gates = payload["exact_repeat"] and all(
        outcome["behavioral_pass"] for outcome in outcomes
    )
    if not all_gates:
        classification = "failure"
    elif payload["legacy_exact"]:
        classification = "exact_survival"
    else:
        classification = "robust_changed_survival"
    payload["classification"] = classification
    payload["status"] = "completed-layer4-pv-like-somatic-routing"
    checkpoint(output, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seal", type=Path, default=DEFAULT_SEAL)
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(output.suffix + ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("an identical L4 PV-like runner is active") from error
        execute(output, args.seal.resolve())


if __name__ == "__main__":
    main()
