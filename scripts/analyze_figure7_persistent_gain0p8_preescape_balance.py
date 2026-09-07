"""Aggregate the pinned Figure 7 current balance before nonoverlap escape."""

from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np
import yaml
from run_figure6_nonspecific_distal_gaba_source import _plain


def _summarize(values: np.ndarray, times: np.ndarray) -> dict[str, float]:
    return {
        "mean_pA": float(np.mean(values)),
        "minimum_pA": float(np.min(values)),
        "maximum_pA": float(np.max(values)),
        "time_integral_pA_ms": float(np.trapz(values, times)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    registration_path = Path(args.registration)
    registration = yaml.safe_load(registration_path.read_text())
    result_path = Path(registration["source_result"])
    if sha256(result_path.read_bytes()).hexdigest() != registration["source_result_sha256"]:
        raise ValueError("source result hash differs from registration")
    result = yaml.safe_load(result_path.read_text())
    trace_path = Path(registration["trace"])
    if sha256(trace_path.read_bytes()).hexdigest() != registration["trace_sha256"]:
        raise ValueError("trace hash differs from registration")

    first_events = {}
    for cell, time_ms in zip(
        result["mismatch"]["relay_spike_indices"],
        result["mismatch"]["relay_spike_times_ms"],
        strict=True,
    ):
        first_events.setdefault(int(cell), float(time_ms))

    contract = registration["analysis_contract"]
    target_cells = [int(cell) for cell in contract["target_nonoverlap_cells"]]
    current_groups = {
        name: tuple(variables) for name, variables in contract["current_groups"].items()
    }
    fixed_start, fixed_end = map(float, contract["fixed_epoch_ms"])
    maintenance_start, maintenance_end_before = map(
        float, contract["event_anchored_epochs_ms_before_first_event"]["maintenance"]
    )
    final_start_before, final_end_before = map(
        float,
        contract["event_anchored_epochs_ms_before_first_event"]["final_preescape"],
    )

    per_cell = {}
    with np.load(trace_path, allow_pickle=False) as trace:
        times = trace["time_ms"]
        rows = {int(cell): row for row, cell in enumerate(trace["cell_indices"].tolist())}
        for cell in target_cells:
            row = rows[cell]
            first_event = first_events[cell]
            epochs = {
                "fixed_early_inhibition": (fixed_start, fixed_end),
                "maintenance": (maintenance_start, first_event - maintenance_end_before),
                "final_preescape": (
                    first_event - final_start_before,
                    first_event - final_end_before,
                ),
            }
            epoch_summaries = {}
            for epoch_name, (start_ms, end_ms) in epochs.items():
                mask = (times >= start_ms) & (times < end_ms)
                epoch_times = times[mask]
                grouped = {
                    name: sum(trace[variable][row, mask] for variable in variables)
                    for name, variables in current_groups.items()
                }
                net = sum(grouped.values())
                epoch_summaries[epoch_name] = {
                    "start_ms": start_ms,
                    "end_ms_exclusive": end_ms,
                    "sample_count": int(np.count_nonzero(mask)),
                    "current_groups": {
                        name: _summarize(values, epoch_times)
                        for name, values in grouped.items()
                    },
                    "net_external_plus_synaptic": {
                        **_summarize(net, epoch_times),
                        "fraction_samples_positive": float(np.mean(net > 0.0)),
                    },
                    "intrinsic_and_axial_ranges_pA": {
                        name: {
                            "minimum": float(np.min(trace[name][row, mask])),
                            "maximum": float(np.max(trace[name][row, mask])),
                        }
                        for name in (
                            "i_ca_soma",
                            "i_na_soma",
                            "i_k_soma",
                            "i_axial_inward_soma",
                        )
                    },
                }
            per_cell[str(cell)] = {
                "first_event_ms": first_event,
                "epochs": epoch_summaries,
            }

    def group_means(epoch: str, group: str) -> list[float]:
        return [
            per_cell[str(cell)]["epochs"][epoch]["current_groups"][group]["mean_pA"]
            for cell in target_cells
        ]

    aggregate = {}
    for epoch in ("fixed_early_inhibition", "maintenance", "final_preescape"):
        aggregate[epoch] = {
            group: {
                "mean_across_cells_pA": float(np.mean(group_means(epoch, group))),
                "range_of_cell_means_pA": [
                    float(np.min(group_means(epoch, group))),
                    float(np.max(group_means(epoch, group))),
                ],
            }
            for group in current_groups
        }
        positive_fractions = [
            per_cell[str(cell)]["epochs"][epoch]["net_external_plus_synaptic"][
                "fraction_samples_positive"
            ]
            for cell in target_cells
        ]
        aggregate[epoch]["net_positive_fraction_across_cells"] = {
            "mean": float(np.mean(positive_fractions)),
            "range": [float(np.min(positive_fractions)), float(np.max(positive_fractions))],
        }

    early_interneuron = abs(
        aggregate["fixed_early_inhibition"]["interneuron_gaba"]["mean_across_cells_pA"]
    )
    late_interneuron = abs(
        aggregate["final_preescape"]["interneuron_gaba"]["mean_across_cells_pA"]
    )
    late_direct = aggregate["final_preescape"]["direct_image_input"][
        "mean_across_cells_pA"
    ]
    late_trn = abs(
        aggregate["final_preescape"]["trn_gaba"]["mean_across_cells_pA"]
    )

    artifact = {
        "schema_version": 1,
        "id": output.stem,
        "date": "2026-09-07",
        "status": "preescape-balance-localized-source-uncertainty-remains",
        "classification": registration["classification"],
        "registration": str(registration_path),
        "source_result_sha256": registration["source_result_sha256"],
        "trace_sha256": registration["trace_sha256"],
        "per_cell": per_cell,
        "aggregate": aggregate,
        "derived_localization": {
            "early_to_final_interneuron_mean_magnitude_ratio": (
                early_interneuron / late_interneuron
            ),
            "final_direct_to_trn_mean_magnitude_ratio": late_direct / late_trn,
            "interneuron_inhibition_wanes_before_escape": late_interneuron < early_interneuron,
            "trn_inhibition_remains_active_before_escape": late_trn > 0.0,
            "persistent_direct_excitation_remains_active_before_escape": late_direct > 0.0,
        },
        "assessment": {
            "calcium_localized_as_primary_cause": False,
            "source_parameter_selected": False,
            "original_smart_reproduced": False,
            "baseline_promoted": False,
        },
    }
    output.open("x").write(yaml.safe_dump(_plain(artifact), sort_keys=False))


if __name__ == "__main__":
    main()
