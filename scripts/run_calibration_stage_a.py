"""Run the predeclared isolated-TRN calibration gate and persist all outcomes."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import yaml

from smart_robustness.validation.calibration import (
    TRN_STAGE_A_DIMENSIONS,
    load_calibration_contract,
    run_trn_stage_a_candidate,
)
from smart_robustness.validation.isolated_cells import TrnRecruitmentProtocol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract", default="configs/calibration/classic_uncertainty_space.yaml"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--where",
        action="append",
        default=[],
        metavar="DIMENSION=VALUE",
        help="retain only candidates with the declared categorical value",
    )
    args = parser.parse_args()

    import brian2 as brian

    brian.prefs.codegen.target = "numpy"
    contract = load_calibration_contract(args.contract)
    candidates = list(contract.iter_candidates(TRN_STAGE_A_DIMENSIONS))
    filters = {}
    for expression in args.where:
        try:
            name, value = expression.split("=", 1)
        except ValueError as exc:
            raise ValueError("--where requires DIMENSION=VALUE") from exc
        if name in filters:
            raise ValueError(f"duplicate --where dimension: {name}")
        filters[name] = value
    if filters:
        unknown = sorted(set(filters) - {item.name for item in contract.dimensions})
        if unknown:
            raise ValueError(f"unknown --where dimensions: {unknown}")
        candidates = [
            candidate
            for candidate in candidates
            if all(str(candidate[name]) == value for name, value in filters.items())
        ]
    if args.limit is not None:
        candidates = candidates[: args.limit]
    protocol = TrnRecruitmentProtocol()
    outcomes = []
    for index, candidate in enumerate(candidates, start=1):
        result = run_trn_stage_a_candidate(
            contract, candidate, protocol=protocol, brian=brian
        )
        outcomes.append(
            {
                "ordinal": index,
                "candidate": candidate,
                "result": asdict(result),
                "quiescent_control_pass": result.quiescent_control_pass,
                "driven_recruitment_pass": result.driven_recruitment_pass,
                "promoted": result.promoted,
            }
        )
        print(
            f"[{index}/{len(candidates)}] {result.candidate_fingerprint[:12]} "
            f"control={result.control_post_drive_spikes} "
            f"driven={result.driven_post_drive_spikes} promoted={result.promoted}",
            flush=True,
        )

    promoted = [item for item in outcomes if item["promoted"]]
    output = Path(args.output)
    artifact = {
        "id": output.stem,
        "date": datetime.now(tz=UTC).date().isoformat(),
        "status": "passing" if promoted else "failed-no-survivor",
        "claim": "Whether declared classic-SMART candidates pass isolated TRN quiescence and recruitment",
        "contract": str(args.contract),
        "contract_fingerprint": contract.fingerprint,
        "base_tag": contract.base_tag,
        "active_dimensions": list(TRN_STAGE_A_DIMENSIONS),
        "filters": filters,
        "protocol": asdict(protocol),
        "gate": {
            "quiescent_control": "finite and zero events after pre-drive",
            "driven_recruitment": "finite and at least one event after drive onset",
        },
        "candidate_count": len(outcomes),
        "promoted_count": len(promoted),
        "promoted_candidate_fingerprints": [
            item["result"]["candidate_fingerprint"] for item in promoted
        ],
        "outcomes": outcomes,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(artifact, sort_keys=False))


if __name__ == "__main__":
    main()
