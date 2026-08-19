"""Merge deterministic Stage A shards after validating complete coverage."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("inputs", nargs="+")
    args = parser.parse_args()

    shards = [yaml.safe_load(Path(path).read_text()) for path in args.inputs]
    shards.sort(key=lambda item: item["candidate_offset"])
    contract_fingerprints = {item["contract_fingerprint"] for item in shards}
    if len(contract_fingerprints) != 1:
        raise ValueError("shards have different contract fingerprints")
    filters = {(str(item.get("filters")), str(item.get("filters_in"))) for item in shards}
    if len(filters) != 1:
        raise ValueError("shards have different filters")

    outcomes = []
    expected_offset = 0
    for shard in shards:
        if shard["candidate_offset"] != expected_offset:
            raise ValueError(
                f"non-contiguous shard: expected offset {expected_offset}, "
                f"got {shard['candidate_offset']}"
            )
        outcomes.extend(shard["outcomes"])
        expected_offset += shard["candidate_count"]

    fingerprints = [item["result"]["candidate_fingerprint"] for item in outcomes]
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError("duplicate candidate fingerprints across shards")
    for ordinal, outcome in enumerate(outcomes, start=1):
        outcome["ordinal"] = ordinal

    promoted = [item for item in outcomes if item["promoted"]]
    merged = dict(shards[0])
    merged.update(
        {
            "id": Path(args.output).stem,
            "status": "passing" if promoted else "failed-no-survivor",
            "candidate_count": len(outcomes),
            "candidate_offset": 0,
            "promoted_count": len(promoted),
            "promoted_candidate_fingerprints": [
                item["result"]["candidate_fingerprint"] for item in promoted
            ],
            "shards": [
                {
                    "id": item["id"],
                    "candidate_offset": item["candidate_offset"],
                    "candidate_count": item["candidate_count"],
                }
                for item in shards
            ],
            "outcomes": outcomes,
        }
    )
    Path(args.output).write_text(yaml.safe_dump(merged, sort_keys=False))


if __name__ == "__main__":
    main()
