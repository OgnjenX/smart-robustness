"""Bounded immutable receipts; acquisition correctness is not physiological validity."""

from __future__ import annotations

import hashlib
from http.client import IncompleteRead
from pathlib import Path
from typing import BinaryIO


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def receive_file(
    stream: BinaryIO,
    path: Path,
    *,
    expected_bytes: int,
    maximum_bytes: int,
    chunk_bytes: int = 1048576,
) -> dict:
    """Save a fresh receipt, including partial failures; never overwrite or resume bytes.

    Read at most the expected length plus one sentinel byte to detect oversized
    responses. A failed receipt remains evidence, not a valid downloadable cell.
    Existing receipts require separate checkpoint validation by the caller.
    """
    if not 0 < expected_bytes <= maximum_bytes or chunk_bytes <= 0:
        raise ValueError("invalid registered receipt bounds")
    result = {"expected_bytes": expected_bytes, "status": "failed", "raw_path": str(path)}
    received = 0
    with path.open("xb") as output:
        try:
            while received <= expected_bytes:
                requested = min(chunk_bytes, expected_bytes + 1 - received)
                try:
                    block = stream.read(requested)
                except IncompleteRead as exc:
                    # Preserve only bounded partial data from the failed read.
                    partial = exc.partial[:requested]
                    output.write(partial)
                    received += len(partial)
                    raise
                if not block:
                    break
                if len(block) > requested:
                    raise ValueError("transport returned more than requested")
                output.write(block)
                received += len(block)
            if received != expected_bytes:
                raise ValueError("received length differs from sealed preflight")
            result["status"] = "complete-receipt-awaiting-format-validation"
        except (OSError, ValueError, IncompleteRead) as exc:
            result.update(error_type=type(exc).__name__, error=str(exc))
    result.update(raw_bytes=path.stat().st_size, raw_sha256=hash_file(path))
    return result
