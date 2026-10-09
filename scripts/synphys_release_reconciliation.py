"""Synthetic-tested reconciliation helpers; no database execution entry point.

Real source queries require a separate sealed registration and independently
verified medium acquisition. These helpers do not authorize baseline joins.
"""

from __future__ import annotations

import hashlib
import math
import sqlite3
import struct
from pathlib import Path


def stream_sha256(path: Path, chunk_bytes: int = 1048576) -> str:
    """Hash large releases without allocating the complete database."""
    if not isinstance(chunk_bytes, int) or isinstance(chunk_bytes, bool) or chunk_bytes <= 0:
        raise ValueError("positive integer chunk size required")
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            checksum.update(chunk)
    return checksum.hexdigest()


def same_value(left, right):
    """Exact typed values; NaN matches NaN, never a missing value.

    Retain signed zero and reject bool/int coercion. No numeric tolerance,
    rounding, unit conversion, string normalization or null imputation.
    """
    if type(left) is not type(right):
        return False
    if isinstance(left, float):
        if math.isnan(left) or math.isnan(right):
            return math.isnan(left) and math.isnan(right)
        return struct.pack("!d", left) == struct.pack("!d", right)
    if left is None or isinstance(left, (int, str, bytes)):
        return left == right
    raise ValueError("unsupported source value type")


def index_rows(rows, columns):
    if not columns or len(set(columns)) != len(columns) or "id" not in columns:
        raise ValueError("unique columns including primary ID required")
    indexed = {}
    for row in rows:
        if set(row) != set(columns):
            raise ValueError("row projection differs from registered columns")
        identity = row["id"]
        if type(identity) is not int or identity <= 0 or identity in indexed:
            raise ValueError("invalid or duplicate primary ID")
        indexed[identity] = row
    return indexed


def compare_rows(small_rows, medium_rows, columns):
    """Retain all mismatches, independent of row order; never select a best fit."""
    small = index_rows(small_rows, columns)
    medium = index_rows(medium_rows, columns)
    differences = []
    for identity in sorted(small.keys() | medium.keys()):
        if identity not in small or identity not in medium:
            differences.append({"id": identity, "kind": "missing-row",
                                "missing_from": "small" if identity not in small else "medium"})
            continue
        for column in columns:
            if not same_value(small[identity][column], medium[identity][column]):
                differences.append({"id": identity, "kind": "changed-value", "column": column,
                                    "small": small[identity][column],
                                    "medium": medium[identity][column]})
    return {"equal": not differences, "small_rows": len(small), "medium_rows": len(medium),
            "differences": differences}


def make_authorizer(columns):
    """Freeze the caller's exact projection; deny functions, writes and traces.

    SQLite emits an empty-column READ for INTEGER PRIMARY KEY projections.
    Permit that callback only for the explicitly registered tables; it does
    not grant access to any named column outside the projection.
    """
    allowed = {table: frozenset(projection) for table, projection in columns.items()}
    if not allowed or any("id" not in projection for projection in allowed.values()):
        raise ValueError("every registered table requires primary ID")

    def authorize(action, table, column, database, trigger):
        if action == sqlite3.SQLITE_READ:
            permitted = table in allowed and (column == "" or column in allowed[table])
            return sqlite3.SQLITE_OK if permitted else sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK if action == sqlite3.SQLITE_SELECT else sqlite3.SQLITE_DENY

    return authorize
