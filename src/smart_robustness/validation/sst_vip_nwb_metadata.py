"""Metadata interpretation remains separate from native numerical reproduction."""

from __future__ import annotations


def text_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    raise TypeError("pipeline metadata must contain text or bytes")


def pipeline_versions(values):
    """Preserve literal SDK comparison and decoded metadata as separate evidence.

    No conversion multiplier is chosen by this function. The pinned SDK falls
    back to (0, 0) on an unassigned/invalid version; that is recorded, not fixed.
    """
    if len(values) % 2:
        raise ValueError("generated_by metadata must contain key/value pairs")
    entries = [
        [text_value(k), text_value(v)] for k, v in zip(values[::2], values[1::2], strict=True)
    ]
    decoded_versions = [v for k, v in entries if k == "version"]
    if len(decoded_versions) != 1:
        raise ValueError("pipeline version missing or ambiguous")
    parts = decoded_versions[0].split(".")
    if len(parts) < 2:
        raise ValueError("pipeline version has no major/minor pair")
    decoded = [int(parts[0]), int(parts[1])]
    if min(decoded) < 0:
        raise ValueError("pipeline version must be nonnegative")
    literal = [0, 0]
    literal_error = None
    try:
        version = None
        for i in range(len(values)):
            if values[i] == "version":
                version = values[i + 1]
                break
        native_parts = version.split(".")
        if len(native_parts) >= 2:
            literal = [int(native_parts[0]), int(native_parts[1])]
    except (AttributeError, TypeError, ValueError, IndexError) as exc:
        literal_error = type(exc).__name__
    return {
        "decoded_entries": entries,
        "decoded_pipeline_version": decoded,
        "literal_sdk_pipeline_version": literal,
        "literal_sdk_fallback_error_type": literal_error,
        "interpretations_agree": decoded == literal,
        "conversion_or_model_execution_performed": False,
    }
