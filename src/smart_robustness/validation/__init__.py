"""Source-backed acceptance targets for the classic SMART baseline."""

from .targets import (
    CLASSIC_SMART_TARGETS,
    EvidenceClass,
    ValidationTarget,
    get_validation_target,
)

__all__ = [
    "CLASSIC_SMART_TARGETS",
    "EvidenceClass",
    "ValidationTarget",
    "get_validation_target",
]
