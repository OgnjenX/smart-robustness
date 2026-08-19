from __future__ import annotations

import pytest

from smart_robustness.validation import (
    CLASSIC_SMART_TARGETS,
    EvidenceClass,
    get_validation_target,
)


def test_target_ids_are_unique_and_source_backed() -> None:
    ids = [target.id for target in CLASSIC_SMART_TARGETS]
    assert len(ids) == len(set(ids))
    assert all(target.source for target in CLASSIC_SMART_TARGETS)
    assert all(target.evidence for target in CLASSIC_SMART_TARGETS)


def test_match_mismatch_target_is_directional_not_false_numeric_precision() -> None:
    target = get_validation_target("fig7_match_mismatch_arousal")
    assert not target.numeric_targets
    assert EvidenceClass.STRUCTURAL in target.evidence
    assert EvidenceClass.QUALITATIVE in target.evidence
    assert EvidenceClass.APPROXIMATE_NUMERIC not in target.evidence


def test_figure14_preserves_both_published_middle_bands() -> None:
    target = get_validation_target("fig14_match_mismatch_spectra")
    assert target.protocol["middle_band_caption_hz"] == (8.0, 20.0)
    assert target.protocol["middle_band_methods_hz"] == (8.0, 10.0)


def test_target_mappings_are_immutable() -> None:
    target = get_validation_target("fig8_relay_tonic_burst")
    with pytest.raises(TypeError):
        target.protocol["current_injection_nA"] = 1.0  # type: ignore[index]


def test_unknown_target_reports_known_ids() -> None:
    with pytest.raises(KeyError, match="fig7_match_mismatch_arousal"):
        get_validation_target("not-a-target")
