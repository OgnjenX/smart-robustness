"""Do not hide a bytes/string version interpretation difference."""

from __future__ import annotations

import pytest

from smart_robustness.validation.sst_vip_nwb_metadata import pipeline_versions


def test_text_version_matches_literal_sdk():
    r = pipeline_versions(["name", "pipeline", "version", "1.2.3"])
    assert r["decoded_pipeline_version"] == r["literal_sdk_pipeline_version"] == [1, 2]
    assert r["interpretations_agree"]


def test_bytes_version_discrepancy_is_not_corrected():
    r = pipeline_versions([b"version", b"1.2.3"])
    assert r["decoded_pipeline_version"] == [1, 2]
    assert r["literal_sdk_pipeline_version"] == [0, 0]
    assert not r["interpretations_agree"]
    assert not r["conversion_or_model_execution_performed"]


def test_mixed_key_value_types_preserve_native_exception_fallback():
    r = pipeline_versions(["version", b"1.2.3"])
    assert r["decoded_pipeline_version"] == [1, 2]
    assert r["literal_sdk_pipeline_version"] == [0, 0]
    assert r["literal_sdk_fallback_error_type"] == "TypeError"


@pytest.mark.parametrize(
    "values",
    [
        ["version"],
        [],
        ["version", "1.2", "version", "1.3"],
        ["version", "bad"],
        ["version", "-1.2"],
    ],
)
def test_unresolved_metadata_not_defaulted(values):
    with pytest.raises(ValueError):
        pipeline_versions(values)
