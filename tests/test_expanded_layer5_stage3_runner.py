from pathlib import Path


def test_expanded_layer5_stage3_runner_is_sealed_resumable_and_staged() -> None:
    source = (
        Path(__file__).parents[1] / "scripts/run_expanded_layer5_stage3.py"
    ).read_text()
    assert "mechanism-expanded-layer5-figure6-assessment-934.yaml" in source
    assert "stage3_match_mismatch_and_reset_authorized" in source
    assert "create_expanded_layer5_population" in source
    assert "{LAYER5_CELL_CLASS}" in source
    assert "_run_stage3_repetition" in source
    assert "--resume-sha256" in source
    assert "refusing to overwrite" in source
    assert "yaml.safe_dump(value, sort_keys=True)" in source
    assert '"stage4_authorized": bool(classic_pass and expanded_pass)' in source
