from pathlib import Path


def test_point_hh_stage3_runner_is_sealed_resumable_and_staged() -> None:
    source = (
        Path(__file__).parents[1] / "scripts/run_cortical_point_hh_stage3.py"
    ).read_text()
    assert "mechanism-cortical-point-hh-figure6-assessment-929.yaml" in source
    assert "stage3_match_mismatch_and_reset_authorized" in source
    assert "create_conserved_cortical_point_hh_population" in source
    assert "CORTICAL_CELL_CLASSES" in source
    assert "_run_stage3_repetition" in source
    assert "--resume-sha256" in source
    assert "refusing to overwrite" in source
    assert "yaml.safe_dump(value, sort_keys=True)" in source
    assert '"stage4_authorized": bool(classic_pass and point_pass)' in source
