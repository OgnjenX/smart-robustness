from pathlib import Path


def test_expanded_layer5_figure6_runner_is_sealed_resumable_and_staged() -> None:
    source = (
        Path(__file__).parents[1] / "scripts/run_expanded_layer5_figure6.py"
    ).read_text()
    assert "create_expanded_layer5_population" in source
    assert "{LAYER5_CELL_CLASS}" in source
    assert 'exact_reruns != 2' in source
    assert "_load_resumable_arms" in source
    assert "Artifact 933 does not authorize Figure 6" in source
    assert '"stage_3_authorized": bool(classic_pass and expanded_pass)' in source
    assert "network_outcome_used_for_parameter_selection" in source
    assert "run_figure7" not in source
    assert "run_figure10" not in source
