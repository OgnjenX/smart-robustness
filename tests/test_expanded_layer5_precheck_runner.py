from pathlib import Path


def test_expanded_layer5_precheck_runner_stays_network_blind() -> None:
    source = (
        Path(__file__).parents[1] / "scripts/run_expanded_layer5_prechecks.py"
    ).read_text()
    assert "create_expanded_layer5_population" in source
    assert "make_selective_population_factory" in source
    assert "EXPANDED_LAYER5_TOPOLOGY" in source
    assert "network_outcomes_observed\": False" in source
    assert "run_figure6" not in source
    assert "run_match_mismatch" not in source
