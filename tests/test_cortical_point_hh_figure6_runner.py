from pathlib import Path


def test_point_hh_figure6_runner_uses_registered_factory_and_two_exact_repeats() -> None:
    source = (
        Path(__file__).parents[1] / "scripts/run_cortical_point_hh_figure6.py"
    ).read_text()
    assert "create_conserved_cortical_point_hh_population" in source
    assert "CORTICAL_CELL_CLASSES" in source
    assert 'exact_reruns != 2' in source
    assert '"stage_3_authorized": bool(classic_pass and point_pass)' in source
    assert "network_outcome_used_for_parameter_selection" in source
