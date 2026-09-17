from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]


def _yaml(path: str):
    return yaml.safe_load((ROOT / path).read_text())


def test_neuron_model_robustness_release_keeps_classic_control_frozen() -> None:
    release = _yaml("configs/releases/neuron_model_robustness_v1.yaml")
    assert release["classic_control"]["tag"] == "classic-smart-calibrated-v1.0.0"
    assert release["classic_control"]["modified_by_phase"] is False
    assert release["outcome_classes"]["exact_survival"] == ["classic_control"]
    assert "post-2008 anatomy additions" in release["scope"]["excluded"]


def test_phase_synthesis_separates_exact_and_qualitative_survival() -> None:
    result = _yaml(
        "docs/validation-results/neuron-model-mechanism-phase-assessment-936.yaml"
    )
    decision = result["decision"]
    assert decision["neuron_model_comparison_complete"] is True
    assert decision["compartment_mechanism_localization_complete"] is True
    assert decision["failed_arms_closed_without_network_guided_tuning"] is True
    assert decision["classic_baseline_remains_frozen"] is True
    assert result["registered_mechanism_results"]["layer5_distal_nak_disabled"][
        "mismatch_nonspecific_events"
    ] == 8
    assert result["registered_mechanism_results"]["expanded_layer5_branched"][
        "mismatch_nonspecific_events"
    ] == 6
    assert "Coarse ART-like" in result["cross-study_synthesis"][
        "qualitative_art_robustness"
    ]
