from pathlib import Path

import yaml

from smart_robustness.models.selective import (
    CORTICAL_CELL_CLASSES,
    CORTICAL_EXCITATORY_CELL_CLASSES,
    CORTICAL_INHIBITORY_CELL_CLASSES,
    THALAMIC_RELAY_TRN_CELL_CLASSES,
    make_selective_population_factory,
)

ROOT = Path(__file__).parents[1]


def test_registered_cell_class_sets_are_disjoint_and_complete() -> None:
    assert CORTICAL_EXCITATORY_CELL_CLASSES.isdisjoint(
        CORTICAL_INHIBITORY_CELL_CLASSES
    )
    assert CORTICAL_CELL_CLASSES == (
        CORTICAL_EXCITATORY_CELL_CLASSES | CORTICAL_INHIBITORY_CELL_CLASSES
    )
    assert CORTICAL_CELL_CLASSES.isdisjoint(THALAMIC_RELAY_TRN_CELL_CLASSES)


def test_selective_factory_dispatches_only_registered_cell_classes() -> None:
    calls: list[tuple[str, str]] = []

    def control_factory(*, name, size, params, brian=None):
        calls.append(("control", params["cell_class"]))
        return name, size

    def alternative_factory(*, name, size, params, brian=None):
        calls.append(("alternative", params["cell_class"]))
        return name, size

    factory = make_selective_population_factory(
        alternative_factory,
        {"layer4_excitatory_v1"},
        control_factory=control_factory,
    )
    factory(
        name="l4e",
        size=81,
        params={"cell_class": "layer4_excitatory_v1"},
    )
    factory(name="relay", size=81, params={"cell_class": "thalamic_relay"})
    assert calls == [
        ("alternative", "layer4_excitatory_v1"),
        ("control", "thalamic_relay"),
    ]


def test_localization_study_freezes_seeds_gates_and_thalamic_fit_boundary() -> None:
    study = yaml.safe_load(
        (ROOT / "configs/robustness/neuron_model_localization_v1.yaml").read_text()
    )
    registration = yaml.safe_load(
        (
            ROOT
            / "docs/validation-results/neuron-model-localization-registration-912.yaml"
        ).read_text()
    )
    seeds = study["network_seed_ensemble"]["seeds"]
    assert len(seeds) == len(set(seeds)) == 20
    assert study["figure6_progression_rule"]["minimum_successful_trials"] == 16
    assert study["thalamic_stage"]["sealed_holdout_required"] is True
    assert study["thalamic_stage"]["network_outcomes_as_fit_inputs"] == "prohibited"
    assert registration["knowledge_boundary"][
        "selective_population_outcomes_observed_at_registration"
    ] is False
