import pytest

from smart_robustness.models import (
    available_models,
    available_smart_models,
    create_population,
    get_smart_population_factory,
)


def test_classic_baseline_is_registered() -> None:
    assert available_models() == ("classic_hh",)


def test_complete_smart_registry_separates_control_and_substitutions() -> None:
    assert available_smart_models() == (
        "classic_multicompartment_hh",
        "somatic_adex_literature",
        "somatic_gif_literature",
    )
    assert callable(get_smart_population_factory("somatic_adex_literature"))
    assert callable(get_smart_population_factory("somatic_gif_literature"))


@pytest.mark.parametrize("name", ["adex", "gif", "point_hh", "multicompartment_hh"])
def test_planned_models_are_not_silent_aliases(name: str) -> None:
    with pytest.raises(NotImplementedError):
        create_population(name)
