import pytest

from smart_robustness.models import available_models, create_population


def test_classic_baseline_is_registered() -> None:
    assert available_models() == ("classic_hh",)


@pytest.mark.parametrize("name", ["adex", "gif", "point_hh", "multicompartment_hh"])
def test_planned_models_are_not_silent_aliases(name: str) -> None:
    with pytest.raises(NotImplementedError):
        create_population(name)

