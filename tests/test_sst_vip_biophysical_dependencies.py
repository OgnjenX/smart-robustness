"""Native resource matching remains identity-bounded, not an equation proof."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def collector(monkeypatch):
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    spec = importlib.util.spec_from_file_location(
        "dependency_collector_test", Path("scripts/collect_sst_vip_biophysical_dependencies.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model():
    params = {"id": 7, "well_known_file_type": {"name": "NeuronalModelParameters"}}
    return {
        "id": 1,
        "specimen_id": 2,
        "specimen": {"id": 2},
        "neuronal_model_template": {
            "id": 3,
            "name": "Biophysical - all active",
            "well_known_files": [],
        },
        "well_known_files": [params],
    }


def test_exact_native_identity(collector):
    r = model()
    assert collector.validate_batch(
        {"success": True, "total_rows": 1, "msg": [r]}, [1], {1: r}
    ) == [r]


@pytest.mark.parametrize(
    "change",
    [
        {"specimen_id": 99},
        {"specimen": {"id": 99}},
        {"well_known_files": []},
        {"neuronal_model_template": {"id": 99, "name": "wrong"}},
    ],
)
def test_changed_identity_rejects(collector, change):
    expected = model()
    with pytest.raises(ValueError):
        collector.validate_batch(
            {"success": True, "total_rows": 1, "msg": [model() | change]}, [1], {1: expected}
        )


def test_no_missing_mechanism_substitution_or_false_validation(collector):
    r = model()
    result = collector.mechanism_candidates(r, {"genome": [{"mechanism": "NaTs"}]})
    assert result["missing_mechanisms"] == ["NaTs"]
    assert not result["channel_equations_validated"] and not result["axon_type_resolved"]


def test_duplicate_and_wrong_type_mechanism_resources(collector):
    r = model()
    file = {
        "id": 8,
        "path": "/source/NaTs.mod",
        "well_known_file_type": {"name": "BiophysicalModelDescription"},
    }
    r["neuronal_model_template"]["well_known_files"] = [file, file | {"id": 9}]
    result = collector.mechanism_candidates(r, {"genome": [{"mechanism": "NaTs"}]})
    assert result["ambiguous_mechanisms"] == ["NaTs"]
    r["neuronal_model_template"]["well_known_files"] = [
        file | {"well_known_file_type": {"name": "Other"}}
    ]
    assert collector.mechanism_candidates(r, {"genome": [{"mechanism": "NaTs"}]})[
        "missing_mechanisms"
    ] == ["NaTs"]
