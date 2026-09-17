from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest
import yaml

brian = pytest.importorskip("brian2")

from smart_robustness.baseline import load_frozen_classic_baseline
from smart_robustness.modeldb_projections import MODELDB_FIRST_ORDER
from smart_robustness.models.direct_visual_thalamus_layer5 import (
    DIRECT_VISUAL_THALAMUS_LAYER5_COMPARTMENT,
    DIRECT_VISUAL_THALAMUS_LAYER5_ID,
    DIRECT_VISUAL_THALAMUS_LAYER5_TARGET,
    DIRECT_VISUAL_THALAMUS_LAYER5_TEMPLATE_ID,
    REGISTERED_DIRECT_VISUAL_THALAMUS_LAYER5_RATIOS,
    direct_visual_thalamus_layer5_record,
    make_direct_visual_thalamus_layer5_sector_builder,
)
from smart_robustness.validation.figure10_search_cycle_spread import (
    build_projection036_variance_sector,
)


def test_registration_and_code_share_the_same_sealed_ratio_grid() -> None:
    study = yaml.safe_load(
        Path("configs/robustness/direct_visual_thalamus_layer5_v1.yaml").read_text()
    )
    assert tuple(study["independent_variable"]["values"]) == (
        REGISTERED_DIRECT_VISUAL_THALAMUS_LAYER5_RATIOS
    )
    assert study["pathway"]["id"] == DIRECT_VISUAL_THALAMUS_LAYER5_ID
    assert study["pathway"]["target_population"] == (
        DIRECT_VISUAL_THALAMUS_LAYER5_TARGET
    )
    assert study["pathway"]["target_compartment"] == (
        DIRECT_VISUAL_THALAMUS_LAYER5_COMPARTMENT
    )


def test_projection_record_changes_only_registered_fields() -> None:
    ratio = 0.125
    template = MODELDB_FIRST_ORDER.by_id(DIRECT_VISUAL_THALAMUS_LAYER5_TEMPLATE_ID)
    record = direct_visual_thalamus_layer5_record(ratio)
    assert record.weight == pytest.approx(float(template.weight) * ratio)
    assert record.asymptotic_weight == pytest.approx(record.weight)
    assert record.id == DIRECT_VISUAL_THALAMUS_LAYER5_ID
    assert record.source_population == template.source_population
    assert record.target_population == DIRECT_VISUAL_THALAMUS_LAYER5_TARGET
    assert record.target_compartment == DIRECT_VISUAL_THALAMUS_LAYER5_COMPARTMENT
    assert record.modifiable is False
    assert record.learning_rule is None
    copied = {
        "kind",
        "source_population",
        "source_compartment",
        "dependency",
        "channel_conductance_mS_cm2",
        "reversal_mV",
        "rise_ms",
        "fall_ms",
        "delay_ms",
        "method",
        "kernel",
        "gate_attributes",
    }
    for name in copied:
        assert getattr(record, name) == getattr(template, name)
    assert {item.name for item in fields(record)} >= copied


@pytest.mark.parametrize("ratio", (-0.1, float("nan"), float("inf"), 1.1))
def test_projection_rejects_unregistered_domain(ratio: float) -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        direct_visual_thalamus_layer5_record(ratio)


def test_ratio_zero_returns_the_exact_unchanged_builder() -> None:
    builder = make_direct_visual_thalamus_layer5_sector_builder(
        strength_ratio=0.0,
        base_builder=build_projection036_variance_sector,
    )
    assert builder is build_projection036_variance_sector


def test_nonzero_builder_adds_one_l5_port_and_one_fixed_projection() -> None:
    brian.start_scope()
    brian.prefs.codegen.target = "numpy"
    baseline = load_frozen_classic_baseline(
        "configs/baselines/classic_smart_calibrated_v1.yaml"
    )
    ratio = 0.125
    builder = make_direct_visual_thalamus_layer5_sector_builder(
        strength_ratio=ratio,
        base_builder=build_projection036_variance_sector,
    )
    sector = builder(conventions=baseline.runtime_conventions(), brian=brian)
    assert set(sector.projections) == {
        record.id
        for record in MODELDB_FIRST_ORDER.projections
        if record.source_population in sector.populations
        and record.target_population in sector.populations
        and record.dependency != "input"
    } | {DIRECT_VISUAL_THALAMUS_LAYER5_ID}
    l5_ports = sector.populations[DIRECT_VISUAL_THALAMUS_LAYER5_TARGET].compiled.synaptic_ports
    assert [port.record_id for port in l5_ports].count(
        DIRECT_VISUAL_THALAMUS_LAYER5_ID
    ) == 1
    assert all(
        DIRECT_VISUAL_THALAMUS_LAYER5_ID
        not in {port.record_id for port in population.compiled.synaptic_ports}
        for name, population in sector.populations.items()
        if name != DIRECT_VISUAL_THALAMUS_LAYER5_TARGET
    )
    projection = sector.projections[DIRECT_VISUAL_THALAMUS_LAYER5_ID]
    template = sector.projections[DIRECT_VISUAL_THALAMUS_LAYER5_TEMPLATE_ID]
    assert np.array_equal(np.asarray(projection.i), np.asarray(template.i))
    assert np.array_equal(np.asarray(projection.j), np.asarray(template.j))
    assert np.asarray(projection.w).max() == pytest.approx(6.0 * ratio)
    assert np.all(np.asarray(projection.modifiable) == 0.0)
    sector.network.run(0.05 * brian.ms)
    assert all(
        np.isfinite(
            np.asarray(
                getattr(population.group, f"v_{compartment}")[:] / brian.mV
            )
        ).all()
        for population in sector.populations.values()
        for compartment in population.compartments
    )
