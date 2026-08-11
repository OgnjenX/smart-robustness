from __future__ import annotations

from collections import Counter

from smart_robustness.modeldb_projections import MODELDB_FIRST_ORDER
from smart_robustness.models.modeldb112923 import SMART_NML_SHA256


def test_derived_modeldb_projection_catalog_is_integrity_pinned_and_complete() -> None:
    catalog = MODELDB_FIRST_ORDER
    assert catalog.source_sha256 == SMART_NML_SHA256
    assert len(catalog.projections) == 55
    assert len(catalog.external_channels) == 11
    assert Counter(record.kind for record in catalog.projections) == {
        "chemical": 51,
        "gap_junction": 4,
    }


def test_modeldb_topology_preserves_legacy_kernel_semantics() -> None:
    records = MODELDB_FIRST_ORDER.projections
    assert Counter(record.method for record in records) == {
        "connectFromMany": 40,
        "connectFromAll": 10,
        "connectFromOne": 5,
    }
    kernels = tuple(record.kernel for record in records if record.kernel is not None)
    assert Counter(kernel.border_effect for kernel in kernels) == {"wrap": 35, "extend": 5}
    assert Counter(kernel.ring for kernel in kernels) == {False: 32, True: 8}


def test_modeldb_and_supplement_are_not_silently_treated_as_identical() -> None:
    records = MODELDB_FIRST_ORDER.projections
    assert sum(record.source_population.endswith("_v2") for record in records) == 2
    assert sum(record.modifiable for record in records) == 4
    relay_wide = next(
        record
        for record in records
        if record.target_population == "thalamic_relay"
        and record.channel_name == "Layer 6II AMPA WIDE (Fransen et al, 2002)"
    )
    assert relay_wide.weight == 1.5
    assert relay_wide.asymptotic_weight == 0.05
    assert relay_wide.kernel is not None
    assert relay_wide.kernel.border_effect == "wrap"
