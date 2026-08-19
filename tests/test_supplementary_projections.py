from __future__ import annotations

from collections import Counter
from dataclasses import replace

import pytest

from smart_robustness.projections import (
    SUPPLEMENTARY_TABLE3,
    CatalogValidationError,
    ConnectionKind,
    Receptor,
    TopologyKind,
    VerificationStatus,
    _validate_record,
    load_projection_catalog,
    serialize_projection_catalog,
)


def test_catalog_has_all_55_nonblank_records_and_stable_ids() -> None:
    catalog = SUPPLEMENTARY_TABLE3
    assert catalog.expected_record_count == 55
    assert len(catalog.records) == 55
    ids = [record.id for record in catalog.records]
    assert ids == sorted(ids)
    assert len(ids) == len(set(ids))
    assert all(record.target.compartment is not None for record in catalog.records)


def test_record_kinds_and_target_table_counts_match_source_audit() -> None:
    assert Counter(record.kind for record in SUPPLEMENTARY_TABLE3.records) == {
        ConnectionKind.CHEMICAL: 49,
        ConnectionKind.EXTERNAL_INPUT: 2,
        ConnectionKind.GAP_JUNCTION: 4,
    }
    assert Counter(record.target.population for record in SUPPLEMENTARY_TABLE3.records) == {
        "thalamic_relay": 7,
        "thalamic_nonspecific": 5,
        "thalamic_matrix": 4,
        "thalamic_interneuron": 5,
        "trn": 7,
        "layer6ii_excitatory_v1": 2,
        "layer6i_excitatory_v1": 4,
        "layer5_excitatory_v1": 5,
        "layer4_inhibitory_v1": 5,
        "layer4_excitatory_v1": 4,
        "layer23_inhibitory_v1": 3,
        "layer23_excitatory_v1": 4,
    }


def test_units_and_kind_specific_constraints() -> None:
    for record in SUPPLEMENTARY_TABLE3.records:
        parsed = record.parsed
        assert parsed.conductance_pS is None or parsed.conductance_pS >= 0
        assert parsed.weight_density_1e6_cm2 is None or parsed.weight_density_1e6_cm2 >= 0
        assert parsed.delay_ms is None or parsed.delay_ms >= 0
        assert parsed.rise_ms is None or parsed.rise_ms >= 0
        assert parsed.fall_ms is None or parsed.fall_ms >= 0
        if parsed.topology.kind is TopologyKind.GAUSSIAN:
            assert parsed.topology.sigma is not None and parsed.topology.sigma > 0
        else:
            assert parsed.topology.sigma is None
        if record.kind is ConnectionKind.EXTERNAL_INPUT:
            assert parsed.receptor is Receptor.INPUT
            assert parsed.weight_density_1e6_cm2 is None
        elif record.kind is ConnectionKind.GAP_JUNCTION:
            assert parsed.receptor is Receptor.GAP_JUNCTION
            assert parsed.reversal_mV is None
            assert parsed.weight_density_1e6_cm2 is None
            assert parsed.delay_ms is None
            assert parsed.rise_ms is None
            assert parsed.fall_ms is None
        else:
            assert parsed.receptor not in {Receptor.INPUT, Receptor.GAP_JUNCTION}


def test_known_relay_plastic_synapse_spot_check() -> None:
    record = SUPPLEMENTARY_TABLE3.by_id("relay.distal_dendrite.from_l6ii.ampa_plastic")
    assert record.raw.weight_density_1e6_cm2 == "1.5(0.05, 0.1)"
    assert record.parsed.receptor is Receptor.AMPA
    assert record.parsed.conductance_pS == pytest.approx(1.0)
    assert record.parsed.topology.sigma == pytest.approx(1.3)
    assert record.parsed.delay_ms == pytest.approx(2.0)
    assert (record.parsed.rise_ms, record.parsed.fall_ms) == (2.0, 7.0)
    assert record.parsed.plasticity is not None
    assert record.parsed.plasticity.max_weight_density_1e6_cm2 == pytest.approx(1.5)
    assert record.parsed.plasticity.baseline_weight_density_1e6_cm2 == pytest.approx(0.05)
    assert record.parsed.plasticity.learning_rate == pytest.approx(0.1)


def test_known_depletion_and_gap_junction_spot_checks() -> None:
    depleted = SUPPLEMENTARY_TABLE3.by_id("l4_e.proximal_dendrite.from_l6i.ampa_depletable")
    assert depleted.raw.weight_density_1e6_cm2 == "(#) 0.7 (1, 400)"
    assert depleted.parsed.weight_density_1e6_cm2 == pytest.approx(0.7)
    assert depleted.parsed.depletion is not None
    assert depleted.parsed.depletion.epsilon == pytest.approx(1.0)
    assert depleted.parsed.depletion.tau_ms == pytest.approx(400.0)

    gap = SUPPLEMENTARY_TABLE3.by_id("trn.distal_dendrite.from_trn_distal.gap")
    assert gap.raw.conductance_pS == "0.010"
    assert gap.parsed.conductance_pS == pytest.approx(0.01)
    assert gap.source.compartment == "distal_dendrite"
    assert gap.target.compartment == "distal_dendrite"


def test_missing_source_cells_remain_explicit_nulls() -> None:
    external = SUPPLEMENTARY_TABLE3.by_id("matrix.proximal_dendrite.from_input.input")
    assert external.raw.weight_density_1e6_cm2 is None
    assert external.raw.delay_ms is None
    assert external.raw.rise_fall_ms is None
    assert external.parsed.weight_density_1e6_cm2 is None
    assert external.parsed.delay_ms is None
    assert external.parsed.rise_ms is None
    assert external.parsed.fall_ms is None


def test_source_anomalies_are_not_silently_corrected() -> None:
    anomalous_receptor = SUPPLEMENTARY_TABLE3.by_id(
        "l4_i.proximal_dendrite.from_l6i.unknown_depletable"
    )
    assert anomalous_receptor.raw.receptor == "N"
    assert anomalous_receptor.parsed.receptor is Receptor.UNKNOWN
    assert anomalous_receptor.verification.status is VerificationStatus.AMBIGUOUS

    anomalous_reversal = SUPPLEMENTARY_TABLE3.by_id("l5_e.distal_dendrite.from_nonspecific.nmda")
    assert anomalous_reversal.raw.reversal_mV == "-70"
    assert anomalous_reversal.parsed.reversal_mV == pytest.approx(-70)
    assert anomalous_reversal.verification.status is VerificationStatus.AMBIGUOUS


def test_every_record_has_a_verification_audit_trail() -> None:
    for record in SUPPLEMENTARY_TABLE3.records:
        assert set(record.verification.source_forms) == {"docx", "html", "txt"}
        if record.verification.status is VerificationStatus.AMBIGUOUS:
            assert record.verification.notes
    assert (
        sum(
            record.verification.status is VerificationStatus.AMBIGUOUS
            for record in SUPPLEMENTARY_TABLE3.records
        )
        == 4
    )


def test_serialization_is_deterministic_and_round_trips(tmp_path) -> None:
    first = serialize_projection_catalog(SUPPLEMENTARY_TABLE3)
    second = serialize_projection_catalog(SUPPLEMENTARY_TABLE3)
    assert first == second
    assert first.index("l23_e.proximal_dendrite.from_l23_e.ampa") < first.index(
        "trn.soma.from_trn.gaba"
    )

    serialized_path = tmp_path / "catalog.yaml"
    serialized_path.write_text(first, encoding="utf-8")
    reloaded = load_projection_catalog(serialized_path)
    assert reloaded == SUPPLEMENTARY_TABLE3


def test_validator_rejects_unknown_receptor_without_ambiguity_note() -> None:
    source = SUPPLEMENTARY_TABLE3.by_id("l4_i.proximal_dendrite.from_l6i.unknown_depletable")
    invalid = replace(
        source,
        verification=replace(
            source.verification,
            status=VerificationStatus.CROSS_CHECKED,
            notes=(),
        ),
    )
    with pytest.raises(CatalogValidationError, match="unknown receptor"):
        _validate_record(invalid)
