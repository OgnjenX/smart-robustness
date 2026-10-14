"""Specimen source annotations retain uncertainty and never establish eligibility."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / 'scripts/build_patchseq_specimen_source_ledger.py'
spec = importlib.util.spec_from_file_location('specimen_ledger', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    july = [{'source_row_number': 2, 'raw_fields': {'T-type Accession ID': 'type-ID',
              'T-type Label': 'type label', 'structure': 'VISp5', 'full_genotype': 'Vip targeting'},
             'canonical_ids': {'cell_specimen_id': 100, 'donor_id': 10, 'ephys_session_id': 1000}}]
    types = [{'legacy_accession': 'type-ID', 'legacy_label': 'type label', 'source_leaf_identity_agrees': True,
              'exclusive_named_inhibitory_subgroup': 'Sst', 'source_groups': ['GABAergic', 'Sst']}]
    identity = [{'asset_id': 'a', 'annotation_candidates': [{'canonical_ids': {'cell_specimen_id': 100}}]}]
    references = [{'asset_id': 'a', 'source_cell_ID_agrees': True}]
    return july, types, identity, references


def test_source_core_annotation_does_not_impute_confidence_or_eligibility():
    result = module.build(*fixture())
    row = result['specimens'][0]
    assert row['source_core_inhibitory_group'] == 'Sst'
    assert row['raw_fields']['full_genotype'] == 'Vip targeting'
    assert row['June_official_manifest_cell_reference_present']
    assert row['transcriptomic_mapping_confidence'] is None
    assert not row['mapping_confidence_available_in_release']
    assert not row['internal_recording_identity_confirmed']
    assert not row['isolated_physiology_validated']
    assert not row['specimen_eligibility_established']
    assert result['summary']['eligible_specimens'] == 0


@pytest.mark.parametrize('fault', ['duplicate-candidates', 'missing-type', 'wrong-label', 'uncovered-group', 'unverified-leaf'])
def test_ambiguous_or_missing_type_evidence_not_repaired(fault):
    args = list(fixture())
    if fault == 'duplicate-candidates':
        args[1].append(copy.deepcopy(args[1][0]))
    elif fault == 'missing-type':
        args[1] = []
    elif fault == 'wrong-label':
        args[0][0]['raw_fields']['T-type Label'] = 'changed'
    elif fault == 'uncovered-group':
        args[1][0]['exclusive_named_inhibitory_subgroup'] = None
        args[1][0]['source_groups'] = ['GABAergic']
    else:
        args[1][0]['source_leaf_identity_agrees'] = False
    result = module.build(*args)
    assert result['specimens'][0]['source_core_inhibitory_group'] is None
    assert result['summary']['unassigned_source_core_groups'] == 1
    assert result['specimens'][0]['raw_fields'] == args[0][0]['raw_fields']


def test_uncovered_manifest_does_not_change_source_group_or_qualify_specimen():
    args = list(fixture())
    args[3][0]['source_cell_ID_agrees'] = False
    row = module.build(*args)['specimens'][0]
    assert row['source_core_inhibitory_group'] == 'Sst'
    assert not row['June_official_manifest_cell_reference_present']
    assert not row['specimen_eligibility_established']


def test_ambiguous_manifest_parent_rejected():
    args = list(fixture())
    args[2][0]['annotation_candidates'].append(copy.deepcopy(args[2][0]['annotation_candidates'][0]))
    with pytest.raises(ValueError, match='ambiguous'):
        module.build(*args)
