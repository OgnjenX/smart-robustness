"""Raw-author target labels are neither RNA alias proofs nor qualified cells."""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

scripts = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(scripts))
try:
    spec = importlib.util.spec_from_file_location('source_candidates', scripts/'build_patchseq_source_candidates.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
finally:
    sys.path.pop(0)


def inputs():
    reg = yaml.safe_load(module.REG.read_text())
    raw = {'structure': 'VISp5', 'cell_specimen_id': '42'}
    ledger = {'specimens': [{'July_source_row_number': 2, 'raw_fields': raw,
        'canonical_ids': {'cell_specimen_id': 42}, 'source_core_inhibitory_group': 'Sst',
        'one_agreeing_source_reference_leaf': True}]}
    coherence = {'records': [{'July_source_row_number': 2, 'canonical_cell_id': 42, 'July_raw_fields': copy.deepcopy(raw),
        'methods_candidates': [{'raw_fields': {'class': 'inhibitory', 'Gouwens_etal_2020': 'True', 'mapping_quality': 'highly consistent'}}],
        'unique_cell_ID_annotation_association': True, 'literal_structure_agrees': True,
        'prior_strict_cell_sample_correspondence': False, 'prior_specimen_name_bridge': False}]}
    path = 'sub-1/sub-1_ses-7_icephys.nwb'
    split = {'records': [{'July_candidate_source_rows': [2], 'group_id': 'original-full-catalog-group',
        'partition': 'held-out', 'group_keys': ['author_cell:42'], 'file_path': path, 'unresolved': []}]}
    association = {'source_cell_specimen_id': '42', 'July_source_row_number': 2, 'file_path': path}
    return [reg, ledger, coherence, split, association]


def test_source_candidate_inherits_partition_but_never_qualifies_or_repairs_names():
    result = module.build(*inputs())
    row = result['records'][0]
    assert row['source_annotation_candidate'] is True
    assert row['frozen_partition'] == 'held-out' and row['frozen_group_id'] == 'original-full-catalog-group'
    assert row['recording_content_session_author_cell_provenance_verified'] is True
    assert row['prior_strict_cell_sample_correspondence'] is row['prior_specimen_name_bridge'] is False
    assert row['RNA_alias_equivalence_established'] is row['mapping_confidence_assigned'] is row['specimen_eligibility_established'] is False
    assert len(result['strata']) == 12 and result['summary']['qualified_specimens'] == 0


@pytest.mark.parametrize('case', ['quality', 'anatomy_conflict', 'missing', 'outside_layer', 'reference', 'group_unresolved'])
def test_nonpass_retained_with_specific_reasons(case):
    values = inputs()
    if case == 'quality':
        values[2]['records'][0]['methods_candidates'][0]['raw_fields']['mapping_quality'] = 'moderately consistent'
    elif case == 'anatomy_conflict':
        values[2]['records'][0]['literal_structure_agrees'] = False
    elif case == 'missing':
        values[2]['records'][0]['methods_candidates'] = []
    elif case == 'outside_layer':
        values[1]['specimens'][0]['raw_fields']['structure'] = 'VISp4'
        values[2]['records'][0]['July_raw_fields']['structure'] = 'VISp4'
    elif case == 'reference':
        values[1]['specimens'][0]['one_agreeing_source_reference_leaf'] = False
    else:
        values[3]['records'][0]['unresolved'] = ['unresolved-identity']
    result = module.build(*values)
    assert len(result['records']) == 1
    assert result['records'][0]['source_annotation_candidate'] is False
    assert result['records'][0]['candidate_failures_or_unknowns']
    assert result['summary']['qualified_specimens'] == 0


def test_unverified_recording_is_unknown_not_copied_from_pilot():
    values = inputs()
    values[4]['source_cell_specimen_id'] = '99'
    row = module.build(*values)['records'][0]
    assert row['source_annotation_candidate'] is True
    assert row['recording_content_session_author_cell_provenance_verified'] is None


def test_source_identity_disagreement_is_an_engineering_error():
    values = inputs()
    values[2]['records'][0]['canonical_cell_id'] = 99
    with pytest.raises(ValueError):
        module.build(*values)
