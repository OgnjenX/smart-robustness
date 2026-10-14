"""Cell-ID associations cannot promote source annotations or overwrite failed tests."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / 'scripts/build_patchseq_cell_ID_coherence.py'
spec = importlib.util.spec_from_file_location('cell_coherence', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixtures():
    methods = [{'source_row_number': 2, 'canonical_cell_id': 123, 'raw_fields': {
        'structure': 'VISp5', 'mapping_quality': 'inconsistent', 'Gouwens_etal_2020': 'False',
        'class': 'inhibitory', 'dataset': 'early', 'cell_id': '123', 'sample_id': 'P_sample'}}]
    july = [{'source_row_number': 2, 'canonical_ids': {'cell_specimen_id': 123},
             'raw_fields': {'cell_specimen_id': '123', 'transcriptomics_sample_id': 'PS_sample', 'structure': 'VISp5'}}]
    strict = [{'July_source_row_number': 2, 'July_canonical_cell_specimen_id': 123,
               'July_raw_sample_id': 'PS_sample', 'July_raw_structure': 'VISp5',
               'methods_cell_candidates': copy.deepcopy(methods), 'consistent_cell_sample_correspondence': False}]
    names = [{'July_source_row_number': 2, 'July_canonical_cell_specimen_id': 123,
              'July_raw_identifier_fields': {'cell_specimen_id': '123', 'transcriptomics_sample_id': 'PS_sample'},
              'methods_cell_candidates': copy.deepcopy(methods), 'unique_explicit_specimen_name_bridge': False,
              'methods_sample_equals_July_transcriptomics_sample': False}]
    return methods, july, strict, names


def test_raw_annotation_associated_but_failed_flags_and_unknowns_remain():
    result = module.build(*fixtures())
    row = result['records'][0]
    assert row['unique_cell_ID_annotation_association']
    assert row['link_evidence'] == 'canonical-cell-ID-only'
    for key in ('prior_strict_cell_sample_correspondence', 'prior_specimen_name_bridge',
                'RNA_sample_alias_equivalence_established', 'mapping_confidence_assigned', 'specimen_eligibility_established'):
        assert row[key] is False
    assert result['summary']['raw_source_category_counts_for_unique_cell_ID_associations']['mapping_quality'] == {'inconsistent': 1}


@pytest.mark.parametrize('change', ['prior_id', 'prior_row', 'prior_raw', 'missing_test'])
def test_inconsistent_prior_evidence_rejected(change):
    methods, july, strict, names = fixtures()
    if change == 'prior_id':
        names[0]['July_canonical_cell_specimen_id'] = 124
    elif change == 'prior_row':
        strict[0]['methods_cell_candidates'][0]['source_row_number'] = 3
    elif change == 'prior_raw':
        names[0]['methods_cell_candidates'][0]['raw_fields']['mapping_quality'] = 'changed'
    else:
        names = []
    with pytest.raises(ValueError):
        module.build(methods, july, strict, names)


def test_ambiguous_cells_not_first_selected():
    methods, july, strict, names = fixtures()
    extra = copy.deepcopy(methods[0])
    extra['source_row_number'] = 3
    methods.append(extra)
    strict[0]['methods_cell_candidates'] = copy.deepcopy(methods)
    names[0]['methods_cell_candidates'] = copy.deepcopy(methods)
    result = module.build(methods, july, strict, names)
    assert len(result['records'][0]['methods_candidates']) == 2
    assert not result['records'][0]['unique_cell_ID_annotation_association']
    assert result['summary']['raw_source_category_counts_for_unique_cell_ID_associations']['mapping_quality'] == {}


@pytest.mark.parametrize('structure,expected', [('', None), ('VISp', False), ('VISp5', True)])
def test_missing_or_different_structure_not_inferred(structure, expected):
    methods, july, strict, names = fixtures()
    methods[0]['raw_fields']['structure'] = structure
    names[0]['methods_cell_candidates'] = copy.deepcopy(methods)
    assert module.build(methods, july, strict, names)['records'][0]['literal_structure_agrees'] is expected


@pytest.mark.parametrize('identity', [True, -1, '123', 123.0])
def test_noncanonical_ID_types_rejected(identity):
    with pytest.raises(ValueError):
        module.validate_id(identity)
