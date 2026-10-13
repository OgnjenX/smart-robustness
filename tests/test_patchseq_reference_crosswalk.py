"""Exact name alignment is not direct identifier equivalence or cell-class proof."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / 'scripts/crosswalk_patchseq_reference.py'
spec = importlib.util.spec_from_file_location('reference_crosswalk', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def row(number, fields):
    return {'source_row_number': number, 'cells': {k: {'storage_type': 's', 'literal_text': v,
                                                     'formula_present': False} for k, v in fields.items()}}


def fixture():
    provenance = row(2, {'Taxonomy id': 'reference', 'Description': 'mouse source', 'Reference': 'paper'})
    leaf = row(3, {'taxonomy_id': 'reference', 'cell_set_ accession': 'new-ID',
                   'cell_set_preferred_alias': 'Sst label', 'species': 'mouse', 'modality': 'RNA'})
    records = {'reference_sheets': {'taxonomy ids': {'rows': [provenance]},
                                    'cell set nomenclature': {'rows': [leaf]}}}
    july = [{'raw_fields': {'T-type Accession ID': 'old-ID', 'T-type Label': 'Sst label'}}]
    return records, july, 'reference', 'mouse source', 'paper'


def test_alignment_explicitly_not_ID_or_biological_equivalence():
    result = module.align(*fixture())
    alignment = result['alignments'][0]
    assert alignment['exact_name_alignment_supported']
    assert alignment['reference_accession'] == 'new-ID'
    assert not alignment['direct_accession_equality']
    assert not alignment['historical_identifier_equivalence_confirmed']
    assert not alignment['class_membership_established']
    assert not alignment['biological_eligibility_established']


def test_other_taxonomy_alias_not_selected():
    args = list(fixture())
    other = copy.deepcopy(args[0]['reference_sheets']['cell set nomenclature']['rows'][0])
    other['source_row_number'] = 4
    other['cells']['taxonomy_id']['literal_text'] = 'different-taxonomy'
    args[0]['reference_sheets']['cell set nomenclature']['rows'].append(other)
    result = module.align(*args)
    assert [r['source_row_number'] for r in result['alignments'][0]['reference_candidates']] == [3]


@pytest.mark.parametrize('fault', ['duplicate-candidate', 'duplicate-accession', 'cached-formula', 'numeric-ID', 'conflicting-legacy-label'])
def test_conflicts_not_silently_repaired(fault):
    args = list(fixture())
    leaves = args[0]['reference_sheets']['cell set nomenclature']['rows']
    if fault in ('duplicate-candidate', 'duplicate-accession'):
        other = copy.deepcopy(leaves[0])
        other['source_row_number'] = 4
        if fault == 'duplicate-accession':
            other['cells']['cell_set_preferred_alias']['literal_text'] = 'different label'
        leaves.append(other)
    elif fault == 'cached-formula':
        leaves[0]['cells']['cell_set_ accession']['formula_present'] = True
    elif fault == 'numeric-ID':
        leaves[0]['cells']['cell_set_ accession']['storage_type'] = 'n'
    else:
        args[1].append({'raw_fields': {'T-type Accession ID': 'old-ID', 'T-type Label': 'other'}})
    assert not module.align(*args)['alignments'][0]['exact_name_alignment_supported']


@pytest.mark.parametrize('label', ['sst label', ' Sst label', 'Sst', 'Sst label extra', ''])
def test_alias_not_normalized_or_substring_matched(label):
    args = list(fixture())
    args[1][0]['raw_fields']['T-type Label'] = label
    assert not module.align(*args)['alignments'][0]['exact_name_alignment_supported']


@pytest.mark.parametrize('field', ['Description', 'Reference', 'Taxonomy id'])
def test_provenance_must_match_exactly(field):
    args = list(fixture())
    args[0]['reference_sheets']['taxonomy ids']['rows'][0]['cells'][field]['literal_text'] = 'changed'
    with pytest.raises(ValueError, match='provenance'):
        module.align(*args)


def test_multiple_legacy_IDs_for_target_reported():
    args = list(fixture())
    args[1].append({'raw_fields': {'T-type Accession ID': 'another-old-ID', 'T-type Label': 'Sst label'}})
    result = module.align(*args)
    assert result['reference_targets_with_multiple_legacy_accessions'] == {'new-ID': ['another-old-ID', 'old-ID']}
