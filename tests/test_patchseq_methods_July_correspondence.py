"""Cell/sample conflicts, raw confidence categories and ambiguity remain visible."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / 'scripts/join_patchseq_methods_July.py'
spec = importlib.util.spec_from_file_location('methods_july', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixtures():
    methods = [{'source_row_number': 2, 'canonical_cell_id': 123, 'raw_fields': {
        'cell_id': '123', 'sample_id': 'sample', 'structure': 'VISp5', 'class': 'inhibitory',
        'mapping_quality': 'highly consistent', 'Gouwens_etal_2020': 'True', 'dataset': 'early'}}]
    july = [{'source_row_number': 2, 'canonical_ids': {'cell_specimen_id': 123}, 'raw_fields': {
        'cell_specimen_id': '123', 'transcriptomics_sample_id': 'sample', 'structure': 'VISp5'}}]
    return methods, july


def test_consistent_correspondence_not_confidence_or_eligibility():
    result = module.join(*fixtures())
    row = result['correspondences'][0]
    assert row['consistent_cell_sample_correspondence']
    assert not row['mapping_confidence_assigned'] and not row['specimen_eligibility_established']


@pytest.mark.parametrize('value', ['', 'inconsistent', ' moderately consistent '])
def test_quality_not_selected_or_normalized(value):
    methods, july = fixtures()
    methods[0]['raw_fields']['mapping_quality'] = value
    assert module.join(methods, july)['summary']['raw_mapping_quality_counts_for_consistent_correspondences'] == {value: 1}


def test_unique_cell_sample_conflict():
    methods, july = fixtures()
    other = copy.deepcopy(methods[0])
    other['source_row_number'] = 3
    other['canonical_cell_id'] = 456
    other['raw_fields']['cell_id'] = '456'
    other['raw_fields']['sample_id'] = 'other'
    methods.append(other)
    july[0]['raw_fields']['transcriptomics_sample_id'] = 'other'
    result = module.join(methods, july)
    assert result['summary']['unique_cell_sample_conflicts'] == 1
    assert not result['correspondences'][0]['consistent_cell_sample_correspondence']
    assert result['unreferenced_methods_source_row_numbers'] == []


@pytest.mark.parametrize('key', ['cell', 'sample'])
def test_duplicate_candidates_never_first_selected(key):
    methods, july = fixtures()
    other = copy.deepcopy(methods[0])
    other['source_row_number'] = 3
    if key == 'cell':
        other['raw_fields']['sample_id'] = 'different'
    else:
        other['canonical_cell_id'] = 456
    methods.append(other)
    result = module.join(methods, july)
    row = result['correspondences'][0]
    assert len(row[f'methods_{key}_candidates']) == 2
    assert not row['consistent_cell_sample_correspondence']


def test_whitespace_not_trimmed_and_unmatched_kept():
    methods, july = fixtures()
    july[0]['raw_fields']['transcriptomics_sample_id'] = 'sample '
    result = module.join(methods, july)
    assert result['summary']['July_rows_without_methods_sample'] == 1
    assert not result['correspondences'][0]['consistent_cell_sample_correspondence']


def test_literal_structure_conflict_visible_not_inferred_layer():
    methods, july = fixtures()
    july[0]['raw_fields']['structure'] = 'VISp'
    result = module.join(methods, july)
    assert result['summary']['consistent_correspondence_structure_conflicts'] == 1
    assert result['summary']['consistent_cell_sample_correspondences'] == 1


def test_unreferenced_source_preserved():
    methods, _ = fixtures()
    assert module.join(methods, [])['unreferenced_methods_source_row_numbers'] == [2]
