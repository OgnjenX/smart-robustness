"""Revision comparison retains disagreements without biological promotion."""

from __future__ import annotations

import copy
import csv
import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / 'scripts/compare_patchseq_annotation_revisions.py'
spec = importlib.util.spec_from_file_location('annotation_revisions', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    raw = dict.fromkeys(module.FIELDS, '')
    raw.update(cell_specimen_id='100', donor_id='10', ephys_session_id='1000', structure='VISp5')
    raw['T-type Label'] = 'Sst test'
    raw['MET-type Label'] = 'Sst-MET-test'
    ids = {k: module.canonical(raw[k]) for k in module.IDS}
    july = [{'source_row_number': 2, 'raw_fields': raw, 'canonical_ids': ids}]
    old = {v: raw[k] for k, v in module.PAIRS.items()}
    june = [{'source_row_number': 3, 'raw_fields': old, 'canonical_ids': copy.deepcopy(ids)}]
    return july, june


def test_identical_references_do_not_promote_class_or_confidence():
    result = module.compare_rows(*fixture())
    assert result['summary']['one_cell_and_session_correspondences'] == 1
    assert all(n == 0 for n in result['raw_field_change_counts'].values())
    assert not result['comparisons'][0]['biological_eligibility_established']
    assert result['raw_MET_label_counts'] == {'Sst-MET-test': 1}


@pytest.mark.parametrize('field', list(module.PAIRS))
def test_raw_changes_reported_without_repair(field):
    july, june = fixture()
    july[0]['raw_fields'][field] = 'changed'
    if field in module.IDS:
        july[0]['canonical_ids'][field] = None
    result = module.compare_rows(july, june)
    if field != 'cell_specimen_id':
        assert result['raw_field_change_counts'][field] == 1
    else:
        assert result['summary']['July_rows_without_June_cell'] == 1
    assert result['July_rows'] == july


def test_multiple_June_cell_candidates_retained():
    july, june = fixture()
    second = copy.deepcopy(june[0])
    second['source_row_number'] = 4
    june.append(second)
    result = module.compare_rows(july, june)
    assert result['comparisons'][0]['June_cell_candidates'] == [3, 4]
    assert result['comparisons'][0]['raw_field_differences'] is None
    assert not result['comparisons'][0]['one_cell_and_session_correspondence']


def test_cell_and_session_must_reference_same_source_row():
    july, june = fixture()
    other = copy.deepcopy(june[0])
    other['source_row_number'] = 4
    other['canonical_ids']['cell_specimen_id'] = 101
    other['raw_fields']['cell_specimen_id'] = '101'
    june[0]['canonical_ids']['ephys_session_id'] = 999
    june[0]['raw_fields']['ephys_session_id'] = '999'
    june.append(other)
    result = module.compare_rows(july, june)
    assert result['comparisons'][0]['June_cell_candidates'] == [3]
    assert result['comparisons'][0]['June_session_candidates'] == [4]
    assert not result['comparisons'][0]['one_cell_and_session_correspondence']


def test_unreferenced_June_and_duplicate_July_cells_remain_visible():
    july, june = fixture()
    duplicate = copy.deepcopy(july[0])
    duplicate['source_row_number'] = 4
    july.append(duplicate)
    other = copy.deepcopy(june[0])
    other['source_row_number'] = 5
    other['canonical_ids']['cell_specimen_id'] = 102
    other['canonical_ids']['ephys_session_id'] = 1002
    june.append(other)
    result = module.compare_rows(july, june)
    assert result['July_duplicate_IDs']['cell_specimen_id'] == [100]
    assert result['unreferenced_June_rows_by_cell'] == [5]


def test_CSV_projects_only_whitelisted_fields(tmp_path):
    path = tmp_path / 'annotations.csv'
    header = list(module.FIELDS) + ['medical_conditions']
    raw = fixture()[0][0]['raw_fields']
    with path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerow([raw[k] for k in module.FIELDS] + ['irrelevant'])
    row = module.read_rows(path, header)[0]
    assert set(row['raw_fields']) == set(module.FIELDS)
    assert row['source_row_number'] == 2


@pytest.mark.parametrize('value', ['001', '1.0', ' 1', '١', None, True])
def test_ids_not_normalized(value):
    assert module.canonical(value) is None
