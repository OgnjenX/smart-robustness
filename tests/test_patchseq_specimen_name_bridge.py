"""A separate exact named-field crosswalk cannot rewrite failed sample equality."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / 'scripts/check_patchseq_specimen_name_bridge.py'
spec = importlib.util.spec_from_file_location('name_bridge', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixtures():
    names = [{'source_row_number': 2, 'canonical_cell_specimen_id': 123, 'raw_fields': {
        'cell_specimen_id': '123', 'cell_specimen_name': 'P8S4_name', 'transcriptomics_sample_id': 'PS_file'}}]
    july = [{'source_row_number': 2, 'canonical_ids': {'cell_specimen_id': 123}, 'raw_fields': {
        'cell_specimen_id': '123', 'transcriptomics_sample_id': 'PS_file'}}]
    methods = [{'source_row_number': 2, 'canonical_cell_id': 123, 'raw_fields': {'sample_id': 'P8S4_name', 'mapping_quality': 'inconsistent'}}]
    return names, july, methods


def test_exact_bridge_preserves_failed_transcriptomic_sample_test():
    result = module.bridge(*fixtures())
    assert result['bridges'][0]['unique_explicit_specimen_name_bridge']
    assert result['bridges'][0]['methods_sample_equals_July_transcriptomics_sample'] is False
    assert result['summary']['raw_quality_counts_for_unique_name_bridges'] == {'inconsistent': 1}
    assert not result['bridges'][0]['mapping_confidence_assigned']


@pytest.mark.parametrize('name', ['', 'P8S4_name ', 'p8s4_name', 'P8S4'])
def test_missing_or_inexact_names_not_repaired(name):
    names, july, methods = fixtures()
    names[0]['raw_fields']['cell_specimen_name'] = name
    assert not module.bridge(names, july, methods)['bridges'][0]['unique_explicit_specimen_name_bridge']


def test_name_candidate_from_different_cell_conflicts():
    names, july, methods = fixtures()
    other = copy.deepcopy(methods[0])
    other['source_row_number'] = 3
    other['canonical_cell_id'] = 456
    methods[0]['raw_fields']['sample_id'] = 'different'
    methods.append(other)
    result = module.bridge(names, july, methods)
    assert result['summary']['unique_cell_name_conflicts'] == 1
    assert not result['bridges'][0]['unique_explicit_specimen_name_bridge']


def test_duplicate_name_preserved_not_first_selected():
    names, july, methods = fixtures()
    other = copy.deepcopy(methods[0])
    other['source_row_number'] = 3
    other['canonical_cell_id'] = 456
    methods.append(other)
    result = module.bridge(names, july, methods)
    assert len(result['bridges'][0]['methods_name_candidates']) == 2
    assert not result['bridges'][0]['unique_explicit_specimen_name_bridge']


def test_old_source_identity_mismatch_rejected():
    names, july, methods = fixtures()
    july[0]['raw_fields']['cell_specimen_id'] = '124'
    with pytest.raises(ValueError):
        module.bridge(names, july, methods)


def test_only_registered_source_fields_read(tmp_path):
    source = tmp_path/'source.csv'
    header = [*module.FIELDS, 'unapproved']
    source.write_text(','.join(header)+'\n123,P8S4_name,PS_file,never-interpret\n')
    assert list(module.read_names(source, header)[0]['raw_fields']) == module.FIELDS
