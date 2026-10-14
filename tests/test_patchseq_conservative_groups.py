"""Conservative grouping separates namespaces and prevents cross-key leakage."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

scripts = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(scripts))
try:
    spec = importlib.util.spec_from_file_location('conservative_groups', scripts/'partition_patchseq_source_groups.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
finally:
    sys.path.pop(0)


def record(asset, *keys):
    return {'asset_id': asset, 'group_keys': sorted(['asset:'+asset, *keys])}


@pytest.mark.parametrize('namespace', ['author_donor', 'archive_subject', 'author_cell'])
def test_sharing_any_registered_key_prevents_cross_partition_leakage(namespace):
    records, groups = module.components([record('a', namespace+':7'), record('b', namespace+':7')], 'fixed-domain')
    assert len(groups) == 1
    assert records[0]['group_id'] == records[1]['group_id']
    assert records[0]['partition'] == records[1]['partition']
    assert records[0]['specimen_eligibility_established'] is False


def test_equal_digits_in_different_namespaces_do_not_imply_equivalence():
    _, groups = module.components([record('a', 'author_donor:7'), record('b', 'archive_subject:7')], 'fixed-domain')
    assert len(groups) == 2


def test_transitive_sharing_and_input_order_invariance():
    source = [record('a', 'author_donor:1', 'archive_subject:2'),
              record('b', 'archive_subject:2', 'author_donor:3'),
              record('c', 'author_donor:3'), record('d', 'author_donor:4')]
    a, groups_a = module.components(source, 'fixed-domain')
    b, groups_b = module.components(list(reversed(source)), 'fixed-domain')
    assert groups_a == groups_b and len(groups_a) == 2
    assert {r['asset_id']: (r['group_id'], r['partition']) for r in a} == {r['asset_id']: (r['group_id'], r['partition']) for r in b}
    assert len({r['group_id'] for r in a if r['asset_id'] in ('a', 'b', 'c')}) == 1


def test_ambiguous_and_noncanonical_metadata_are_retained():
    reference = {'assets': [{'asset_id': 'synthetic', 'path': 'sub-1/sub-1_ses-7_icephys.nwb',
                             'complete_external_file_reference_agreement': True}]}
    july = {'July_rows': [
        {'source_row_number': 2, 'canonical_ids': {'cell_specimen_id': 42, 'donor_id': 5, 'ephys_session_id': 7},
         'raw_fields': {'cell_specimen_id': '42', 'donor_id': '5', 'ephys_session_id': '7'}},
        {'source_row_number': 3, 'canonical_ids': {'cell_specimen_id': 43, 'donor_id': 6, 'ephys_session_id': 7},
         'raw_fields': {'cell_specimen_id': '43', 'donor_id': '06', 'ephys_session_id': '7'}}]}
    records = module.project(reference, july)
    assert len(records) == 1 and records[0]['July_candidate_source_rows'] == [2, 3]
    assert 'author_donor:5' in records[0]['group_keys']
    assert 'author_donor:06' not in records[0]['group_keys']
    assert records[0]['unresolved']


def test_empty_group_input_rejected():
    with pytest.raises(ValueError):
        module.components([{'asset_id': 'a', 'group_keys': []}], 'fixed-domain')
