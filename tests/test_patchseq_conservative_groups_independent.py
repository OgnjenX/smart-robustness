"""Independent graph traversal and explicit leakage guards."""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import pytest

scripts = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(scripts))
try:
    spec = importlib.util.spec_from_file_location('independent_groups', scripts/'verify_patchseq_conservative_groups.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
finally:
    sys.path.pop(0)


def record(asset, *keys):
    return {'asset_id': asset, 'group_keys': ['asset:'+asset, *keys]}


def test_graph_reconstructs_transitive_components_without_numeric_namespace_aliases():
    records = [record('a', 'author_donor:1', 'archive_subject:2'),
               record('b', 'archive_subject:2', 'author_donor:3'), record('c', 'author_donor:3'),
               record('d', 'archive_subject:1')]
    assigned, groups = module.graph_components(records, 'fixed-domain')
    module.assert_no_leakage(assigned)
    assert len(groups) == 2
    assert len({r['group_id'] for r in assigned[:3]}) == 1
    assert assigned[3]['group_id'] != assigned[0]['group_id']
    assert all(r['specimen_eligibility_established'] is False for r in assigned)


def test_independent_hashes_and_memberships_are_input_order_invariant():
    records = [record('a', 'author_donor:1'), record('b', 'author_donor:1'), record('c', 'author_donor:2')]
    _, groups = module.graph_components(records, 'fixed-domain')
    reordered = [{**r, 'group_keys': list(reversed(r['group_keys']))} for r in reversed(records)]
    assert module.graph_components(reordered, 'fixed-domain')[1] == groups


@pytest.mark.parametrize('corruption', ['partition', 'group', 'duplicate'])
def test_shared_key_or_membership_corruption_is_rejected(corruption):
    records, _ = module.graph_components([record('a', 'author_donor:1'), record('b', 'author_donor:1')], 'fixed-domain')
    if corruption == 'partition':
        records[1]['partition'] = 'held-out' if records[0]['partition'] == 'fitting' else 'fitting'
    elif corruption == 'group':
        records[1]['group_id'] = 'unrelated'
    else:
        records.append(copy.deepcopy(records[0]))
    with pytest.raises(ValueError):
        module.assert_no_leakage(records)


def test_graph_requires_explicit_namespaced_asset_node():
    with pytest.raises(ValueError):
        module.graph_components([{'asset_id': 'a', 'group_keys': ['author_donor:1']}], 'fixed-domain')
