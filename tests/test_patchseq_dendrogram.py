"""Source topology is retained without inferring biological membership."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / 'scripts/read_patchseq_dendrogram.py'
spec = importlib.util.spec_from_file_location('source_dendrogram', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def leaf(label='source', array=False):
    attrs = {'label': label, 'cell_set_accession': 'CS001', 'markers': 'uninterpreted-marker-values'}
    return {'leaf_attributes': [attrs] if array else attrs}


def test_topology_and_missing_attributes_preserved_not_promoted():
    tree = {'node_attributes': [{'label': 'root'}], 'children': [leaf(), leaf('other', True)]}
    result = module.project(json.dumps(tree).encode())
    assert result['summary'] == {'nodes': 3, 'leaves': 2, 'maximum_depth': 1}
    assert result['nodes'][0]['children'] == ['root/children/0', 'root/children/1']
    assert result['nodes'][1]['attributes']['taxonomy_id'] == {'present': False, 'raw_value': None, 'JSON_type': None}
    assert 'markers' in result['nodes'][1]['source_attribute_keys']
    assert 'uninterpreted-marker-values' not in str(result)
    assert not result['class_membership_established']


def test_scalar_array_not_normalized_to_string():
    tree = leaf()
    tree['leaf_attributes']['label'] = ['source']
    result = module.project(json.dumps(tree).encode())
    assert result['nodes'][0]['attributes']['label']['raw_value'] == ['source']
    assert result['nodes'][0]['attributes']['label']['JSON_type'] == 'list'


@pytest.mark.parametrize('tree', [[], {}, {'leaf_attributes': []}, {'leaf_attributes': [{}, {}]},
                                {'node_attributes': {}, 'children': []},
                                {'node_attributes': {}, 'children': [leaf()]},
                                {'leaf_attributes': {}, 'children': []}])
def test_unknown_shapes_fail_without_repairs(tree):
    with pytest.raises(ValueError):
        module.project(json.dumps(tree).encode())


@pytest.mark.parametrize('data', [b'{"leaf_attributes":{},"leaf_attributes":{}}',
                                b'{"leaf_attributes":{"label":NaN}}',
                                b'{"leaf_attributes":{"label":Infinity}}',
                                b'{"leaf_attributes":{"label":1e309}}'])
def test_duplicate_or_nonfinite_JSON_rejected(data):
    with pytest.raises(ValueError):
        module.project(data)


def test_tree_depth_bound():
    tree = leaf()
    for _ in range(65):
        tree = {'node_attributes': {}, 'children': [tree, leaf()]}
    with pytest.raises(ValueError, match='depth'):
        module.project(json.dumps(tree).encode())
