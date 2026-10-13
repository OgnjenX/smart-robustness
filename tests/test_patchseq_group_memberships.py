"""Class-node membership follows edges, never a marker substring fallback."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / 'scripts/trace_patchseq_group_memberships.py'
spec = importlib.util.spec_from_file_location('group_memberships', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def node(path, accession, alias, children):
    attrs = {'cell_set_accession': accession, 'cell_set_preferred_alias': alias, 'taxonomy_id': 'reference'}
    return {'path': path, 'leaf': not children, 'children': children,
            'attributes': {k: {'present': True, 'JSON_type': 'str', 'raw_value': v} for k, v in attrs.items()}}


def fixture():
    nodes = [node('root', 'gaba', 'GABAergic', ['sst', 'vip', 'uncovered']),
             node('sst', 'sst-ID', 'Sst', ['a', 'b']), node('vip', 'vip-ID', 'Vip', ['c', 'd']),
             node('a', 'a-ID', 'Pvalb marker-containing label', []), node('b', 'b-ID', 'other', []),
             node('c', 'c-ID', 'Vip type', []), node('d', 'd-ID', 'other2', []),
             node('uncovered', 'u-ID', 'Sst label outside subgroup', [])]
    alignments = [{'legacy_accession': 'old-'+n['path'], 'legacy_label': n['attributes']['cell_set_preferred_alias']['raw_value'],
                   'reference_accession': n['attributes']['cell_set_accession']['raw_value'],
                   'reference_taxonomy': 'reference', 'exact_name_alignment_supported': True} for n in nodes if n['leaf']]
    return nodes, alignments, 'reference', {'GABAergic': 'gaba', 'Sst': 'sst-ID', 'Vip': 'vip-ID'}


def test_membership_follows_ancestry_not_marker_names():
    result = module.trace(*fixture())
    rows = {r['reference_accession']: r for r in result['memberships']}
    assert rows['a-ID']['exclusive_named_inhibitory_subgroup'] == 'Sst'
    assert rows['u-ID']['source_groups'] == ['GABAergic']
    assert rows['u-ID']['exclusive_named_inhibitory_subgroup'] is None
    assert not rows['a-ID']['specimen_qualification_established']
    assert result['summary']['GABAergic_types_outside_named_subgroups'] == 1


@pytest.mark.parametrize('fault', ['unsupported', 'wrong-alias', 'wrong-taxonomy', 'missing-leaf', 'nonleaf-target'])
def test_unverified_alignment_retained_as_unresolved(fault):
    args = list(fixture())
    a = args[1][0]
    if fault == 'unsupported':
        a['exact_name_alignment_supported'] = False
    elif fault == 'wrong-alias':
        a['legacy_label'] = 'changed'
    elif fault == 'wrong-taxonomy':
        a['reference_taxonomy'] = 'different'
    elif fault == 'missing-leaf':
        a['reference_accession'] = 'absent'
    else:
        a['reference_accession'] = 'sst-ID'
    result = module.trace(*args)
    assert not result['memberships'][0]['source_leaf_identity_agrees']
    assert result['memberships'][0]['source_groups'] == []


@pytest.mark.parametrize('fault', ['duplicate-ID', 'duplicate-path', 'dangling-child', 'duplicate-parent',
                                'wrong-taxonomy', 'wrong-group-alias', 'nonboolean-leaf', 'missing-root'])
def test_invalid_source_graph_rejected(fault):
    args = list(fixture())
    nodes = args[0]
    if fault in ('duplicate-ID', 'duplicate-path'):
        other = copy.deepcopy(nodes[-1])
        if fault == 'duplicate-ID':
            other['path'] = 'new'
        else:
            other['attributes']['cell_set_accession']['raw_value'] = 'new-ID'
        nodes.append(other)
    elif fault == 'dangling-child':
        nodes[0]['children'].append('absent')
    elif fault == 'duplicate-parent':
        nodes[2]['children'].append('a')
    elif fault == 'wrong-taxonomy':
        nodes[-1]['attributes']['taxonomy_id']['raw_value'] = 'different'
    elif fault == 'wrong-group-alias':
        nodes[1]['attributes']['cell_set_preferred_alias']['raw_value'] = 'changed'
    elif fault == 'nonboolean-leaf':
        nodes[-1]['leaf'] = 1
    else:
        nodes[0]['path'] = 'no-root'
    with pytest.raises(ValueError):
        module.trace(*args)


def test_overlapping_named_subgroups_remain_unresolved():
    args = list(fixture())
    nodes = args[0]
    nodes[0]['children'] = ['sst', 'uncovered']
    nodes[1]['children'] = ['vip', 'a', 'b']
    result = module.trace(*args)
    c = next(r for r in result['memberships'] if r['reference_accession'] == 'c-ID')
    assert c['named_inhibitory_subgroups'] == ['Vip', 'Sst']
    assert c['exclusive_named_inhibitory_subgroup'] is None
    assert result['summary']['types_with_overlapping_named_subgroups'] == 2
