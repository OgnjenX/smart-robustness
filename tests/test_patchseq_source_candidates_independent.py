"""Independent prospective label predicates retain missingness and all failures."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

scripts = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(scripts))
try:
    spec = importlib.util.spec_from_file_location('independent_candidates', scripts/'verify_patchseq_source_candidates.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
finally:
    sys.path.pop(0)


def inputs():
    reg = yaml.safe_load(module.REG.read_text())
    specimen = {'source_core_inhibitory_group': 'Vip', 'raw_fields': {'structure': 'VISp2/3'},
                'one_agreeing_source_reference_leaf': True}
    annotation = {'methods_candidates': [{'raw_fields': {'class': 'inhibitory', 'Gouwens_etal_2020': 'True',
                                                        'mapping_quality': 'highly consistent'}}],
                  'unique_cell_ID_annotation_association': True, 'literal_structure_agrees': True}
    return reg, specimen, annotation, {'unresolved': []}


def test_independent_predicates_pass_only_exact_author_labels():
    reasons, raw = module.reasons_for(*inputs())
    assert reasons == [] and raw['mapping_quality'] == 'highly consistent'


@pytest.mark.parametrize(('field', 'value'), [('class', 'Inhibitory'), ('Gouwens_etal_2020', True),
    ('mapping_quality', ' highly consistent'), ('mapping_quality', ''), ('mapping_quality', 'moderately consistent')])
def test_no_source_label_cleanup_or_boolean_conversion(field, value):
    values = list(inputs())
    values[2]['methods_candidates'][0]['raw_fields'][field] = value
    reasons, _ = module.reasons_for(*values)
    assert reasons == ['raw-source-label-not-required:'+field]


def test_missing_methods_preserves_all_unknown_conditions():
    values = list(inputs())
    values[2]['methods_candidates'] = []
    reasons, raw = module.reasons_for(*values)
    assert raw is None
    assert reasons == ['methods-cell-association-not-unique', 'raw-source-label-not-required:class',
                       'raw-source-label-not-required:Gouwens_etal_2020', 'raw-source-label-not-required:mapping_quality']


def test_conflicting_anatomy_and_unresolved_partition_remain_separate_failures():
    values = list(inputs())
    values[2]['literal_structure_agrees'] = False
    values[3]['unresolved'] = ['unknown-source-identity']
    reasons, _ = module.reasons_for(*values)
    assert reasons == ['literal-anatomy-agreement-not-established', 'frozen-membership-not-unique-and-resolved']
