"""Raw label counts never stand in for a qualified biological cohort."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / 'scripts/inventory_patchseq_raw_labels.py'
spec = importlib.util.spec_from_file_location('raw_labels', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    rows = [{'source_row_number': i, 'raw_fields': dict(zip(module.FIELDS, values, strict=True))}
            for i, values in [(2, ('VISp5', 'id', 'Sst type')), (3, ('', '', ''))]]
    identity = {'annotation_rows': rows, 'assets': [{'asset_id': 'a', 'annotation_candidates': [rows[0]]}]}
    refs = {'assets': [{'asset_id': 'a', 'source_cell_ID_agrees': True}]}
    return identity, refs


def test_raw_labels_remain_unqualified():
    result = module.inventory(*fixture())
    assert result['all_source_rows']['rows'] == 2
    assert result['official_manifest_covered_rows']['rows'] == 1
    assert result['all_source_rows']['raw_value_counts']['structure'] == {'VISp5': 1, '': 1}
    assert not result['biological_eligibility_established']
    assert not result['mapping_confidence_imputed']
    assert not result['cell_class_or_layer_qualification_performed']


def test_unmatched_references_not_in_covered_group():
    identity, refs = fixture()
    refs['assets'][0]['source_cell_ID_agrees'] = False
    assert module.inventory(identity, refs)['official_manifest_covered_rows']['rows'] == 0


def test_duplicate_agreeing_assets_do_not_double_count_cells():
    identity, refs = fixture()
    refs['assets'].append(copy.deepcopy(refs['assets'][0]))
    assert module.inventory(identity, refs)['official_manifest_covered_rows']['rows'] == 1


def test_ambiguous_annotations_rejected():
    identity, refs = fixture()
    identity['assets'][0]['annotation_candidates'].append(identity['annotation_rows'][1])
    with pytest.raises(ValueError, match='ambiguous'):
        module.inventory(identity, refs)


def test_nonstring_raw_label_rejected():
    identity, refs = fixture()
    identity['annotation_rows'][0]['raw_fields']['structure'] = None
    with pytest.raises(ValueError, match='raw string'):
        module.inventory(identity, refs)
