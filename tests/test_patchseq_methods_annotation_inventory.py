"""Raw confidence labels never become eligibility and no columns are imputed."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / 'scripts/inventory_patchseq_methods_annotations.py'
spec = importlib.util.spec_from_file_location('methods_annotations', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
FIELDS = ['cell_id', 'sample_id', 'mapping_quality', 'Gouwens_etal_2020', 'dataset', 'class']


def test_raw_quality_indicator_and_duplicates_preserved(tmp_path):
    source = tmp_path/'source.csv'
    header = [*FIELDS, 'unapproved_physiology']
    source.write_text(','.join(header)+'\n123,s, unknown ,TRUE,d,c,not-a-number\n123,s,,1,d,c,never-interpret\n')
    result = module.project(source, header, FIELDS)
    assert result['duplicate_canonical_cell_ids'] == [123]
    assert result['duplicate_nonempty_sample_ids'] == ['s']
    assert result['raw_value_counts']['mapping_quality'] == {'': 1, ' unknown ': 1}
    assert result['raw_value_counts']['Gouwens_etal_2020'] == {'1': 1, 'TRUE': 1}
    assert 'unapproved_physiology' not in result['rows'][0]['raw_fields']


@pytest.mark.parametrize('identity', ['00123', '123.0', '١٢٣', ' 123', ''])
def test_invalid_ids_not_repaired(tmp_path, identity):
    source = tmp_path/'source.csv'
    source.write_text(','.join(FIELDS)+'\n'+identity+',s,q,flag,d,c\n')
    result = module.project(source, FIELDS, FIELDS)
    assert result['rows'][0]['raw_fields']['cell_id'] == identity
    assert result['rows'][0]['canonical_cell_id'] is None
    assert result['noncanonical_cell_ids'] == 1


@pytest.mark.parametrize('row', ['123,s,q,flag,d\n', '123,s,q,flag,d,c,extra\n'])
def test_source_shape_rejected(tmp_path, row):
    source = tmp_path/'source.csv'
    source.write_text(','.join(FIELDS)+'\n'+row)
    with pytest.raises(ValueError):
        module.project(source, FIELDS, FIELDS)
