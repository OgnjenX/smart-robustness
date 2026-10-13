"""Raw reference types and conflicting candidates remain visible."""

from __future__ import annotations

import importlib.util
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

path = Path(__file__).resolve().parents[1] / 'scripts/read_patchseq_ccn_reference.py'
spec = importlib.util.spec_from_file_location('ccn_records', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def element(body):
    return ET.fromstring(f'<c xmlns="{module.NS[1:-1]}" r="A2" {body}</c>')


@pytest.mark.parametrize('kind', ['', 't="n"', 't="b"', 't="e"', 't="str"'])
def test_nonliteral_storage_not_coerced_to_identifier(kind):
    result = module.cell_snapshot(element(kind + '><v>123.0</v>'))
    assert result['raw_value'] == '123.0'
    assert result['literal_text'] is None


def test_inline_string_preserves_whitespace():
    result = module.cell_snapshot(element('t="inlineStr"><is><t> CS001 </t></is>'))
    assert result['literal_text'] == ' CS001 '


def test_cached_formula_string_not_literal():
    result = module.cell_snapshot(element('t="inlineStr"><f>FUNCTION()</f><is><t>CS001</t></is>'))
    assert result['formula_present']
    assert result['literal_text'] is None
    assert result['raw_inline_text'] == 'CS001'


@pytest.mark.parametrize('value', ['-1', '1.0', '10001', '١'])
def test_invalid_shared_string_index_rejected(value):
    with pytest.raises(ValueError):
        module.cell_snapshot(element(f't="s"><v>{value}</v>'))


def record(number, accession, alias):
    return {'source_row_number': number, 'cells': {
        'cell_set_ accession': {'literal_text': accession},
        'cell_set_preferred_alias': {'literal_text': alias}}}


def test_exact_lookups_retain_ambiguous_and_conflicting_candidates():
    records = {'cell set nomenclature': {'rows': [record(2, 'CS001', 'Sst type'),
                                                  record(3, 'CS001', 'other'),
                                                  record(4, 'other-ID', 'Sst type')]}}
    july = [{'raw_fields': {'T-type Accession ID': 'CS001', 'T-type Label': 'Sst type'}}]
    result = module.lookup(records, july)[0]
    assert result['exact_accession_candidates'] == [2, 3]
    assert result['exact_preferred_alias_candidates'] == [2, 4]
    assert result['accession_candidate_alias_agreement'] == [True, False]
    assert not result['class_membership_established']


def test_empty_accessions_not_matched_as_valid_ids():
    records = {'cell set nomenclature': {'rows': [record(2, '', '')]}}
    july = [{'raw_fields': {'T-type Accession ID': '', 'T-type Label': ''}}]
    assert module.lookup(records, july)[0]['exact_accession_candidates'] == []


@pytest.mark.parametrize('fault', ['none', 'duplicate-header', 'missing-header', 'duplicate-cell', 'outside-field'])
def test_source_rows_preserve_gaps_and_missing_cells(tmp_path, fault):
    path = tmp_path / 'source.xlsx'
    header = '<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row>'
    rows = '<row r="2"><c r="A2" t="s"><v>2</v></c></row><row r="4"/>'
    if fault == 'duplicate-header':
        header += header
    elif fault == 'missing-header':
        header = ''
    elif fault == 'duplicate-cell':
        rows = rows.replace('</row>', '<c r="A2" t="s"><v>2</v></c></row>', 1)
    elif fault == 'outside-field':
        rows = rows.replace('r="A2"', 'r="C2"')
    data = f'<worksheet xmlns="{module.NS[1:-1]}"><sheetData>{header}{rows}</sheetData></worksheet>'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('xl/sheet.xml', data)
    schema = {'sheets': [{'name': 'selected', 'path': 'xl/sheet.xml',
                         'cells': [{'cell': 'A1', 'text': 'ID'}, {'cell': 'B1', 'text': 'label'}]}]}
    reader = SimpleNamespace(guard=lambda handle, root: None, shared_headers=lambda archive, needed: {2: 'CS001'})
    if fault != 'none':
        with pytest.raises(ValueError):
            module.project(path, schema, {'selected': ['ID', 'label']}, reader)
    else:
        result = module.project(path, schema, {'selected': ['ID', 'label']}, reader)['selected']
        assert result['missing_row_numbers_between_explicit_rows'] == [3]
        assert result['rows'][0]['cells']['ID']['literal_text'] == 'CS001'
        assert result['rows'][0]['cells']['label'] is None
        assert result['rows'][1]['cells'] == {'ID': None, 'label': None}
