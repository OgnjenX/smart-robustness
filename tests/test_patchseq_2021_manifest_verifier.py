"""Independent reconstruction must reject tampered typed evidence."""

from __future__ import annotations

import copy
import importlib.util
import zipfile
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / 'scripts/verify_patchseq_2021_manifest.py'
spec = importlib.util.spec_from_file_location('typed_verifier', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize('value,kind,expected', [('123', 'n', 123), ('123', 's', 123),
    ('1234567890123456', 'n', None), ('1234567890123456', 's', 1234567890123456),
    ('123.0', 'n', None), ('00123', 's', None), ('١٢٣', 's', None), ('123', 'b', None)])
def test_registered_integer_rules(value, kind, expected):
    assert module.integer({'storage_type': kind, 'raw_value': value,
                           'literal_text': value, 'formula_present': False}) == expected


def test_formula_not_evaluated():
    assert module.integer({'storage_type': 's', 'raw_value': '0',
                           'literal_text': '123', 'formula_present': True}) is None


@pytest.mark.parametrize('mutation', ['type', 'gaps', 'count', 'flag'])
def test_changed_evidence_rejected(mutation):
    rows = [{'source_row_number': 2, 'canonical_integers': {'file_id': 123}}]
    result = {'rows': copy.deepcopy(rows), 'explicit_rows': 1,
              'missing_row_numbers_between_explicit_rows': [],
              'workbook_modified': False, 'formulas_evaluated': False}
    module.check(rows, [], result)
    if mutation == 'type':
        result['rows'][0]['canonical_integers']['file_id'] = '123'
    elif mutation == 'gaps':
        result['missing_row_numbers_between_explicit_rows'] = [3]
    elif mutation == 'count':
        result['explicit_rows'] = 2
    else:
        result['formulas_evaluated'] = True
    with pytest.raises(ValueError):
        module.check(rows, [], result)


def test_external_entity_rejected():
    with pytest.raises(ValueError):
        module.xml(b'<!DOCTYPE x [<!ENTITY y SYSTEM "file:///etc/passwd">]><x>&y;</x>')


@pytest.mark.parametrize('invalid', [None, 'duplicate', 'coordinate', 'header'])
def test_dom_reconstruction_strings_formulas_gaps_and_rejections(tmp_path, invalid):
    workbook = tmp_path / 'fixture.xlsx'
    header = ''.join(f'<c r="{chr(65+i)}1" t="s"><v>{i}</v></c>' for i in range(11))
    if invalid == 'header':
        header = header.replace('A1', 'B1')
    cells = '<c r="A2"><v>123</v></c><c r="B2" t="s"><f>1</f><v>11</v></c><c r="C2"><v>123.0</v></c><c r="K2" t="inlineStr"><is><t>456</t></is></c>'
    if invalid == 'duplicate':
        cells += '<c r="A2"><v>123</v></c>'
    elif invalid == 'coordinate':
        cells = cells.replace('A2', 'A3')
    sheet = f'<worksheet xmlns="{module.NS}"><sheetData><row r="1">{header}</row><row r="2">{cells}</row><row r="4"/></sheetData></worksheet>'
    strings = ''.join(f'<si><t>{name}</t></si>' for name in module.FIELDS+['cached'])
    with zipfile.ZipFile(workbook, 'w') as archive:
        archive.writestr('xl/worksheets/sheet1.xml', sheet)
        archive.writestr('xl/sharedStrings.xml', f'<sst xmlns="{module.NS}">{strings}</sst>')
    if invalid:
        with pytest.raises(ValueError):
            module.rebuild(workbook)
        return
    rows, gaps = module.rebuild(workbook)
    assert gaps == [3]
    assert rows[0]['canonical_integers'] == {'file_id': 123, 'file_size': None, 'cell_specimen_id': 456}
    assert rows[0]['cells']['file_name']['decoded_shared_string'] == 'cached'
    assert rows[0]['cells']['file_name']['literal_text'] is None
    assert rows[1]['cells'] == dict.fromkeys(module.FIELDS)
