"""Typed manifest integers preserve precision, source types and rejection evidence."""

from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / 'scripts/read_patchseq_2021_manifest.py'
spec = importlib.util.spec_from_file_location('manifest_records', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize('kind', [None, 'n', 's', 'inlineStr'])
def test_positive_decimal_from_registered_types_without_float_conversion(kind):
    cell = {'storage_type': kind, 'raw_value': '123', 'literal_text': '123', 'formula_present': False}
    assert module.canonical_integer(cell) == 123


@pytest.mark.parametrize('value', ['00123', '123.0', '1.23e2', ' 123', '+123', '0', '١٢٣', None])
def test_noncanonical_lexemes_not_repaired(value):
    cell = {'storage_type': 'n', 'raw_value': value, 'literal_text': None, 'formula_present': False}
    assert module.canonical_integer(cell) is None


def test_numeric_precision_limit_but_exact_string_not_rounded():
    cell = {'storage_type': 'n', 'raw_value': '1234567890123456', 'literal_text': '1234567890123456', 'formula_present': False}
    assert module.canonical_integer(cell) is None
    cell['storage_type'] = 's'
    assert module.canonical_integer(cell) == 1234567890123456


@pytest.mark.parametrize('kind', ['b', 'e', 'str', 'd'])
def test_other_types_not_identifiers(kind):
    cell = {'storage_type': kind, 'raw_value': '123', 'literal_text': '123', 'formula_present': False}
    assert module.canonical_integer(cell) is None


def test_formula_result_not_identifier():
    cell = {'storage_type': 'n', 'raw_value': '123', 'literal_text': None, 'formula_present': True}
    assert module.canonical_integer(cell) is None


def test_rows_types_gaps_and_missing_values_preserved(tmp_path):
    path = tmp_path / 'source.xlsx'
    fields = ['file_id', 'file_size', 'cell_specimen_id']
    definition = {'path': 'xl/sheet.xml', 'cells': [{'cell': chr(65+i)+'1', 'text': k} for i, k in enumerate(fields)]}
    data = f'<worksheet xmlns="{module.NS[1:-1]}"><sheetData><row r="1"/><row r="2"><c r="A2"><v>123</v></c><c r="B2"><v>123.0</v></c></row><row r="4"/></sheetData></worksheet>'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('xl/sheet.xml', data)
    result = module.project(path, definition, fields, lambda handle, root: None)
    assert result['missing_row_numbers_between_explicit_rows'] == [3]
    assert result['rows'][0]['canonical_integers'] == {'file_id': 123, 'file_size': None, 'cell_specimen_id': None}
    assert result['rows'][0]['cells']['file_size']['raw_value'] == '123.0'
    assert result['rows'][1]['cells'] == dict.fromkeys(fields)
    assert not result['workbook_modified']
