"""Bounded workbook headers do not imply permission to inspect source data."""

from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / 'scripts/acquire_patchseq_workbook_schema.py'
spec = importlib.util.spec_from_file_location('workbook_schema', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(tmp_path, fault=None):
    path = tmp_path / 'manifest.xlsx'
    namespace = module.NS[1:-1]
    book = f'<workbook xmlns="{namespace}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="files" sheetId="1" r:id="rId1"/></sheets></workbook>'
    relation = '<Relationships><Relationship Id="rId1" Target="worksheets/sheet1.xml" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet"/></Relationships>'
    sheet = f'<worksheet xmlns="{namespace}"><dimension ref="A1:B2"/><sheetData><row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row><row r="2"><c r="A2"><f>UNAUTHORIZED_FORMULA()</f><v>999</v></c></row></sheetData></worksheet>'
    strings = f'<sst xmlns="{namespace}"><si><t>file_id</t></si><si><t>cell_specimen_id</t></si><si><t>irrelevant-source-value</t></si></sst>'
    if fault == 'external':
        relation = relation.replace('Target="worksheets/sheet1.xml"', 'Target="https://example.org/file" TargetMode="External"')
    elif fault == 'unsafe-target':
        relation = relation.replace('worksheets/sheet1.xml', '../sheet1.xml')
    elif fault == 'formula-header':
        sheet = sheet.replace('<v>0</v>', '<f>SUM(1,2)</f><v>0</v>')
    elif fault == 'duplicate-header':
        strings = strings.replace('cell_specimen_id', 'file_id')
    elif fault == 'missing-string':
        sheet = sheet.replace('<v>1</v>', '<v>99</v>')
    elif fault == 'no-row1':
        sheet = sheet.replace('row r="1"', 'row r="3"')
    elif fault == 'DTD':
        strings = '<!DOCTYPE sst [<!ENTITY x "unsafe">]>' + strings
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('xl/workbook.xml', book)
        archive.writestr('xl/_rels/workbook.xml.rels', relation)
        archive.writestr('xl/worksheets/sheet1.xml', sheet)
        archive.writestr('xl/sharedStrings.xml', strings)
        if fault == 'unsafe-member':
            archive.writestr('../unsafe.xml', '')
        elif fault == 'macro':
            archive.writestr('xl/vbaProject.bin', b'')
        elif fault == 'external-link':
            archive.writestr('xl/externalLinks/externalLink1.xml', '')
    return path


def test_inspects_header_only_without_workbook_edits_or_formula_evaluation(tmp_path):
    path = fixture(tmp_path)
    before = module.digest(path)
    result = module.inspect(path)
    assert module.digest(path) == before
    assert [c['text'] for c in result['sheets'][0]['cells']] == ['file_id', 'cell_specimen_id']
    assert 'irrelevant-source-value' not in str(result)
    assert 'UNAUTHORIZED_FORMULA' not in str(result)
    assert not result['data_rows_interpreted']
    assert not result['formulas_evaluated']
    assert not result['workbook_modified']


@pytest.mark.parametrize('fault', ['external', 'unsafe-target', 'formula-header', 'duplicate-header',
                                'missing-string', 'no-row1', 'DTD', 'unsafe-member', 'macro', 'external-link'])
def test_unsupported_or_unsafe_structure_rejected(tmp_path, fault):
    with pytest.raises(ValueError):
        module.inspect(fixture(tmp_path, fault))
