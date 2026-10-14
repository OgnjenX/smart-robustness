"""Independent identifier interpretation is strict and type preserving."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

scripts = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(scripts))
try:
    spec = importlib.util.spec_from_file_location('independent_author_cell', scripts/'verify_patchseq_author_cell_association.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
finally:
    sys.path.pop(0)


@pytest.mark.parametrize('storage', [None, 'n', 's', 'inlineStr'])
def test_independent_positive_lexeme_keeps_original_text(storage):
    cell = {'formula_present': False, 'storage_type': storage, 'raw_value': '42', 'literal_text': '42'}
    assert module.positive_lexeme(cell) == '42'


@pytest.mark.parametrize('raw', ['042', '42.0', '4.2e1', ' 42', '0', '４２', True])
def test_independent_reader_rejects_unapproved_identifier(raw):
    cell = {'formula_present': False, 'storage_type': None, 'raw_value': raw, 'literal_text': None}
    with pytest.raises(ValueError):
        module.positive_lexeme(cell)


def test_formula_and_bool_integer_substitution_rejected():
    cell = {'formula_present': True, 'storage_type': 's', 'raw_value': '42', 'literal_text': '42'}
    with pytest.raises(ValueError):
        module.positive_lexeme(cell)
    with pytest.raises(ValueError):
        module.same(1, True)
