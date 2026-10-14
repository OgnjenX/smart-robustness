"""Session comparison retains unknowns and never normalizes source strings."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

scripts = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(scripts))
try:
    spec = importlib.util.spec_from_file_location('session_read', scripts/'read_patchseq_NWB_general_session.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
finally:
    sys.path.pop(0)


@pytest.mark.parametrize(('scalar', 'expected'), [
    ({'raw_type': 'bytes', 'decoded_text': '123'}, True),
    ({'raw_type': 'str', 'value': '123'}, True),
    ({'raw_type': 'str', 'value': ' 123'}, False),
    ({'raw_type': 'str', 'value': '0123'}, False),
    ({'raw_type': 'int', 'value': 123}, None),
    ({'raw_type': 'bool', 'value': True}, None),
    ({}, None),
])
def test_exact_type_preserving_session_comparison(scalar, expected):
    assert module.comparison({'scalar': scalar}, '123') is expected


def test_missing_session_not_filled_from_filename():
    assert module.comparison({'present': False}, '123') is None
