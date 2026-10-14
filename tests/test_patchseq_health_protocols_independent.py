"""Independent code-tag pairing and missingness reconstruction on synthetic metadata."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from test_patchseq_health_protocols import CODES, health_fixture
from test_patchseq_health_protocols import module as collector
from test_patchseq_pairing import attr

scripts = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(scripts))
try:
    spec = importlib.util.spec_from_file_location('independent_health', scripts/'verify_patchseq_health_protocols.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
finally:
    sys.path.pop(0)


def test_all_metadata_and_missing_codes_match():
    rows = health_fixture()
    assert module.reconstruct(rows, CODES) == collector.evaluate(rows, CODES)
    for index in (0, 3):
        rows[index]['attributes']['stimulus_description'] = attr('Test')
    assert module.reconstruct(rows, CODES) == collector.evaluate(rows, CODES)


@pytest.mark.parametrize('kind', ['clock', 'namespace', 'unit', 'partner'])
def test_bad_metadata_not_validated_as_success(kind):
    rows = health_fixture()
    if kind == 'clock':
        rows[5]['attributes']['rate'] = attr(49999.)
    elif kind == 'namespace':
        rows[0]['attributes']['namespace'] = attr('unknown')
    elif kind == 'unit':
        rows[1]['attributes']['unit'] = attr('pA')
    else:
        rows = rows[:3]
    with pytest.raises(ValueError):
        module.reconstruct(rows, CODES)
