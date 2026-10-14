"""Independent checker rejects omitted pairs, changed scaling and ambiguous IDs."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from test_patchseq_pairing import fixture
from test_patchseq_pairing import module as collector

scripts = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(scripts))
try:
    spec = importlib.util.spec_from_file_location('independent_pair', scripts/'verify_patchseq_pairing.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
finally:
    sys.path.pop(0)


def test_independent_checker_matches_complete_fixture():
    rows = fixture()
    assert module.verify_case(rows, collector.evaluate(rows)) == 1


@pytest.mark.parametrize('mutation', ['omit', 'scale', 'membership', 'count', 'eligibility', 'deviation'])
def test_output_mutations_rejected(mutation):
    rows = fixture()
    out = collector.evaluate(rows)
    if mutation == 'omit':
        out['series'].pop()
    elif mutation == 'scale':
        out['series'][0]['conversion'] = 0.001
    elif mutation == 'membership':
        out['sweeps'][0]['series_paths'] = ['/acquisition/r']*2
    elif mutation == 'count':
        out['metadata_pair_passes'] = 2
    elif mutation == 'eligibility':
        out['physiology_eligibility_established'] = True
    else:
        out['series'][0]['exporter_time_spelling_deviation'] = False
    with pytest.raises(ValueError):
        module.verify_case(rows, out)


def test_changed_source_clock_and_duplicated_ID_rejected():
    rows = fixture()
    out = collector.evaluate(rows)
    rows[5]['attributes']['rate']['scalar']['value'] = 49999.9
    with pytest.raises(ValueError):
        module.verify_case(rows, out)
    rows = fixture()
    rows.append({'path': '/acquisition/r/sweep_number', 'scalar_metadata': {'value': 9}})
    with pytest.raises(ValueError, match='identity'):
        module.verify_case(rows, out)
