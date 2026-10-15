"""Reference qualification cannot promote a favorable subset or partial result."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from run_patchseq_reference_convergence import qualification

NAMES = ('DOP853-medium/DOP853-tight','Radau-medium/Radau-tight','DOP853-tight/Radau-tight')


def test_all_three_reference_comparisons_required():
    assert not qualification({})
    assert not qualification({n:{'passed':True} for n in NAMES[:2]})
    assert qualification({n:{'passed':True} for n in NAMES})


@pytest.mark.parametrize('failure',NAMES)
def test_one_disagreement_rejects_reference(failure):
    evidence = {n:{'passed':True} for n in NAMES}
    evidence[failure]['passed'] = False
    assert not qualification(evidence)
