"""Diagnostic comparison does not relax discrete or arithmetic tolerances."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from diagnose_patchseq_fit_mismatch import differing_fields


def test_discrete_differences_are_never_tolerated():
    assert differing_fields({'count': 1}, {'count': 2}) == {
        'count': {'primary': 1, 'independent': 2}}


def test_only_sealed_float_arithmetic_tolerance_is_used():
    assert not differing_fields({'voltage': -65.}, {'voltage': -65.+1e-7})
    assert differing_fields({'voltage': -65.}, {'voltage': -65.+.01})
