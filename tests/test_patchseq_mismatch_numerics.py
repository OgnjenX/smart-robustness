"""Pure diagnostic arithmetic; no solver execution or biological input."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from diagnose_patchseq_mismatch_numerics import comparison


def test_exact_agreement_has_no_first_divergence():
    out = comparison(np.zeros(10), np.zeros(10), 50000.)
    assert out['maximum_difference_mV'] == 0
    assert all(t is None for t in out['first_error_above_threshold_seconds'].values())


def test_divergence_time_uses_original_grid_and_fixed_threshold():
    out = comparison(np.zeros(10), np.array([0.]*5+[2.]*5), 50000.)
    assert out['maximum_difference_mV'] == 2.
    assert out['first_error_above_threshold_seconds']['1e-05'] == 5/50000
    assert out['first_error_above_threshold_seconds']['10.0'] is None
