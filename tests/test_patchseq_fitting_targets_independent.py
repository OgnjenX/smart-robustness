"""Independent event/mean reconstruction and discrete corruption rejection."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
from test_patchseq_fitting_targets import characterize, pulse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
try:
    from verify_patchseq_fitting_targets import compare, reconstruct
finally:
    sys.path.pop(0)


@pytest.mark.parametrize('seed', range(5))
@pytest.mark.parametrize('spikes', [(), (.1,), (.1, .3, .6)])
def test_independent_indices_counts_means_and_nominal_current_match(seed, spikes):
    v, i, rate, epoch = pulse(spikes)
    v += np.random.default_rng(seed).normal(0, .02, len(v))
    compare(characterize(v, i, rate, epoch, -25.), reconstruct(v, i, rate, epoch, -25.))


@pytest.mark.parametrize('key', ['detected_event_count', 'event_candidate_indices',
                                'nominal_input_step_pA', 'first_event_latency_s'])
def test_corrupt_event_membership_current_or_latency_not_accepted(key):
    v, i, rate, epoch = pulse((.1, .3, .6))
    expected = reconstruct(v, i, rate, epoch, 0.)
    altered = copy.deepcopy(expected)
    if key == 'event_candidate_indices':
        altered[key][0] += 1
    else:
        altered[key] += 1
    with pytest.raises(ValueError):
        compare(altered, expected)


def test_unrecorded_or_missing_feature_never_imputed_by_verifier():
    v, i, rate, epoch = pulse()
    expected = reconstruct(v, i, rate, epoch, 0.)
    altered = copy.deepcopy(expected)
    altered['first_event_latency_s'] = 0.
    with pytest.raises(ValueError):
        compare(altered, expected)
