"""Synthetic transition gates only; no biological files opened."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
import check_patchseq_compiled_fitting_equivalence as mod
import patchseq_compiled_fit_engine as fitting


def test_failure_disagreement_blocks_transition():
    v = np.zeros(20)
    assert not mod.compare((v, None), (None, 'failed'), None, None, None, None)['passed']
    assert mod.compare((None, 'a'), (None, 'b'), None, None, None, None)['passed']


def test_batch_failure_isolates_and_retains_every_candidate():
    def engine(rows, *args):
        if len(rows) > 1 or rows[0] == 1:
            raise ValueError('synthetic failure')
        return np.zeros((1, 20))
    results = mod.traces(engine, [0, 1, 2], None, None, None, None)
    assert len(results) == 3
    assert results[0][0] is not None and results[1][0] is None and results[2][0] is not None


def test_synthetic_windows_are_valid():
    config = {'duration_s': .3, 'step_start_s': .05, 'step_stop_s': .25,
              'pre_window_s': [.01, .04]}
    command, epoch = mod.synthetic_input(config, 50000, 30.)
    v = np.full(len(command), -65.)
    assert mod.compare((v, None), (v.copy(), None), command, 50000, 20., epoch)['passed']


def test_numerical_control_gates_preserved(monkeypatch):
    calls = []
    def replay(*args, **kwargs):
        calls.append(1)
        return np.full((1, 10), -65. if len(calls) != 2 else -64.)
    monkeypatch.setattr(fitting, 'characterize', lambda *args: {
        'baseline_mean_mV': -65., 'detected_event_count': 0})
    result = fitting.numerical_controls(replay, None, None, None, None, None)
    assert len(calls) == 4 and not result['passed'] and not result['gates']['exact_repeat']
