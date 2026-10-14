"""Synthetic replay runner access and checkpoint controls; no biological fitting."""

import importlib.util
import sys
from pathlib import Path

import pytest

scripts = Path(__file__).parents[1]/'scripts'
for name in ('verify_patchseq_whole_file_integrity', 'run_patchseq_waveform_QC',
             'patchseq_response_features', 'patchseq_isolated_hh', 'patchseq_fit_search',
             'patchseq_fit_objective', 'run_patchseq_isolated_fit'):
    spec = importlib.util.spec_from_file_location(name, scripts/(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
mod = sys.modules['run_patchseq_isolated_fit']


def test_partition_filtered_before_reader():
    records = [{'partition': p, 'source_group': 'Sst', 'layer_label': 'L5'}
               for p in ('fitting', 'held-out')]
    assert mod.fitting_records({'records': records}, 'Sst-L5') == records[:1]


def test_held_out_cannot_open_any_receipt_or_file(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('source access occurred before partition gate')
    monkeypatch.setattr(mod, 'read_yaml', forbidden)
    with pytest.raises(ValueError, match='held-out'):
        mod.prepare_inputs([{'partition': 'held-out'}], {}, {})


def test_empty_replay_forbidden():
    with pytest.raises(ValueError): mod.prepare_inputs([], {}, {})


def test_checkpoint_atomic_and_exact_readback(tmp_path):
    import json
    path = tmp_path/'checkpoint.json'
    for number in (1, 2):
        state = {'candidates': {'0': {'loss': number}}}
        mod.checkpoint(path, state)
        assert json.loads(path.read_text()) == state
        assert not path.with_suffix('.json.pending').exists()


def test_checkpoint_unknown_candidate_rejected():
    p = sys.modules['patchseq_isolated_hh'].Parameters(100, 10, -65, 2000, 500, 0, -60, 1000)
    with pytest.raises(ValueError): mod.validate_rows({'8': {'state': 'scored'}}, [p])
    with pytest.raises(ValueError): mod.validate_rows({'0': {'state': 'scored', 'parameters': {}}}, [p])


def test_failed_candidate_sweep_is_retained_not_removed(monkeypatch):
    def failed(*args, **kwargs):
        raise ValueError('synthetic numerical failure')
    monkeypatch.setattr(mod, 'replay', failed)
    p = sys.modules['patchseq_isolated_hh'].Parameters(100, 10, -65, 2000, 500, 0, -60, 1000)
    f = {'baseline_mean_mV': -65., 'early_step_mean_mV': -60., 'steady_step_mean_mV': -60.,
         'detected_event_count': 0, 'first_event_latency_s': None, 'last_to_first_interval_ratio': None}
    records = [{'partition': 'fitting', 'source_group': 'Sst', 'layer_label': 'L5',
                'source_batch': 'original', 'slot': 1, 'frozen_group_id': 'a',
                'sweeps': [{'sweep_number': 1, 'state': 'fitting-response-characterized', 'features': f}]}]
    out = mod.score_batch([p], records, {('original', 1, 1): (None, 50000, 0, {})}, 1.)
    assert len(out[0]['simulation_failures']) == 1
    assert out[0]['loss'] == pytest.approx(4000/6)
    assert out[0]['all_training_observable_gates_pass'] is False
