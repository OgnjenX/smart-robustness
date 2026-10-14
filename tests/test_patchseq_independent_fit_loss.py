"""Synthetic independent scoring controls and cross-implementation comparison."""

import importlib.util
import sys
from pathlib import Path

import pytest

modules = {}
for name in ('patchseq_independent_fit_loss', 'patchseq_fit_objective'):
    spec = importlib.util.spec_from_file_location(name+'_loss_control', Path(__file__).parents[1]/'scripts'/(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    modules[name] = module
mod = modules['patchseq_independent_fit_loss']


def feature(count):
    return {'baseline_mean_mV': -65., 'early_step_mean_mV': -60., 'steady_step_mean_mV': -58.,
            'detected_event_count': count, 'first_event_latency_s': .03 if count else None,
            'last_to_first_interval_ratio': 1.2 if count >= 3 else None}


@pytest.mark.parametrize('count', [0, 1, 2, 3, 12])
@pytest.mark.parametrize('mode', ['identical', 'missing', 'error'])
def test_cross_implementation_synthetic(count, mode):
    target = feature(count); predicted = feature(count)
    if mode == 'missing': predicted = None
    if mode == 'error': predicted['baseline_mean_mV'] += 3.
    assert mod.observable_loss(target, predicted) == modules['patchseq_fit_objective'].score_sweep(target, predicted)


def test_equal_group_not_record_weight():
    records, predictions = [], {}
    for slot, group in [(1, 'a'), (2, 'a'), (3, 'b')]:
        records.append({'partition': 'fitting', 'source_group': 'Sst', 'layer_label': 'L5',
                        'source_batch': 'original', 'slot': slot, 'frozen_group_id': group,
                        'sweeps': [{'state': 'fitting-response-characterized', 'sweep_number': 1,
                                    'features': feature(3)}]})
        predictions[('original', slot, 1)] = feature(3)
    predictions[('original', 3, 1)]['baseline_mean_mV'] += 2.
    assert mod.reconstruct_training_loss(records, predictions)['loss'] == pytest.approx(1/12)
    assert mod.reconstruct_training_loss(records, predictions) == modules['patchseq_fit_objective'].training_loss(records, predictions)


def test_held_out_rejected_before_feature_access():
    with pytest.raises(ValueError, match='held-out'):
        mod.reconstruct_training_loss([{'partition': 'held-out'}], {})


def test_missing_target_not_zero_filled():
    target = feature(3); target['first_event_latency_s'] = None
    with pytest.raises(ValueError): mod.observable_loss(target, feature(3))


def test_silence_requires_zero_even_with_one_count_tolerance():
    assert not mod.observable_loss(feature(0), feature(1))['all_applicable_gates_pass']
