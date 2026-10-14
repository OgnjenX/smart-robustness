"""Synthetic training loss controls; no biological or held-out responses."""

import importlib.util
import sys
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('patchseq_fit_objective', Path(__file__).parents[1]/'scripts/patchseq_fit_objective.py')
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def features(count=0):
    return {'baseline_mean_mV': -65., 'early_step_mean_mV': -70., 'steady_step_mean_mV': -72.,
            'detected_event_count': count, 'first_event_latency_s': .02 if count else None,
            'last_to_first_interval_ratio': 1.1 if count >= 3 else None}


def record(slot=1, group='a', count=0):
    return {'source_batch': 'original', 'slot': slot, 'partition': 'fitting', 'source_group': 'Sst',
            'layer_label': 'L5', 'frozen_group_id': group, 'sweeps': [{'sweep_number': 1,
            'state': 'fitting-response-characterized', 'features': features(count)}]}


@pytest.mark.parametrize('count', [0, 1, 2, 3, 10])
def test_identical(count):
    out = mod.score_sweep(features(count), features(count))
    assert out['loss'] == 0 and out['all_applicable_gates_pass']
    assert (out['terms']['last_to_first_interval_ratio'] is None) == (count < 3)


def test_silence_requires_exact_zero():
    assert not mod.score_sweep(features(), features(1))['all_applicable_gates_pass']


def test_missing_prediction_not_dropped():
    out = mod.score_sweep(features(3), None)
    assert out['loss'] == 1000 and not out['all_applicable_gates_pass']


def test_group_equal_weight():
    records = [record(1), record(2), record(3, 'b')]
    predictions = {('original', k, 1): features() for k in (1, 2, 3)}
    predictions[('original', 3, 1)]['baseline_mean_mV'] += 2
    assert mod.training_loss(records, predictions)['loss'] == pytest.approx(1/12)


@pytest.mark.parametrize('mutation', ['held', 'pool', 'duplicate', 'missing', 'extra'])
def test_reject_support_or_partition_changes(mutation):
    records = [record()]
    predictions = {('original', 1, 1): features()}
    if mutation == 'held': records[0]['partition'] = 'held-out'
    if mutation == 'pool':
        records.append(record(2)); records[-1]['layer_label'] = 'L23'
    if mutation == 'duplicate': records.append(record())
    if mutation == 'missing': predictions.clear()
    if mutation == 'extra': predictions[('original', 2, 1)] = features()
    with pytest.raises(ValueError): mod.training_loss(records, predictions)


@pytest.mark.parametrize('value', [0, -1, float('nan'), float('inf')])
def test_invalid_scales(value):
    with pytest.raises(ValueError): mod.Objective(voltage_mV=value)


def test_source_missing_not_filled():
    f = features(3); f['first_event_latency_s'] = None
    with pytest.raises(ValueError): mod.score_sweep(f, features(3))
