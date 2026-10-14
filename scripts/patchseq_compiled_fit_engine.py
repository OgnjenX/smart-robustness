"""Explicit opt-in fitting engine; does not modify the sealed Brian2 runner."""

from __future__ import annotations

import numpy as np
from patchseq_fit_objective import training_loss
from patchseq_fit_search import candidate_dict
from patchseq_response_features import characterize


def adapter(compiled_replay):
    def replay(parameters, *args, **kwargs):
        return compiled_replay([candidate_dict(p) for p in parameters], *args, **kwargs)
    return replay


def score_batch(replay, parameters, records, inputs, warmup):
    """Retain primary batch fallback, every source failure, and fixed objective."""
    predictions = [{} for _ in parameters]
    errors = [[] for _ in parameters]
    for key, (command, rate, holding, epoch) in inputs.items():
        try:
            voltage = replay(parameters, command, rate, holding, warmup)
            for index, v in enumerate(voltage):
                predictions[index][key] = characterize(v, command, rate, epoch, holding)
        except (ValueError, FloatingPointError, OverflowError) as batch_error:
            for index, p in enumerate(parameters):
                try:
                    v = replay([p], command, rate, holding, warmup)[0]
                    predictions[index][key] = characterize(v, command, rate, epoch, holding)
                except (ValueError, FloatingPointError, OverflowError) as exc:
                    predictions[index][key] = None
                    errors[index].append({'source_key': list(key), 'error': str(exc),
                                          'batch_error': str(batch_error)})
    rows = []
    for p, prediction, failures in zip(parameters, predictions, errors, strict=True):
        result = training_loss(records, prediction)
        rows.append({'state': 'scored', 'parameters': candidate_dict(p), 'loss': result['loss'],
                     'group_losses': result['group_losses'], 'simulation_failures': failures,
                     'all_training_observable_gates_pass': result['all_applicable_gates_pass']})
    return rows


def numerical_controls(replay, parameter, command, rate, holding, epoch, warmup=1.):
    original = replay([parameter], command, rate, holding, warmup)[0]
    repeated = replay([parameter], command, rate, holding, warmup)[0]
    refined = replay([parameter], command, rate, holding, warmup, refinement=2)[0]
    longer = replay([parameter], command, rate, holding, warmup*2)[0]
    features = [characterize(v, command, rate, epoch, holding)
                for v in (original, refined, longer)]
    error = float(np.max(np.abs(original-refined)))
    baseline = abs(features[0]['baseline_mean_mV']-features[2]['baseline_mean_mV'])
    gates = {'exact_repeat': bool(np.array_equal(original, repeated)),
             'refinement_voltage': error <= .1,
             'refinement_event_count': features[0]['detected_event_count'] == features[1]['detected_event_count'],
             'warmup_baseline': baseline <= .1,
             'warmup_event_count': features[0]['detected_event_count'] == features[2]['detected_event_count']}
    return {'gates': gates, 'maximum_refinement_voltage_error_mV': error,
            'warmup_baseline_difference_mV': baseline, 'passed': all(gates.values())}
