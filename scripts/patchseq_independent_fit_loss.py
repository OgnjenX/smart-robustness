"""Independent six-observable scoring; no fitting/model/extractor imports."""

from __future__ import annotations

import math

SCALES = {'baseline_mean_mV': 2., 'early_step_mean_mV': 2., 'steady_step_mean_mV': 2.,
          'detected_event_count': 1., 'first_event_latency_s': .010,
          'last_to_first_interval_ratio': .25}


def observable_loss(target, predicted):
    count = target['detected_event_count']
    if type(count) is not int or count < 0:
        raise ValueError('invalid source count')
    if ((target.get('first_event_latency_s') is not None) != (count > 0)
        or (target.get('last_to_first_interval_ratio') is not None) != (count >= 3)):
        raise ValueError('source event-dependent feature applicability differs')
    contributions, gates = {}, {}
    for name, scale in SCALES.items():
        observed = target.get(name)
        if observed is None:
            if name not in ('first_event_latency_s', 'last_to_first_interval_ratio'):
                raise ValueError('missing mandatory source observable')
            contributions[name], gates[name] = None, None
            continue
        if type(observed) not in (int, float) or not math.isfinite(observed):
            raise ValueError('invalid source observable')
        value = predicted.get(name) if predicted is not None else None
        valid = type(value) in (int, float) and math.isfinite(value)
        if name == 'detected_event_count' and valid:
            valid = value >= 0 and int(value) == value
        if not valid:
            contributions[name], gates[name] = 1000., False
        else:
            difference = value-observed
            contributions[name] = (difference/scale)*(difference/scale)
            gates[name] = difference == 0 if name == 'detected_event_count' and count == 0 else abs(difference) <= scale
    return {'loss': math.fsum(x for x in contributions.values() if x is not None)/6,
            'terms': contributions, 'gates': gates,
            'all_applicable_gates_pass': all(x for x in gates.values() if x is not None)}


def reconstruct_training_loss(records, predictions):
    if not records or any(r['partition'] != 'fitting' for r in records):
        raise ValueError('independent training scorer forbids held-out or empty input')
    if len({(r['source_group'], r['layer_label']) for r in records}) != 1:
        raise ValueError('independent scorer forbids class/layer pooling')
    grouped, seen, decisions = {}, set(), []
    for record in records:
        for sweep in record['sweeps']:
            if sweep['state'] != 'fitting-response-characterized':
                continue
            key = (record['source_batch'], record['slot'], sweep['sweep_number'])
            if key in seen or key not in predictions:
                raise ValueError('duplicate or missing source prediction')
            seen.add(key)
            row = observable_loss(sweep['features'], predictions[key])
            grouped.setdefault(record['frozen_group_id'], []).append(row['loss'])
            decisions.append({'key': key, **row})
    if not seen or set(predictions) != seen:
        raise ValueError('independent prediction/source scope differs')
    means = {group: math.fsum(values)/len(values) for group, values in grouped.items()}
    return {'loss': math.fsum(means.values())/len(means), 'group_losses': means,
            'sweeps': decisions, 'all_applicable_gates_pass': all(
                row['all_applicable_gates_pass'] for row in decisions)}
