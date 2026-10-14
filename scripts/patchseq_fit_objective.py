"""Training-only objective and conditional observable gates, not model validity."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class Objective:
    voltage_mV: float = 2.0
    count: float = 1.0
    latency_s: float = .010
    adaptation: float = .25
    missing_penalty: float = 1000.0

    def __post_init__(self):
        if not all(math.isfinite(v) and v > 0 for v in vars(self).values()):
            raise ValueError('objective scales must be finite and positive')


DEFAULT_OBJECTIVE = Objective()


def score_sweep(target, prediction, objective=DEFAULT_OBJECTIVE):
    """Score six fixed dimensions; absent target-dependent terms are inapplicable."""
    names = ('baseline_mean_mV', 'early_step_mean_mV', 'steady_step_mean_mV',
             'detected_event_count', 'first_event_latency_s', 'last_to_first_interval_ratio')
    scales = (objective.voltage_mV,)*3+(objective.count, objective.latency_s,
                                       objective.adaptation)
    required = names[:4]
    for name in required:
        if target.get(name) is None or not math.isfinite(target[name]):
            raise ValueError('missing or nonfinite mandatory source observable')
    n = target['detected_event_count']
    if isinstance(n, bool) or n < 0 or int(n) != n:
        raise ValueError('invalid source event count')
    if (target.get('first_event_latency_s') is not None) != (n > 0):
        raise ValueError('source latency/count disagreement')
    if (target.get('last_to_first_interval_ratio') is not None) != (n >= 3):
        raise ValueError('source adaptation/count disagreement')
    terms, gates = {}, {}
    for name, scale in zip(names, scales, strict=True):
        observed = target.get(name)
        if observed is None:
            terms[name], gates[name] = None, None
            continue
        if not math.isfinite(observed):
            raise ValueError('nonfinite source observable')
        predicted = prediction.get(name) if prediction is not None else None
        invalid = predicted is None or not math.isfinite(predicted)
        if name == 'detected_event_count' and not invalid:
            invalid = isinstance(predicted, bool) or predicted < 0 or int(predicted) != predicted
        if invalid:
            terms[name], gates[name] = objective.missing_penalty, False
            continue
        error = abs(predicted-observed)
        terms[name] = (error/scale)**2
        # Silence is exact; otherwise allow one detected event of count error.
        gates[name] = error == 0 if name == 'detected_event_count' and n == 0 else error <= scale
    return {'loss': math.fsum(v for v in terms.values() if v is not None)/len(names),
            'terms': terms, 'gates': gates,
            'all_applicable_gates_pass': all(v for v in gates.values() if v is not None)}


def training_loss(records, predictions, objective=DEFAULT_OBJECTIVE):
    """Require one frozen stratum and exact prediction keys; average group then sweep."""
    if not records or any(r['partition'] != 'fitting' for r in records):
        raise ValueError('training objective forbids held-out or empty records')
    strata = {(r['source_group'], r['layer_label']) for r in records}
    if len(strata) != 1:
        raise ValueError('class/layer pooling forbidden')
    groups, keys, rows = defaultdict(list), set(), []
    for record in records:
        for sweep in record['sweeps']:
            if sweep['state'] != 'fitting-response-characterized':
                continue
            key = (record['source_batch'], record['slot'], sweep['sweep_number'])
            if key in keys:
                raise ValueError('duplicate source sweep')
            keys.add(key)
            if key not in predictions:
                raise ValueError('missing candidate sweep: retain explicit failure instead')
            result = score_sweep(sweep['features'], predictions[key], objective)
            groups[record['frozen_group_id']].append(result['loss'])
            rows.append({'key': key, **result})
    if not keys or set(predictions) != keys:
        raise ValueError('prediction/source support differs or empty')
    means = {key: math.fsum(values)/len(values) for key, values in groups.items()}
    return {'loss': math.fsum(means.values())/len(means), 'group_losses': means,
            'sweeps': rows, 'all_applicable_gates_pass': all(
                row['all_applicable_gates_pass'] for row in rows)}
