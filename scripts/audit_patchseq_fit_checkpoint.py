"""Independent structural/accounting audit; does NOT verify forward simulations."""

from __future__ import annotations

import math
from collections import defaultdict

from scipy.stats import qmc

FIELDS = ('capacitance_pF', 'leak_nS', 'leak_reversal_mV', 'sodium_nS', 'potassium_nS',
          'm_current_nS', 'threshold_shift_mV', 'm_tau_max_ms')
STATUS = 'training-search-and-selected-numerical-controls-retained-independent-verification-pending'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def audit(state, protocol, catalog, seal):
    """Pure metadata check; no runner/model/objective imports or HDF5 access."""
    require(state.get('status') == STATUS, 'search not terminal-complete')
    require(state.get('held_out_response_features_read') is False
            and state.get('network_promotion') is False
            and state.get('frozen_baseline_modified') is False, 'claim/access boundary differs')
    require(set(state['strata']) == set(protocol['strata']), 'stratum scope differs')
    require(all(state['source_hashes'].get(p) == h for p, h in seal['source_sha256'].items()),
            'execution seal differs')
    n = protocol['search']['candidates_per_stratum']
    require(type(n) is int and n > 1 and n & (n-1) == 0, 'invalid sealed count')
    bounds = protocol['search']['bounds']
    require(set(bounds) == set(FIELDS), 'model dimensions differ')
    samples = qmc.Sobol(d=8, scramble=False).random_base2(n.bit_length()-1)
    table = [{name: float(bounds[name][0]+sample[j]*(bounds[name][1]-bounds[name][0]))
              for j, name in enumerate(FIELDS)} for sample in samples]
    require(state['candidate_table'] == table, 'candidate table differs')
    records = catalog['records']
    require(all(r['partition'] in ('fitting', 'held-out') for r in records), 'unknown source partition')
    require(len({(r['source_batch'], r['slot']) for r in records}) == len(records), 'duplicate case')
    reports = []
    for stratum, rows in state['strata'].items():
        gene, layer = stratum.split('-')
        fitting = [r for r in records if r['partition'] == 'fitting'
                   and (r['source_group'], r['layer_label']) == (gene, layer)]
        group_keys, opened = defaultdict(set), set()
        for r in fitting:
            for s in r['sweeps']:
                if s['state'] == 'fitting-response-characterized':
                    key = (r['source_batch'], r['slot'], s['sweep_number'])
                    require(not any(key in keys for keys in group_keys.values()), 'duplicate source sweep')
                    group_keys[r['frozen_group_id']].add(key)
                    opened.add((r['source_batch'], r['slot']))
        require(bool(group_keys), 'no training support')
        known = set().union(*group_keys.values())
        ledger = rows['opened_training_files']
        require(all(r['partition'] == 'fitting' for r in ledger), 'held-out file in ledger')
        require(len(ledger) == len(opened) and
                {(r['source_batch'], r['slot']) for r in ledger} == opened, 'opened file scope differs')
        require(set(rows['candidates']) == {str(i) for i in range(n)}, 'candidate support incomplete')
        ranking = []
        for key, row in rows['candidates'].items():
            index = int(key)
            require(row['state'] == 'scored' and row['parameters'] == table[index], 'candidate identity differs')
            losses = row['group_losses']
            require(set(losses) == set(group_keys), 'independent group coverage differs')
            require(all(finite_number(v) and v >= 0 for v in losses.values()), 'invalid group losses')
            require(finite_number(row['loss']) and row['loss'] >= 0, 'invalid candidate loss')
            expected = math.fsum(losses.values())/len(losses)
            require(math.isclose(expected, row['loss'], rel_tol=1e-12, abs_tol=1e-12), 'group average differs')
            failures = [tuple(f['source_key']) for f in row['simulation_failures']]
            require(len(failures) == len(set(failures)) and set(failures) <= known, 'failure support differs')
            require(type(row['all_training_observable_gates_pass']) is bool, 'observable claim is not boolean')
            require(not failures or row['all_training_observable_gates_pass'] is False, 'failed simulation promoted')
            ranking.append((row['loss'], tuple(row['parameters'][p] for p in FIELDS), index))
        require(type(rows['selected_candidate']) is int and rows['selected_candidate'] == min(ranking)[2],
                'selection differs from sealed deterministic ranking')
        import json
        controls = rows['numerical_controls']
        keys = [tuple(json.loads(k)) for k in controls]
        require(len(keys) == len(set(keys)) and set(keys) == known, 'numerical control scope incomplete')
        require(rows['numerical_controls_complete'] is True, 'numerical completion missing')
        for control in controls.values():
            require(type(control['passed']) is bool, 'invalid numerical claim')
            if 'error' in control:
                require(control['passed'] is False and isinstance(control['error'], str), 'error promoted')
                continue
            gates = control['gates']
            require(set(gates) == {'exact_repeat', 'refinement_voltage', 'refinement_event_count',
                                   'warmup_baseline', 'warmup_event_count'}, 'numerical gate scope differs')
            require(all(type(v) is bool for v in gates.values()), 'nonboolean gate')
            delta = control['maximum_refinement_voltage_error_mV']
            baseline = control['warmup_baseline_difference_mV']
            require(finite_number(delta) and finite_number(baseline) and min(delta, baseline) >= 0,
                    'invalid numerical diagnostics')
            require(gates['refinement_voltage'] == (delta <= .1)
                    and gates['warmup_baseline'] == (baseline <= .1), 'numerical threshold differs')
            require(control['passed'] == all(gates.values()), 'numerical conjunction differs')
        require(rows['all_numerical_controls_pass'] == all(c['passed'] for c in controls.values()),
                'numerical aggregate differs')
        reports.append({'stratum': stratum, 'candidate_count': n, 'training_sweeps': len(known),
                        'independent_groups': len(group_keys), 'structural_accounting_pass': True})
    return {'status': 'structural-accounting-only-forward-replay-not-verified', 'strata': reports,
            'held_out_access_authorized': False, 'biological_validation_complete': False}
