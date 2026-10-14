"""Sealed synthetic boundary/long-trace checks; no biological data or fitting."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import yaml
from patchseq_hh_compiled_prototype import compiled_engine
from patchseq_hh_independent_replay import FIELDS, integrate
from scipy.stats import qmc
from verify_patchseq_whole_file_integrity import digest

REG = Path('docs/validation-results/post2008-patchseq-compiled-equivalence-registration-1453.yaml')


def compare_case(engine, row, current, rate):
    traces, failures, elapsed = [], [], []
    for function in (integrate, engine):
        started = time.perf_counter()
        try:
            traces.append(function([row], current, rate, [row['leak_reversal_mV']]))
            failures.append(None)
        except (ValueError, FloatingPointError, OverflowError) as exc:
            traces.append(None); failures.append(str(exc))
        elapsed.append(time.perf_counter()-started)
    if traces[0] is None or traces[1] is None:
        return {'passed': (traces[0] is None) == (traces[1] is None),
                'failure_support_matches': (traces[0] is None) == (traces[1] is None),
                'reference_error': failures[0], 'compiled_error': failures[1],
                'maximum_voltage_difference_mV': None, 'elapsed_s': elapsed}
    difference = float(np.max(np.abs(traces[0]-traces[1])))
    return {'passed': difference <= 1e-5, 'failure_support_matches': True,
            'reference_error': None, 'compiled_error': None,
            'maximum_voltage_difference_mV': difference, 'elapsed_s': elapsed}


def main():
    reg = yaml.safe_load(REG.read_text())
    fit = yaml.safe_load(Path(reg['fit_protocol']).read_text())
    if digest(reg['fit_protocol']) != reg['fit_protocol_sha256']:
        raise ValueError('sealed parameter table source changed')
    pins = {str(REG): digest(REG), str(Path(__file__)): digest(__file__)}
    for name in ('patchseq_hh_compiled_prototype.py', 'patchseq_hh_replay_kernel.cpp', 'patchseq_hh_independent_replay.py'):
        p = Path('scripts')/name; pins[str(p)] = digest(p)
    bounds = fit['search']['bounds']
    samples = qmc.Sobol(len(FIELDS), scramble=False).random_base2(7)
    rows = [{name: float(bounds[name][0]+sample[j]*(bounds[name][1]-bounds[name][0]))
             for j, name in enumerate(FIELDS)} for sample in samples]
    rate, results = reg['sample_rate_hz'], []
    with compiled_engine() as engine:
        for index, row in enumerate(rows):
            for amplitude in reg['short_currents_pA']:
                current = np.zeros((int(rate*.1), 1))
                current[int(rate*.02):int(rate*.06)] = amplitude
                result = compare_case(engine, row, current, rate)
                results.append({'kind': 'short-boundary', 'candidate_index': index,
                                'amplitude_pA': amplitude, **result})
            if (index+1) % 16 == 0:
                print('Synthetic boundary candidates checked', index+1, '/ 128', flush=True)
        standard = reg['long_parameters']
        for rate in reg['long_rates_hz']:
            current = np.full((int(rate*4), 1), 20.)
            current[int(rate):int(rate*2)] += 500.
            current[int(rate*2):int(rate*3)] -= 110.
            results.append({'kind': 'long-step-and-return', 'rate_hz': rate,
                            **compare_case(engine, standard, current, rate)})
    if any(digest(p) != h for p, h in pins.items()):
        raise ValueError('synthetic equivalence source changed')
    destination = Path(reg['output'])
    destination.parent.mkdir()
    with destination.open('x') as file:
        yaml.safe_dump({'status': 'synthetic-compiled-equivalence-retained', 'source_hashes': pins,
                        'cases': results, 'all_cases_pass': all(r['passed'] for r in results),
                        'biological_data_used': False, 'compiled_biological_replay_enabled': False},
                       file, sort_keys=False)
    print('Synthetic equivalence completed; optional backend remains disabled', flush=True)


if __name__ == '__main__':
    main()
