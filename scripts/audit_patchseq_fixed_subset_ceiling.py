"""Necessary source coverage only; does not invent fitting-support minimums."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import yaml
from verify_patchseq_whole_file_integrity import digest

PINS = {
    'results/patchseq-intrinsic-subset-1309/manifest.yaml': 'dcf2bdf39e44c82190b63aa62de46df84314ffec47f76e01e85726cdbd9088d3',
    'results/patchseq-waveform-QC-1328/manifest.yaml': '13e054b090b22b45196bb2571d63cee4577c4e169a2a3c788fb4efbef3683564',
    'results/patchseq-waveform-QC-independent-1330/manifest.yaml': '7f2db295a590d818f0334684caa913e366b6452a76e7562274c3934ad2cefbeb',
    'results/patchseq-recording-quality-independent-1368/manifest.yaml': 'a49a7928668e86dba714ed4f006063d89aa3592d6c44ec73e9c869f52b1de427',
}
OUTPUT = Path('results/patchseq-fixed-subset-ceiling-1370/manifest.yaml')


def audit(slots, waveforms, quality, expected_strata=None):
    waves = {c['slot']: c for c in waveforms}
    health = {c['slot']: c for c in quality}
    if len({s['slot'] for s in slots}) != len(slots) or len(waves) != len(waveforms) or len(health) != len(quality) or set(waves) != {s['slot'] for s in slots} or set(health) != set(waves):
        raise ValueError('case scope or duplicates differ')
    groups = defaultdict(lambda: {'selected': set(), 'any_wave_QC': set(), 'joint': set()})
    if expected_strata is not None:
        for key in expected_strata:
            groups[key]  # retain even an entirely absent design stratum
    cases = []
    for s in slots:
        cell = waves[s['slot']]
        if cell['source_cell_specimen_id'] != s['source_cell_specimen_id']:
            raise ValueError('cell identity differs')
        key = (s['source_group'], s['layer_label'], s['partition'])
        if expected_strata is not None and key not in expected_strata:
            raise ValueError('unexpected design stratum')
        group = s['frozen_group_id']
        groups[key]['selected'].add(group)
        any_wave = any(t['sweep_QC_pass'] for t in cell['sweeps'])
        health_pass = health[s['slot']]['mandatory_gate'] == 'pass'
        if any_wave:
            groups[key]['any_wave_QC'].add(group)
        if any_wave and health_pass:
            groups[key]['joint'].add(group)
        cases.append({'slot': s['slot'], 'source_group': s['source_group'], 'layer': s['layer_label'],
            'partition': s['partition'], 'frozen_group_id': group,
            'any_passing_waveform': any_wave, 'mandatory_recording_gate_pass': health_pass})
    rows = [{'source_group': key[0], 'layer': key[1], 'partition': key[2],
             **{name+'_groups': len(values) for name, values in counts.items()}}
            for key, counts in sorted(groups.items())]
    absent = [r for r in rows if r['joint_groups'] == 0]
    return {'records': cases, 'strata': rows, 'strata_without_any_joint_supported_group': absent,
            'all_strata_nonempty_after_fixed_QC': not absent,
            'fit_target_diversity_established': False, 'specimen_eligibility_established': False}


def main():
    if any(digest(p) != h for p, h in PINS.items()):
        raise ValueError('sealed source changed')
    data = {p: yaml.load(Path(p).read_text(), Loader=yaml.CSafeLoader) for p in PINS}
    slots = data['results/patchseq-intrinsic-subset-1309/manifest.yaml']['slots']
    wave = data['results/patchseq-waveform-QC-1328/manifest.yaml']['records']
    independent_wave = data['results/patchseq-waveform-QC-independent-1330/manifest.yaml']
    health = data['results/patchseq-recording-quality-independent-1368/manifest.yaml']
    if len(slots) != 48 or len(independent_wave['records']) != 48 or len(health['records']) != 48:
        raise ValueError('fixed 48-case scope differs')
    expected = {(c, layer, partition) for c in ('Pvalb', 'Sst', 'Vip')
                for layer in ('L23', 'L5') for partition in ('fitting', 'held-out')}
    result = audit(slots, wave, health['records'], expected)
    if any(digest(p) != h for p, h in PINS.items()):
        raise ValueError('source changed during coverage audit')
    OUTPUT.parent.mkdir()
    with OUTPUT.open('x') as handle:
        yaml.safe_dump({'status': 'fixed-subset-necessary-coverage-ceiling-not-model-failure',
            'auditor_sha256': digest(__file__), 'source_hashes': PINS, **result,
            'five_three_acquisition_counts_used_as_fit_minimum': False,
            'QC_changed': False, 'source_replacements': 0, 'model_fitting_performed': False}, handle, sort_keys=False)
    print('Recorded necessary coverage ceiling without altering QC, cells, partitions or fitting criteria')


if __name__ == '__main__':
    main()
