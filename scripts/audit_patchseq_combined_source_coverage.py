"""Necessary joint source coverage, retaining both batches and all failures."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml
from audit_patchseq_fixed_subset_ceiling import audit
from verify_patchseq_whole_file_integrity import digest

REG = Path('docs/validation-results/post2008-patchseq-combined-coverage-registration-1404.yaml')
STRATA = {(group, layer, partition) for group in ('Pvalb', 'Sst', 'Vip')
          for layer in ('L23', 'L5') for partition in ('fitting', 'held-out')}


def combine(batches, expected_strata=STRATA):
    merged_slots, merged_waves, merged_quality, reports = [], [], [], []
    group_partitions, paths, assets, batch_names = {}, set(), set(), set()
    for name, slots, waves, health in batches:
        if name in batch_names:
            raise ValueError('duplicate batch name')
        batch_names.add(name)
        reports.append({'batch': name, **audit(slots, waves, health, expected_strata)})
        by_wave, by_health = {r['slot']: r for r in waves}, {r['slot']: r for r in health}
        for source in slots:
            if source['file_path'] in paths or source['asset_id'] in assets:
                raise ValueError('recording duplicated across batches')
            paths.add(source['file_path'])
            assets.add(source['asset_id'])
            gid, partition = source['frozen_group_id'], source['partition']
            if partition not in ('fitting', 'held-out') or group_partitions.setdefault(gid, partition) != partition:
                raise ValueError('global frozen group crosses partitions')
            original_slot, merged_slot = source['slot'], len(merged_slots)+1
            if by_health[original_slot]['mandatory_gate'] not in ('pass', 'fail', 'unknown'):
                raise ValueError('unknown quality classification')
            selected, wave, quality = (copy.deepcopy(value) for value in
                                       (source, by_wave[original_slot], by_health[original_slot]))
            selected.update(slot=merged_slot, source_batch=name, original_slot=original_slot)
            wave['slot'], quality['slot'] = merged_slot, merged_slot
            merged_slots.append(selected)
            merged_waves.append(wave)
            merged_quality.append(quality)
    combined = audit(merged_slots, merged_waves, merged_quality, expected_strata)
    for case, selected in zip(combined['records'], merged_slots, strict=True):
        case.update(source_batch=selected['source_batch'], original_slot=selected['original_slot'])
    return {'batch_reports': reports, 'combined': combined, 'recordings_retained': len(merged_slots),
            'unique_global_groups': len(group_partitions), 'original_failures_removed': 0,
            'five_three_counts_used_as_fit_minimum': False, 'neuron_model_eligibility_established': False}


def main():
    reg = yaml.safe_load(REG.read_text())
    pins = {REG: digest(REG), Path(__file__): digest(__file__),
            Path('scripts/audit_patchseq_fixed_subset_ceiling.py'): '10fd425b39bd2a2275647e2cfd512ae2475429c35e22fca1652b7af7767a4b93'}
    for batch in reg['batches']:
        for key in ('selection', 'waveforms', 'independent_waveforms', 'independent_quality'):
            pins[Path(batch[key])] = batch[key+'_sha256']
    if any(digest(p) != h for p, h in pins.items()):
        raise ValueError('registered combined source changed')
    inputs = []
    for batch in reg['batches']:
        data = {key: yaml.load(Path(batch[key]).read_text(), Loader=yaml.CSafeLoader)
                for key in ('selection', 'waveforms', 'independent_waveforms', 'independent_quality')}
        if (data['independent_waveforms']['status'] != 'all-array-QC-outcomes-independently-reconstructed'
            or data['independent_quality']['status'] != 'all-recording-quality-arrays-measurements-and-mandatory-gates-independently-reconstructed'
            or len(data['selection']['slots']) != 48 or len(data['independent_waveforms']['records']) != 48):
            raise ValueError('independent evidence or fixed scope incomplete')
        inputs.append((batch['name'], data['selection']['slots'], data['waveforms']['records'],
                       data['independent_quality']['records']))
    result = combine(inputs)
    if result['recordings_retained'] != 96 or any(digest(p) != h for p, h in pins.items()):
        raise ValueError('combined scope or integrity differs')
    output = Path(reg['output'])
    output.parent.mkdir()
    with output.open('x') as handle:
        yaml.safe_dump({'status': 'combined-necessary-source-coverage-audited-not-model-eligibility',
                       'source_hashes': {str(p): h for p, h in pins.items()}, **result,
                       'model_fitting_performed': False, 'QC_changed': False,
                       'frozen_baseline_modified': False}, handle, sort_keys=False)
    print('Combined source coverage audited with all original failures retained')


if __name__ == '__main__':
    main()
