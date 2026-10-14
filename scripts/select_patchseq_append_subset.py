"""Select the next metadata-ranked batch without replacing any original case."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml
from select_patchseq_intrinsic_subset import select
from verify_patchseq_whole_file_integrity import digest, same

REG = Path('docs/validation-results/post2008-patchseq-append-selection-registration-1372.yaml')


def append_selection(base, candidates, references, manifest, original):
    reconstructed = select(base, candidates, references, manifest)
    for key in ('slots', 'strata', 'problems', 'summary'):
        same(reconstructed[key], original[key])
    if reconstructed['problems']:
        raise ValueError('original selection preflight not complete')
    excluded = {(s['source_group'], s['layer_label'], s['partition'], s['frozen_group_id'])
                for s in original['slots']}
    remaining = copy.deepcopy(candidates)
    remaining['records'] = [r for r in remaining['records'] if
        (r['source_core_group'], r['target_layer_label'], r['frozen_partition'],
         r['frozen_group_id']) not in excluded]
    result = select(base, remaining, references, manifest)
    if {s['file_path'] for s in result['slots']} & {s['file_path'] for s in original['slots']}:
        raise ValueError('append duplicates original recording')
    if {s['asset_id'] for s in result['slots'] if 'asset_id' in s} & {
        s['asset_id'] for s in original['slots']}:
        raise ValueError('append duplicates original asset')
    partitions = {s['frozen_group_id']: s['partition'] for s in original['slots']}
    for slot in result['slots']:
        if partitions.setdefault(slot['frozen_group_id'], slot['partition']) != slot['partition']:
            raise ValueError('append crosses original frozen partition')
    return result


def load_inputs():
    reg = yaml.safe_load(REG.read_text())
    pins = {REG: digest(REG), Path(reg['base_registration']): reg['base_registration_sha256'],
            Path(reg['original_subset']): reg['original_subset_sha256']}
    base = yaml.safe_load(Path(reg['base_registration']).read_text())
    keys = ('candidates', 'independent_candidates', 'file_references', 'typed_manifest')
    pins.update({Path(base[k]): base[k+'_sha256'] for k in keys})
    pins[Path('scripts/select_patchseq_intrinsic_subset.py')] = '44f8bfe95110bcee15098ab4f32fbbc8471c9ade4fb2b98560a5acf58ec8adb8'
    pins[Path('scripts/read_patchseq_2021_manifest.py')] = '2518610d766da2c0765b2152037cd9e24ad44378132affba538ef3b6ffdee826'
    if any(digest(p) != value for p, value in pins.items()):
        raise ValueError('append source metadata changed')
    data = {k: yaml.load(Path(base[k]).read_text(), Loader=yaml.CSafeLoader) for k in keys}
    if data['independent_candidates']['status'] != 'all-source-candidate-records-and-strata-independently-reconstructed':
        raise ValueError('candidate prerequisite missing')
    original = yaml.load(Path(reg['original_subset']).read_text(), Loader=yaml.CSafeLoader)
    return reg, base, data, original, pins


def main():
    reg, base, data, original, pins = load_inputs()
    pins[Path(__file__)] = digest(__file__)
    result = append_selection(base, data['candidates'], data['file_references'], data['typed_manifest'], original)
    result.update(source_hashes={str(p): h for p, h in pins.items()}, network_requests=0,
                  physiology_accessed=False, recording_downloaded=False,
                  specimen_eligibility_established=False, original_cases_replaced=0)
    if any(digest(p) != h for p, h in pins.items()):
        raise ValueError('append source changed during selection')
    output = Path(reg['output'])
    output.parent.mkdir()
    with output.open('x') as handle:
        yaml.safe_dump(result, handle, sort_keys=False)
    print(result['status'], result['summary'])


if __name__ == '__main__':
    main()
