"""Exact raw annotation availability counts; no cell-class or layer inference."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import yaml

REG = Path('docs/validation-results/post2008-patchseq-raw-label-inventory-registration-1190.yaml')
FIELDS = ('structure', 'corresponding_AIT2.3.1_ID', 'corresponding_AIT2.3.1_alias')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(65536), b''):
            h.update(block)
    return h.hexdigest()


def inventory(identity, references):
    by_asset = {a['asset_id']: a for a in identity['assets']}
    covered = set()
    for ref in references['assets']:
        if ref['source_cell_ID_agrees'] is True:
            parents = by_asset[ref['asset_id']]['annotation_candidates']
            if len(parents) != 1:
                raise ValueError('agreeing reference has ambiguous annotations')
            covered.add(parents[0]['source_row_number'])
    rows = identity['annotation_rows']
    if not covered <= {r['source_row_number'] for r in rows}:
        raise ValueError('covered annotation row absent')
    groups = {'all_source_rows': rows, 'official_manifest_covered_rows': [r for r in rows if r['source_row_number'] in covered]}
    result = {}
    for name, records in groups.items():
        if any(type(r['raw_fields'][k]) is not str for r in records for k in FIELDS):
            raise ValueError('exact raw string annotations required')
        result[name] = {'rows': len(records), 'raw_value_counts': {
            k: dict(sorted(Counter(r['raw_fields'][k] for r in records).items())) for k in FIELDS}}
    return {**result, 'biological_eligibility_established': False,
            'cell_class_or_layer_qualification_performed': False, 'mapping_confidence_imputed': False}


def main():
    if digest(REG) != '117a21b2634a8afc2de1e2ecc25dcf84ab32b85319fd0bebb115768af10e21d1':
        raise ValueError('sealed label inventory registration changed')
    reg = yaml.safe_load(REG.read_text())
    pins = {Path(reg[k]): reg[k + '_sha256'] for k in ('identity_result', 'cross_reference', 'verification')}
    if any(digest(p) != h for p, h in pins.items()):
        raise ValueError('pinned source evidence changed')
    load = lambda path: yaml.load(Path(path).read_text(), Loader=yaml.CSafeLoader)
    if load(reg['verification'])['status'] != 'file-cross-references-independently-verified':
        raise ValueError('independent source cross-reference gate absent')
    result = inventory(load(reg['identity_result']), load(reg['cross_reference']))
    if any(digest(p) != h for p, h in pins.items()):
        raise ValueError('source changed during label inventory')
    result.update(registration_sha256=digest(REG), collector_sha256=digest(__file__),
                  source_hashes={str(p): h for p, h in pins.items()}, network_requests=0,
                  physiological_arrays_read=False, parameter_fitting_performed=False)
    with Path(reg['output']).open('x') as handle:
        yaml.safe_dump(result, handle, sort_keys=False)
    print({k: v['rows'] for k, v in result.items() if k.endswith('_rows')})


if __name__ == '__main__':
    main()
