"""Independent raw-key reconstruction of a manifest-anchored cell association."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import yaml
from verify_patchseq_whole_file_integrity import digest, same

REG = Path('docs/validation-results/post2008-patchseq-author-cell-association-registration-1294.yaml')
RESULT = Path('results/patchseq-author-cell-association-1295/manifest.yaml')
OUTPUT = Path('results/patchseq-author-cell-association-independent-1296/manifest.yaml')


def positive_lexeme(cell):
    if cell is None or cell['formula_present']:
        raise ValueError('missing/formula identity')
    if cell['storage_type'] in ('s', 'inlineStr'):
        value = cell['literal_text']
    elif cell['storage_type'] in (None, 'n'):
        value = cell['raw_value']
        if type(value) is not str or len(value) > 15:
            raise ValueError('invalid numeric source lexeme')
    else:
        raise ValueError('unapproved identity storage')
    if type(value) is not str or not value.isascii() or not value.isdecimal() or value.startswith('0'):
        raise ValueError('noncanonical identity lexeme')
    return value


def reconstruct(reg, integrity, session, references, manifest, july):
    assets = [r for r in references['assets'] if r['asset_id'] == reg['asset_id']]
    if len(assets) != 1 or assets[0]['complete_external_file_reference_agreement'] is not True:
        raise ValueError('unique full external reference missing')
    ref = assets[0]
    if len(ref['manifest_candidates']) != 1:
        raise ValueError('author reference ambiguous')
    source_number = ref['manifest_candidates'][0]['manifest_source_row_number']
    source_rows = [r for r in manifest['rows'] if r['source_row_number'] == source_number]
    if len(source_rows) != 1:
        raise ValueError('author row missing/ambiguous')
    row = source_rows[0]
    cell = positive_lexeme(row['cells']['cell_specimen_id'])
    size = positive_lexeme(row['cells']['file_size'])
    same(int(cell), row['canonical_integers']['cell_specimen_id'])
    same(int(size), row['canonical_integers']['file_size'])
    for key in ('file_name', 'sha256_checksum'):
        if row['cells'][key] is None or row['cells'][key]['formula_present']:
            raise ValueError('author filename/hash missing or formula')
    scalar = session['inspection']['datasets']['/general/session_id']['scalar']
    if scalar['raw_type'] == 'bytes':
        internal = bytes.fromhex(scalar['raw_hex']).decode('utf-8')
        same(internal, scalar['decoded_text'])
    elif scalar['raw_type'] == 'str':
        internal = scalar['value']
    else:
        raise ValueError('text session required')
    if not internal.isascii() or not internal.isdecimal() or internal.startswith('0'):
        raise ValueError('noncanonical internal session')
    # Raw CSV-derived keys rather than the collector's canonical-key selection.
    by_cell = [r for r in july['July_rows'] if r['raw_fields']['cell_specimen_id'] == cell]
    by_session = [r for r in july['July_rows'] if r['raw_fields']['ephys_session_id'] == internal]
    if len(by_cell) != 1 or len(by_session) != 1 or by_cell[0]['source_row_number'] != by_session[0]['source_row_number']:
        raise ValueError('raw cell/session keys do not identify one row')
    annotation = by_cell[0]
    same(int(cell), annotation['canonical_ids']['cell_specimen_id'])
    same(int(internal), annotation['canonical_ids']['ephys_session_id'])
    candidate = ref['manifest_candidates'][0]
    same(ref['July_session_source_row_numbers'], [annotation['source_row_number']])
    same(candidate['July_cell_source_row_numbers'], [annotation['source_row_number']])
    if candidate['manifest_cell_agrees_with_unique_July_session'] is not True or candidate['unique_July_cell_and_session_same_row'] is not True:
        raise ValueError('prior source row proof differs')
    if (ref['path'].split('_ses-')[1].removesuffix('_icephys.nwb') != internal
        or row['cells']['file_name']['literal_text'] != ref['path'].split('/')[-1]
        or row['cells']['sha256_checksum']['literal_text'] != integrity['complete_file_SHA256']
        or int(size) != integrity['complete_body_bytes'] or session['literal_path_session_match'] is not True):
        raise ValueError('content/session source anchor differs')
    return {'source_cell_specimen_id': cell, 'recording_session_id': internal,
        'author_manifest_source_row_number': source_number, 'July_source_row_number': annotation['source_row_number'],
        'file_path': ref['path'], 'complete_file_SHA256': integrity['complete_file_SHA256'],
        'link_evidence': 'external-author-file-reference-plus-computed-content-hash-plus-internal-session'}, assets, row, annotation


def main():
    pins = {REG: '8472f00e58dcf096ff5dd9d31b8bb6f0a6e055c77c901bf870327772ca6cbf5d',
        RESULT: '3b9abcc02a07b189fdb10b018c0709dba1d214f7de18ab70b517cb351d35e7bd',
        Path('scripts/associate_patchseq_author_cell.py'): 'e73761244e71ce326afe460a65aaede17877a26d941c63e13774410da6209f8a'}
    if any(digest(p) != value for p, value in pins.items()):
        raise ValueError('association source changed')
    reg, report = yaml.safe_load(REG.read_text()), yaml.safe_load(RESULT.read_text())
    pins.update({Path(p): value for p, value in report['source_hashes'].items()})
    if any(digest(p) != value for p, value in pins.items()):
        raise ValueError('prerequisite or binary changed')
    if (report['status'] != reg['outcomes']['success_label'] or report['failures']
        or any(report[k] is not False for k in ('internal_cell_ID_proven', 'RNA_alias_equivalence_established',
            'donor_equivalence_established', 'mapping_confidence_assigned', 'specimen_eligibility_established',
            'physiological_arrays_read', 'RNA_expression_values_read')) or report['network_requests'] != 0):
        raise ValueError('reported association scope differs')
    keys = ('integrity', 'session', 'file_references', 'typed_author_manifest', 'July_annotations')
    data = [yaml.load(Path(reg[k]).read_text(), Loader=yaml.CSafeLoader) for k in keys]
    if data[0]['status'] != 'complete-file-author-checksum-and-cache-offsets-independently-verified' or data[1]['status'] != 'general-session-scalar-independently-reconstructed':
        raise ValueError('independent prerequisites missing')
    association, assets, row, annotation = reconstruct(reg, *data)
    same(association, report['association'])
    same(assets, report['asset_candidates'])
    fields = reg['registered_fields']['author_manifest']
    same([{'source_row_number': row['source_row_number'], 'registered_cells': {k: row['cells'][k] for k in fields}}], report['author_manifest_candidates'])
    same({'by_cell': [annotation['source_row_number']], 'by_session': [annotation['source_row_number']]}, report['July_candidate_row_numbers'])
    if any(digest(p) != value for p, value in pins.items()):
        raise ValueError('sources changed during independent reconstruction')
    OUTPUT.parent.mkdir()
    with OUTPUT.open('x') as handle:
        yaml.safe_dump({'status': 'external-author-cell-association-independently-reconstructed',
            'verifier_sha256': digest(__file__), 'source_hashes': {str(p): value for p, value in pins.items()},
            'association': association, 'verified_at_utc': datetime.now(UTC).isoformat(),
            'association_collector_imported': False, 'network_requests': 0,
            'RNA_alias_equivalence_established': False, 'internal_cell_ID_proven': False,
            'mapping_confidence_assigned': False, 'specimen_eligibility_established': False}, handle, sort_keys=False)
    print('External author cell association independently reconstructed from raw identifier keys')


if __name__ == '__main__':
    main()
