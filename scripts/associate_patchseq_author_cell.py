"""Registered external author-cell association; no RNA aliases or fitting gates."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

import yaml
from verify_patchseq_whole_file_integrity import digest

REG = Path('docs/validation-results/post2008-patchseq-author-cell-association-registration-1294.yaml')
REG_HASH = '8472f00e58dcf096ff5dd9d31b8bb6f0a6e055c77c901bf870327772ca6cbf5d'


def canonical_typed(cell):
    if cell is None or cell['formula_present']:
        return None
    storage = cell['storage_type']
    if storage in ('s', 'inlineStr'):
        text = cell['literal_text']
    elif storage in (None, 'n'):
        text = cell['raw_value']
        if type(text) is not str or len(text) > 15:
            return None
    else:
        return None
    return int(text) if type(text) is str and re.fullmatch('[1-9][0-9]*', text) else None


def associate(asset_id, integrity, session, references, manifest, annotations):
    result = {'status': 'unresolved-no-association', 'failures': [], 'asset_id': asset_id,
        'internal_cell_ID_proven': False, 'RNA_alias_equivalence_established': False,
        'donor_equivalence_established': False, 'mapping_confidence_assigned': False,
        'specimen_eligibility_established': False}
    assets = [r for r in references['assets'] if r['asset_id'] == asset_id]
    result['asset_candidates'] = assets
    if len(assets) != 1:
        result['failures'].append('asset-reference-not-unique')
        return result
    asset = assets[0]
    candidates = asset['manifest_candidates']
    rows = [r for r in manifest['rows'] if r['source_row_number'] in {c['manifest_source_row_number'] for c in candidates}]
    fields = ('file_name', 'file_size', 'sha256_checksum', 'cell_specimen_id')
    result['author_manifest_candidates'] = [{'source_row_number': r['source_row_number'],
        'registered_cells': {k: r['cells'][k] for k in fields}} for r in rows]
    if len(candidates) != 1 or len(rows) != 1 or asset['complete_external_file_reference_agreement'] is not True:
        result['failures'].append('author-file-reference-not-unique-or-complete')
        return result
    row, cells = rows[0], rows[0]['cells']
    cell_id, size = canonical_typed(cells['cell_specimen_id']), canonical_typed(cells['file_size'])
    if cell_id is None or size is None:
        result['failures'].append('author-cell-or-size-not-canonical')
        return result
    if (type(row['canonical_integers']['cell_specimen_id']) is not int
        or type(row['canonical_integers']['file_size']) is not int
        or cell_id != row['canonical_integers']['cell_specimen_id'] or size != row['canonical_integers']['file_size']):
        result['failures'].append('typed-canonical-projection-disagrees')
    for key in ('file_name', 'sha256_checksum'):
        if cells[key] is None or cells[key]['formula_present'] or type(cells[key]['literal_text']) is not str:
            result['failures'].append('author-text-missing-or-formula:'+key)
    if result['failures']:
        return result
    metadata = session['inspection']['datasets']['/general/session_id']
    scalar = metadata.get('scalar', {})
    internal = scalar.get('decoded_text') if scalar.get('raw_type') == 'bytes' else scalar.get('value') if scalar.get('raw_type') == 'str' else None
    if internal is None or re.fullmatch('[1-9][0-9]*', internal) is None:
        result['failures'].append('internal-session-not-canonical-text')
        return result
    cell_rows = [r for r in annotations['July_rows'] if r['canonical_ids']['cell_specimen_id'] == cell_id]
    session_rows = [r for r in annotations['July_rows'] if str(r['canonical_ids']['ephys_session_id']) == internal]
    result['July_candidate_row_numbers'] = {'by_cell': [r['source_row_number'] for r in cell_rows],
                                           'by_session': [r['source_row_number'] for r in session_rows]}
    if (len(cell_rows) != 1 or len(session_rows) != 1
        or cell_rows[0]['source_row_number'] != session_rows[0]['source_row_number']):
        result['failures'].append('July-cell-and-session-not-the-same-unique-row')
        return result
    july = cell_rows[0]
    if (type(july['canonical_ids']['cell_specimen_id']) is not int
        or type(july['canonical_ids']['ephys_session_id']) is not int
        or july['raw_fields']['cell_specimen_id'] != str(cell_id)
        or july['raw_fields']['ephys_session_id'] != internal):
        result['failures'].append('July-raw-identifiers-disagree-with-canonical-keys')
    candidate = candidates[0]
    if (asset['July_session_source_row_numbers'] != [july['source_row_number']]
        or candidate['July_cell_source_row_numbers'] != [july['source_row_number']]
        or candidate['manifest_cell_agrees_with_unique_July_session'] is not True
        or candidate['unique_July_cell_and_session_same_row'] is not True):
        result['failures'].append('independently-verified-file-reference-row-disagrees')
    path_session = asset['path'].split('_ses-')[1].removesuffix('_icephys.nwb')
    if (internal != path_session or session['literal_path_session_match'] is not True
        or cells['file_name']['literal_text'] != asset['path'].split('/')[-1]
        or cells['sha256_checksum']['literal_text'] != integrity['complete_file_SHA256']
        or size != integrity['complete_body_bytes']):
        result['failures'].append('file-content-name-size-or-internal-session-disagrees')
    if not result['failures']:
        result.update(status='externally-author-linked-checksum-and-internal-session-verified-cell',
            association={'source_cell_specimen_id': str(cell_id), 'recording_session_id': internal,
                'author_manifest_source_row_number': row['source_row_number'],
                'July_source_row_number': july['source_row_number'], 'file_path': asset['path'],
                'complete_file_SHA256': integrity['complete_file_SHA256'],
                'link_evidence': 'external-author-file-reference-plus-computed-content-hash-plus-internal-session'})
    return result


def main():
    if digest(REG) != REG_HASH:
        raise ValueError('sealed association registration changed')
    reg = yaml.safe_load(REG.read_text())
    keys = ('integrity', 'session', 'file_references', 'typed_author_manifest', 'July_annotations')
    pins = {REG: REG_HASH, Path(__file__): digest(__file__)}
    pins.update({Path(reg[k]): reg[k+'_sha256'] for k in keys})
    if any(digest(p) != value for p, value in pins.items()):
        raise ValueError('registered prerequisite changed')
    data = {k: yaml.load(Path(reg[k]).read_text(), Loader=yaml.CSafeLoader) for k in keys}
    if (data['integrity']['status'] != 'complete-file-author-checksum-and-cache-offsets-independently-verified'
        or data['session']['status'] != 'general-session-scalar-independently-reconstructed'):
        raise ValueError('independent content/session gate missing')
    for key in ('integrity', 'session'):
        pins.update({Path(p): value for p, value in data[key]['source_hashes'].items()})
    body = Path(data['integrity']['complete_body_path'])
    pins[body] = data['integrity']['complete_file_SHA256']
    if any(digest(p) != value for p, value in pins.items()):
        raise ValueError('source or complete recording changed')
    output = Path(reg['output'])
    # Exclusive directory creation prevents duplicates and terminal overwrites.
    output.parent.mkdir()
    result = associate(reg['asset_id'], *(data[k] for k in keys))
    if any(digest(p) != value for p, value in pins.items()):
        raise ValueError('source changed during association')
    result.update(source_hashes={str(p): value for p, value in pins.items()},
                  finished_at_utc=datetime.now(UTC).isoformat(), network_requests=0,
                  physiological_arrays_read=False, RNA_expression_values_read=False)
    with output.open('x') as handle:
        yaml.safe_dump(result, handle, sort_keys=False)
    print(result['status'])


if __name__ == '__main__':
    main()
