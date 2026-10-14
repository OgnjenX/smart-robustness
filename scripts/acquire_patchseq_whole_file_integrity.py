"""Acquire one sealed complete binary for hashing only, without HDF5 access."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import yaml
from patchseq_bounded_NWB import select_url
from run_patchseq_NWB_metadata_pilot import Transport, sha

REG = Path('docs/validation-results/post2008-patchseq-whole-file-integrity-registration-1284.yaml')


def check_author_anchor(reg, manifest, references, raw):
    asset = reg['asset']
    candidates = [x for x in references['assets'] if x['asset_id'] == asset['asset_id']]
    if len(candidates) != 1 or candidates[0]['path'] != asset['path']:
        raise ValueError('unique matching external file reference required')
    refs = candidates[0]['manifest_candidates']
    if (len(refs) != 1 or refs[0]['manifest_source_row_number'] != reg['author_source_row_number']
        or candidates[0]['complete_external_file_reference_agreement'] is not True):
        raise ValueError('author source row reference differs')
    matches = [x for x in manifest['rows'] if x['source_row_number'] == reg['author_source_row_number']]
    if len(matches) != 1:
        raise ValueError('unique author source row required')
    cells = matches[0]['cells']
    if any(cells[k] is None or cells[k]['formula_present'] for k in ('file_size', 'file_name', 'sha256_checksum')):
        raise ValueError('author anchor absent or formula-derived')
    if (cells['file_size']['raw_value'] != str(asset['size'])
        or cells['file_name']['literal_text'] != asset['path'].split('/')[-1]
        or cells['sha256_checksum']['literal_text'] != reg['expected_SHA256']
        or raw['digest'].get('dandi:sha2-256') != reg['expected_SHA256']
        or raw.get('contentSize') != asset['size']):
        raise ValueError('literal author/archive anchor disagreement')


def main():
    reg = yaml.safe_load(REG.read_text())
    pins = {REG: sha(REG), Path(__file__): sha(__file__)}
    for key in ('typed_author_manifest', 'file_reference', 'independent_session',
                'original_archive_metadata', 'pilot', 'transport_runner'):
        pins[Path(reg[key])] = reg[key+'_sha256']
    pins[Path('scripts/patchseq_bounded_NWB.py')] = '722128c416bf640247e5eb9aac1284c16d570e3785452ed4037c1a7392019206'
    if any(sha(p) != digest for p, digest in pins.items()):
        raise ValueError('sealed prerequisite changed')
    def load(key):
        return yaml.load(Path(reg[key]).read_text(), Loader=yaml.CSafeLoader)
    raw = json.loads(Path(reg['original_archive_metadata']).read_bytes())
    check_author_anchor(reg, load('typed_author_manifest'), load('file_reference'), raw)
    session, pilot = load('independent_session'), load('pilot')
    if session['status'] != 'general-session-scalar-independently-reconstructed' or session['literal_path_session_match'] is not True:
        raise ValueError('independent session gate absent')
    if pilot['asset'] != reg['asset']:
        raise ValueError('not the same source-blind pilot')
    etags = {r['header_fields']['etag'][0] for r in pilot['HTTP_receipts'] if r['kind'] == 'RANGE'}
    if len(etags) != 1:
        raise ValueError('pilot ETag ambiguous')
    url, _ = select_url(raw, reg['URL_resolution'])
    root = Path(reg['output_directory'])
    root.mkdir()  # Exclusive output: never duplicate or restart terminal acquisition.
    transport = Transport(root, url, reg['asset'], reg)
    report = {'status': 'whole-file-integrity-acquisition-failure-retained', 'asset': reg['asset'],
        'source_hashes': {str(p): digest for p, digest in pins.items()},
        'started_at_utc': datetime.now(UTC).isoformat(), 'expected_SHA256': reg['expected_SHA256'],
        'HDF5_opened': False, 'physiological_arrays_interpreted': False, 'specimen_eligibility_established': False}
    try:
        transport.head()
        if transport.etag != etags.pop():
            raise ValueError('immutable pilot object changed')
        _, _, body = transport.request('RANGE', 0, reg['asset']['size']-1)
        transport.check_deadline()
        digest = sha(body)
        report.update(body_path=str(body), complete_file_SHA256=digest,
                      author_SHA256_match=digest == reg['expected_SHA256'])
        report['status'] = ('whole-file-integrity-acquisition-complete-independent-verification-pending'
                            if report['author_SHA256_match'] else 'whole-file-integrity-checksum-mismatch-retained')
    except (OSError, ValueError, TypeError, KeyError, TimeoutError, subprocess.TimeoutExpired) as exc:
        report['error'] = {'type': type(exc).__name__, 'message': str(exc)}
    finally:
        report.update(HTTP_receipts=transport.records, finished_at_utc=datetime.now(UTC).isoformat(),
                      pinned_sources_unchanged=all(sha(p) == digest for p, digest in pins.items()))
        with (root/'manifest.yaml').open('x') as handle:
            yaml.safe_dump(report, handle, sort_keys=False)
    print(report['status'])


if __name__ == '__main__':
    main()
