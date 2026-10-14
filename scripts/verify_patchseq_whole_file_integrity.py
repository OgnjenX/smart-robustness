"""Independent binary hashing, header parsing and cached-block reconciliation."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit

import yaml

REG = Path('docs/validation-results/post2008-patchseq-whole-file-integrity-registration-1284.yaml')
RESULT = Path('results/patchseq-whole-file-integrity-1285/manifest.yaml')
OUTPUT = Path('results/patchseq-whole-file-integrity-independent-1287/manifest.yaml')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while block := handle.read(1048576):
            value.update(block)
    return value.hexdigest()


def same(a, b):
    if json.dumps(a, sort_keys=True) != json.dumps(b, sort_keys=True):
        raise ValueError('type-preserving comparison differs')


def allowed(url, policy):
    parsed = urlsplit(url)
    try:
        port = parsed.port
    except ValueError:
        return False
    return (parsed.scheme == 'https' and parsed.hostname in policy['allowed_hosts']
            and parsed.username is None and parsed.password is None and port in (None, 443)
            and not parsed.fragment and parsed.path.startswith('/blobs/')
            and not any(p in ('.', '..') for p in unquote(parsed.path).split('/')))


def verify_transport(report, reg, locator, etag):
    pins, head, ranges, head_ok, body = {}, 0, 0, False, None
    size = reg['asset']['size']
    for receipt in report['HTTP_receipts']:
        parsed = urlsplit(locator)
        if receipt['URL_host'] != parsed.hostname or receipt['URL_path'] != parsed.path:
            raise ValueError('receipt target differs')
        for kind in ('headers', 'body'):
            path = Path(receipt[kind+'_path'])
            if path.stat().st_size != receipt[kind+'_bytes'] or digest(path) != receipt[kind+'_sha256']:
                raise ValueError('raw receipt differs')
            pins[path] = receipt[kind+'_sha256']
        header_path = Path(receipt['headers_path'])
        if header_path.stat().st_size > 65536:
            raise ValueError('header bound exceeded')
        lines = header_path.read_text().replace('\r\n', '\n').strip().split('\n\n')[-1].splitlines()
        status, fields = int(lines[0].split()[1]), {}
        for line in lines[1:]:
            key, value = line.split(':', 1)
            fields.setdefault(key.lower(), []).append(value.strip())
        same(fields, receipt['header_fields'])
        if (status != receipt['http_status'] or receipt['client_exit_code'] != 0
            or receipt['status'] != 'response-retained' or fields.get('content-encoding', ['identity']) != ['identity']):
            raise ValueError('retained receipt not successful')
        if receipt['kind'] == 'HEAD':
            head += 1
            if head_ok or head > reg['URL_resolution']['maximum_HEAD_requests']:
                raise ValueError('extra HEAD')
            if status in (301, 302, 307, 308):
                if head > reg['URL_resolution']['maximum_HEAD_redirects'] or len(fields.get('location', [])) != 1:
                    raise ValueError('redirect bound exceeded')
                target = urljoin(locator, fields['location'][0])
                if not allowed(target, reg['URL_resolution']) or urlsplit(target).path != parsed.path:
                    raise ValueError('redirect not permitted')
                locator = target
                continue
            if status != 200 or fields.get('etag') != [etag] or fields.get('content-length') != [str(size)]:
                raise ValueError('immutable HEAD differs')
            head_ok = True
        elif receipt['kind'] == 'RANGE':
            ranges += 1
            if (not head_ok or ranges != 1 or receipt['start'] != 0 or receipt['end'] != size-1
                or status != 206 or fields.get('etag') != [etag]
                or fields.get('content-range') != [f'bytes 0-{size-1}/{size}']
                or receipt['body_bytes'] != size):
                raise ValueError('complete immutable 206 range required')
            body = Path(receipt['body_path'])
        else:
            raise ValueError('unsupported receipt kind')
    if not head_ok or ranges != 1 or size > reg['HTTP_range_limits']['maximum_unique_downloaded_bytes']:
        raise ValueError('transport accounting differs')
    elapsed = (datetime.fromisoformat(report['finished_at_utc'])-datetime.fromisoformat(report['started_at_utc'])).total_seconds()
    if not 0 <= elapsed <= reg['HTTP_range_limits']['maximum_run_seconds']:
        raise ValueError('deadline exceeded')
    return body, pins


def compare_cached_blocks(body, reports, pins):
    starts = set()
    with body.open('rb') as complete:
        for report in reports:
            for receipt in report['HTTP_receipts']:
                if receipt['kind'] != 'RANGE':
                    continue
                source, start, end = Path(receipt['body_path']), receipt['start'], receipt['end']
                if source not in pins or digest(source) != pins[source] or start in starts:
                    raise ValueError('cached span unpinned or duplicated')
                cached = source.read_bytes()
                if len(cached) != end-start+1:
                    raise ValueError('cached span length differs')
                complete.seek(start)
                if complete.read(len(cached)) != cached:
                    raise ValueError('cached bytes differ from complete-file offsets')
                starts.add(start)
    return len(starts)


def main():
    pins = {REG: 'b049c6c715c35354b13cdbc65fe1ad404d897f598edf780aa8139ae2c70d21a0',
            RESULT: 'a6b8c851bce7093182f8466800bbaaa318aea5f0e385823cb95895f0f4ad915d',
            Path('scripts/acquire_patchseq_whole_file_integrity.py'): '92d64b587e34d0360b2b1b88024ce980a4cf1b2cd31faef7b91a000856eab2f9'}
    if any(digest(p) != value for p, value in pins.items()):
        raise ValueError('sealed acquisition changed')
    reg, report = yaml.safe_load(REG.read_text()), yaml.safe_load(RESULT.read_text())
    pins.update({Path(p): value for p, value in report['source_hashes'].items()})
    if any(digest(p) != value for p, value in pins.items()):
        raise ValueError('author or acquisition source changed')
    same(report['asset'], reg['asset'])
    if (report['status'] != 'whole-file-integrity-acquisition-complete-independent-verification-pending'
        or report['pinned_sources_unchanged'] is not True or report['HDF5_opened'] is not False
        or report['physiological_arrays_interpreted'] is not False or report['specimen_eligibility_established'] is not False):
        raise ValueError('acquisition scope differs')
    def load(key):
        return yaml.load(Path(reg[key]).read_text(), Loader=yaml.CSafeLoader)
    raw = json.loads(Path(reg['original_archive_metadata']).read_bytes())
    locators = {url for url in raw['contentUrl'] if allowed(url, reg['URL_resolution'])}
    if len(locators) != 1:
        raise ValueError('unique pinned locator required')
    manifest, references = load('typed_author_manifest'), load('file_reference')
    rows = [r for r in manifest['rows'] if r['source_row_number'] == reg['author_source_row_number']]
    refs = [r for r in references['assets'] if r['asset_id'] == reg['asset']['asset_id']]
    if len(rows) != 1 or len(refs) != 1:
        raise ValueError('author reference ambiguous')
    cells = rows[0]['cells']
    if (any(cells[k] is None or cells[k]['formula_present'] for k in ('file_size', 'file_name', 'sha256_checksum'))
        or cells['file_size']['raw_value'] != str(reg['asset']['size'])
        or cells['file_name']['literal_text'] != reg['asset']['path'].split('/')[-1]
        or cells['sha256_checksum']['literal_text'] != reg['expected_SHA256']
        or raw['digest']['dandi:sha2-256'] != reg['expected_SHA256']
        or raw['contentSize'] != reg['asset']['size']
        or refs[0]['path'] != reg['asset']['path']
        or refs[0]['complete_external_file_reference_agreement'] is not True
        or len(refs[0]['manifest_candidates']) != 1
        or refs[0]['manifest_candidates'][0]['manifest_source_row_number'] != reg['author_source_row_number']):
        raise ValueError('author/archive literal reference differs')
    pilot, session = load('pilot'), load('independent_session')
    same(pilot['asset'], reg['asset'])
    if session['status'] != 'general-session-scalar-independently-reconstructed' or session['literal_path_session_match'] is not True:
        raise ValueError('session prerequisite differs')
    pins.update({Path(p): value for p, value in session['source_hashes'].items()})
    if any(digest(p) != value for p, value in pins.items()):
        raise ValueError('saved pilot evidence changed')
    etags = {r['header_fields']['etag'][0] for r in pilot['HTTP_receipts'] if r['kind'] == 'RANGE'}
    if len(etags) != 1:
        raise ValueError('pilot ETag ambiguous')
    body, receipts = verify_transport(report, reg, locators.pop(), etags.pop())
    pins.update(receipts)
    computed = digest(body)
    if computed != reg['expected_SHA256'] or computed != report['complete_file_SHA256'] or report['author_SHA256_match'] is not True:
        raise ValueError('independent complete-file hash differs')
    if body != Path(report['body_path']):
        raise ValueError('reported complete body differs')
    source = Path('results/patchseq-NWB-structure-range-1275/manifest.yaml')
    if source not in pins:
        raise ValueError('expanded cache manifest not pinned')
    count = compare_cached_blocks(body, [pilot, yaml.safe_load(source.read_text())], pins)
    if count != 38:
        raise ValueError('cached block inventory differs')
    if any(digest(p) != value for p, value in pins.items()):
        raise ValueError('source changed during independent verification')
    OUTPUT.parent.mkdir()
    with OUTPUT.open('x') as handle:
        yaml.safe_dump({'status': 'complete-file-author-checksum-and-cache-offsets-independently-verified',
            'verifier_sha256': digest(__file__), 'source_hashes': {str(p): value for p, value in pins.items()},
            'complete_body_path': str(body), 'complete_body_bytes': body.stat().st_size,
            'complete_file_SHA256': computed, 'cached_blocks_exactly_agree': count,
            'verified_at_utc': datetime.now(UTC).isoformat(), 'network_requests': 0,
            'HDF5_opened': False, 'physiological_arrays_interpreted': False,
            'acquisition_runner_imported': False, 'specimen_eligibility_established': False}, handle, sort_keys=False)
    print('Complete binary hash, author anchor, headers and 38 cached offsets independently verified')


if __name__ == '__main__':
    main()
