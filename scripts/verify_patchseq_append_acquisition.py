"""Independent append transport/accounting and identity-only replay; no trace scoring."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import yaml
from verify_patchseq_NWB_metadata_pilot import approved_locator, replay
from verify_patchseq_whole_file_integrity import digest, same

REG = Path('docs/validation-results/post2008-patchseq-append-acquisition-registration-1375.yaml')
OUTPUT = Path('results/patchseq-append-acquisition-independent-1377/manifest.yaml')


def check_admission_and_times(state, slot, directory, receipt):
    attempt = directory/'attempt.yaml'
    if attempt.exists():
        recorded = yaml.safe_load(attempt.read_text())
        same(recorded['slot'], slot)
        epoch = datetime.fromisoformat(recorded['started_at_utc']).timestamp()
        if not state['started_epoch'] <= epoch <= state['deadline_epoch']:
            raise ValueError('slot admitted outside original deadline')
    elif receipt['status'] != 'interrupted-attempt-retained-no-retry':
        raise ValueError('processed slot lacks original admission marker')
    marker = directory/'GET-attempt.yaml'
    charged, admitted = 0, None
    if marker.exists():
        admission = yaml.safe_load(marker.read_text())
        same(admission['slot'], slot['slot'])
        same(admission['charged_bytes'], slot['expected_recording_bytes'])
        admitted = datetime.fromisoformat(admission['admitted_at_utc']).timestamp()
        if not state['started_epoch'] <= admitted <= state['deadline_epoch']:
            raise ValueError('GET admission outside original deadline')
        charged = admission['charged_bytes']
    for event in receipt['HTTP_receipts']:
        begin = datetime.fromisoformat(event['started_at_utc']).timestamp()
        end = datetime.fromisoformat(event['finished_at_utc']).timestamp()
        if not state['started_epoch'] <= begin <= state['deadline_epoch'] or end < begin:
            raise ValueError('HTTP attempt began outside deadline or has negative duration')
        if event['kind'] == 'RANGE' and (admitted is None or begin < admitted):
            raise ValueError('GET began before its immutable byte admission')
        if event['status'] == 'response-retained' and end > state['deadline_epoch']:
            raise ValueError('successful response finished after absolute deadline')
    return charged


def identity_flags(inspection, slot):
    subject = slot['file_path'].split('/')[0][4:]
    session = slot['file_path'].split('_ses-')[1].removesuffix('_icephys.nwb')
    flags, active = {}, []
    for path, expected in [('/general/session_id', session), ('/session_id', session), ('/general/subject/subject_id', subject)]:
        entry = inspection['datasets'][path]
        scalar = entry.get('scalar', {})
        if scalar.get('raw_type') == 'bytes':
            value = bytes.fromhex(scalar['raw_hex']).decode('utf-8')
            same(value, scalar['decoded_text'])
        elif scalar.get('raw_type') == 'str':
            value = scalar['value']
        else:
            value = None
        flags[path] = value == expected if value is not None else None
        if path in ('/general/session_id', '/session_id') and entry['present']:
            active.append(path)
    accepted = len(active) > 0 and all(flags[p] is True for p in active) and flags['/general/subject/subject_id'] is True
    return flags, accepted


def inspect_http(receipts, slot, reg, locator):
    head, gets, immutable, pins, body = 0, 0, None, {}, None
    for receipt in receipts:
        parsed = urlsplit(locator)
        if receipt['URL_host'] != parsed.hostname or receipt['URL_path'] != parsed.path:
            raise ValueError('HTTP target differs from pinned original locator')
        for kind in ('headers', 'body'):
            if kind+'_path' in receipt:
                path = Path(receipt[kind+'_path'])
                if digest(path) != receipt[kind+'_sha256'] or path.stat().st_size != receipt[kind+'_bytes']:
                    raise ValueError('raw HTTP evidence changed')
                pins[path] = receipt[kind+'_sha256']
        if receipt['kind'] == 'HEAD':
            head += 1
            if head > reg['transport']['maximum_HEAD_requests_per_slot'] or immutable is not None:
                raise ValueError('extra HEAD attempt')
        elif receipt['kind'] == 'RANGE':
            gets += 1
            if gets > reg['transport']['maximum_GET_attempts_per_slot'] or immutable is None:
                raise ValueError('GET retry or no successful immutable HEAD')
            same(receipt['start'], 0)
            same(receipt['end'], slot['expected_recording_bytes']-1)
            if receipt.get('body_bytes', 0) > slot['expected_recording_bytes']:
                raise ValueError('GET physical payload exceeds sealed size')
        else:
            raise ValueError('unsupported HTTP attempt kind')
        if receipt['status'] != 'response-retained':
            # Retain failed/partial evidence without calling it a valid response.
            if 'header_fields' in receipt:
                header = Path(receipt['headers_path'])
                if header.stat().st_size > 65536:
                    raise ValueError('failed-response header bound exceeded')
                lines = header.read_text().replace('\r\n', '\n').strip().split('\n\n')[-1].splitlines()
                observed = {}
                for line in lines[1:]:
                    key, value = line.split(':', 1)
                    observed.setdefault(key.lower(), []).append(value.strip())
                same(observed, receipt['header_fields'])
                same(int(lines[0].split()[1]), receipt['http_status'])
            continue
        header = Path(receipt['headers_path'])
        if header.stat().st_size > 65536:
            raise ValueError('HTTP header bound exceeded')
        lines = header.read_text().replace('\r\n', '\n').strip().split('\n\n')[-1].splitlines()
        code, fields = int(lines[0].split()[1]), {}
        for line in lines[1:]:
            key, value = line.split(':', 1)
            fields.setdefault(key.lower(), []).append(value.strip())
        same(fields, receipt['header_fields'])
        if code != receipt['http_status'] or receipt['client_exit_code'] != 0 or fields.get('content-encoding', ['identity']) != ['identity']:
            raise ValueError('saved successful response differs')
        if receipt['kind'] == 'HEAD':
            if code in (301, 302, 307, 308):
                if head > reg['transport']['maximum_HEAD_redirects_per_slot'] or len(fields.get('location', [])) != 1:
                    raise ValueError('HEAD redirect limit exceeded')
                target = urljoin(locator, fields['location'][0])
                if not approved_locator(target, reg['URL_resolution']) or urlsplit(target).path != parsed.path:
                    raise ValueError('HEAD redirect target unapproved')
                locator = target
                continue
            values = fields.get('etag', [])
            if (code != 200 or fields.get('content-length') != [str(slot['expected_recording_bytes'])]
                or len(values) != 1 or len(values[0]) < 3 or not values[0].startswith('"') or not values[0].endswith('"')
                or '"' in values[0][1:-1] or any(ord(c) < 32 for c in values[0])):
                raise ValueError('immutable successful HEAD missing')
            immutable = values[0]
        else:
            size = slot['expected_recording_bytes']
            if (code != 206 or fields.get('etag') != [immutable] or fields.get('content-range') != [f'bytes 0-{size-1}/{size}']
                or receipt['body_bytes'] != size):
                raise ValueError('complete immutable GET differs')
            body = Path(receipt['body_path'])
    return body, gets, pins


def main():
    pins = {REG: '8cd8ef66904c7dd23640042d9a9776d3ad4acc6299152bf627affea7eb0b477d',
        Path('scripts/acquire_patchseq_append_recordings.py'): 'a4640f505aa37bd46cae43abe90158930665b12a44540be36edb35ae750d8d07',
        Path('scripts/verify_patchseq_NWB_metadata_pilot.py'): 'e1d4d3482106bc56d13114a25e95ac884ff96b600851556e54e27e5a97b69087'}
    if any(digest(p) != value for p, value in pins.items()):
        raise ValueError('sealed acquisition sources changed')
    reg = yaml.safe_load(REG.read_text())
    root = Path(reg['output_directory'])
    manifest = root/'manifest.yaml'
    pins[manifest] = digest(manifest)
    state = yaml.safe_load(manifest.read_text())
    allowed = {'terminal-all-selected-slots-accounted-independent-verification-pending',
        'terminal-storage-preflight-stop', 'terminal-storage-stop', 'terminal-absolute-deadline-stop'}
    if state['status'] not in allowed:
        raise ValueError('acquisition is not terminal; no rerun or network fallback')
    pins.update({Path(p): value for p, value in state['source_hashes'].items()})
    if any(digest(p) != value for p, value in pins.items()):
        raise ValueError('acquisition source changed')
    subset = yaml.safe_load(Path(reg['subset']).read_text())
    same(subset['slots'], state['selected_slots'])
    slots = subset['slots']
    recorded_ids = [r['slot'] for r in state['receipts']]
    if (len(slots) != 48 or any(type(n) is not int or n not in range(1, 49) for n in recorded_ids)
        or len(set(recorded_ids)) != len(recorded_ids)):
        raise ValueError('selected-slot accounting invalid or duplicated')
    unattempted = sorted(set(range(1, 49))-set(recorded_ids))
    if state['status'] == 'terminal-all-selected-slots-accounted-independent-verification-pending' and unattempted:
        raise ValueError('declared complete acquisition lacks selected slots')
    if state['status'] != 'terminal-storage-preflight-stop':
        same(unattempted, state['unattempted_slots'])
    elif recorded_ids:
        raise ValueError('first-transfer storage stop contains later receipts')
    if any((root/f'slot-{n:03d}').exists() for n in unattempted):
        raise ValueError('unattempted slot has unexplained acquisition artifacts')
    same(state['deadline_epoch'], state['started_epoch']+reg['execution']['deadline_seconds_from_first_start'])
    if version('h5py') != reg['logical_metadata_reads']['parser_dependency'].split('==')[1]:
        raise ValueError('registered HDF5 parser differs')
    os.environ['HDF5_PLUGIN_PRELOAD'] = '::'
    import h5py
    reports, charged = [], 0
    for item in state['receipts']:
        slot = slots[item['slot']-1]
        directory = root/f'slot-{slot["slot"]:03d}'
        receipt_path = directory/'receipt.yaml'
        if str(receipt_path) != item['path'] or digest(receipt_path) != item['sha256']:
            raise ValueError('immutable terminal receipt differs')
        pins[receipt_path] = item['sha256']
        receipt = yaml.safe_load(receipt_path.read_text())
        same(receipt['slot'], slot)
        if receipt['physiological_arrays_interpreted'] is not False:
            raise ValueError('physiological access claim exceeds registration')
        for path, value in receipt['retained_file_hashes'].items():
            if Path(path).parent != directory or digest(path) != value:
                raise ValueError('retained slot artifact differs')
            pins[Path(path)] = value
        marker = directory/'GET-attempt.yaml'
        charged += check_admission_and_times(state, slot, directory, receipt)
        source = Path(reg['archive_acquisition']).parent/slot['asset_id']/'response.json'
        if source not in pins:
            raise ValueError('original archive locator not pinned')
        raw = json.loads(source.read_bytes())
        if (raw['path'] != slot['file_path'] or type(raw['contentSize']) is not int or raw['contentSize'] != slot['expected_recording_bytes']
            or raw['digest']['dandi:sha2-256'] != slot['expected_SHA256']):
            raise ValueError('selected author/archive source anchor differs')
        locators = {v for v in raw['contentUrl'] if approved_locator(v, reg['URL_resolution'])}
        if len(locators) != 1:
            raise ValueError('unique pinned locator required')
        body, gets, http_pins = inspect_http(receipt['HTTP_receipts'], slot, reg, locators.pop())
        pins.update(http_pins)
        if gets and not marker.exists():
            raise ValueError('GET occurred without immutable byte admission')
        result = {'slot': slot['slot'], 'status': receipt['status'], 'GET_attempts': gets,
                  'content_SHA256_matches': None, 'identity_accepted': None}
        accepted_statuses = ('content-and-internal-identity-verified-independent-check-pending', 'identity-unresolved-or-conflict-retained')
        if receipt['status'] in accepted_statuses:
            if body is None or not marker.exists():
                raise ValueError('identity inspection lacks successful admitted complete GET')
            complete = digest(body)
            same(complete, slot['expected_SHA256'])
            same(complete, receipt['complete_file_SHA256'])
            same(str(body), receipt['complete_body_path'])
            inspection = replay(body, reg['logical_metadata_reads'], h5py)
            same(inspection, receipt['inspection'])
            flags, accepted = identity_flags(inspection, slot)
            same(flags, receipt['identity_matches'])
            if accepted != (receipt['status'] == accepted_statuses[0]):
                raise ValueError('identity outcome not reproduced')
            result.update(content_SHA256_matches=True, identity_accepted=accepted)
        elif receipt['status'] not in ('acquisition-or-identity-failure-retained', 'interrupted-attempt-retained-no-retry'):
            raise ValueError('undeclared terminal slot status')
        elif body is not None and receipt['status'] == 'acquisition-or-identity-failure-retained':
            complete = digest(body)
            if 'complete_file_SHA256' in receipt:
                same(complete, receipt['complete_file_SHA256'])
                same(str(body), receipt['complete_body_path'])
                result['content_SHA256_matches'] = complete == slot['expected_SHA256']
            if complete == slot['expected_SHA256'] and 'inspection' in receipt:
                # Retain failure, even if its metadata happened to parse before a
                # later deadline error. Unsupported metadata remains unresolved.
                try:
                    inspection = replay(body, reg['logical_metadata_reads'], h5py)
                except (ValueError, TypeError, OSError):
                    result['metadata_independently_rejected'] = True
                else:
                    same(inspection, receipt['inspection'])
                    flags, accepted = identity_flags(inspection, slot)
                    if 'identity_matches' in receipt:
                        same(flags, receipt['identity_matches'])
                    result['metadata_inspection_reproduced_without_promoting_failed_slot'] = True
        reports.append(result)
    same(charged, state.get('charged_recording_bytes', 0))
    if charged > reg['transport']['maximum_charged_recording_bytes'] or any(digest(p) != value for p, value in pins.items()):
        raise ValueError('byte cap or source integrity differs')
    reports.extend({'slot': n, 'status': 'unattempted-under-retained-terminal-stop', 'GET_attempts': 0,
                    'content_SHA256_matches': None, 'identity_accepted': None} for n in unattempted)
    reports.sort(key=lambda r: r['slot'])
    OUTPUT.parent.mkdir()
    with OUTPUT.open('x') as handle:
        yaml.safe_dump({'status': 'all-selected-acquisition-accounting-and-identity-outcomes-independently-verified',
            'verifier_sha256': digest(__file__), 'source_hashes': {str(p): value for p, value in pins.items()},
            'records': reports, 'summary': {'selected_slots': len(reports), 'charged_recording_bytes': charged,
                'attempted_slots': len(recorded_ids), 'unattempted_slots': len(unattempted),
                'content_and_identity_passes': sum(r['identity_accepted'] is True for r in reports),
                'retained_nonpasses_or_unknowns': sum(r['identity_accepted'] is not True for r in reports)},
            'verified_at_utc': datetime.now(UTC).isoformat(), 'acquisition_collector_imported': False,
            'network_requests': 0, 'physiological_arrays_interpreted': False, 'specimen_eligibility_established': False}, handle, sort_keys=False)
    print('All selected terminal outcomes independently verified without trace interpretation')


if __name__ == '__main__':
    main()
