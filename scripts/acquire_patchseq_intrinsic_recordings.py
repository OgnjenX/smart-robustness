"""Checkpointed single-attempt binary acquisition and whitelisted identity only."""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import subprocess
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import yaml
from patchseq_bounded_NWB import inspect_metadata, select_url
from run_patchseq_NWB_metadata_pilot import Transport
from verify_patchseq_whole_file_integrity import digest, same

REG = Path('docs/validation-results/post2008-patchseq-intrinsic-acquisition-registration-1313.yaml')
REG_HASH = 'bfb669450fbb3354dcf9c2b00688e5fe180ccdfc1160b077ae765af8ee122eb4'


def save(path, value, exclusive=False):
    if exclusive:
        with path.open('x') as handle:
            yaml.safe_dump(value, handle, sort_keys=False)
            handle.flush()
            os.fsync(handle.fileno())
    else:
        temporary = path.with_suffix(path.suffix+'.tmp')
        with temporary.open('w') as handle:
            yaml.safe_dump(value, handle, sort_keys=False)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)


@contextmanager
def lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root/'runner.lock').open('a+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def identity(inspection, slot):
    subject = slot['file_path'].split('/')[0].removeprefix('sub-')
    session = slot['file_path'].split('_ses-')[1].removesuffix('_icephys.nwb')
    flags, present_sessions = {}, []
    for path, expected in [('/general/session_id', session), ('/session_id', session), ('/general/subject/subject_id', subject)]:
        entry = inspection['datasets'][path]
        scalar = entry.get('scalar', {})
        text = scalar.get('decoded_text') if scalar.get('raw_type') == 'bytes' else scalar.get('value') if scalar.get('raw_type') == 'str' else None
        flags[path] = text == expected if text is not None else None
        if path.endswith('session_id') and entry['present']:
            present_sessions.append(path)
    accepted = bool(present_sessions) and all(flags[p] is True for p in present_sessions) and flags['/general/subject/subject_id'] is True
    return flags, accepted


def charge(directory, slot, all_directories, cap):
    marker = directory/'GET-attempt.yaml'
    used = sum(yaml.safe_load((d/'GET-attempt.yaml').read_text())['charged_bytes'] for d in all_directories if (d/'GET-attempt.yaml').exists())
    if marker.exists():
        raise ValueError('GET attempt already admitted; no retry')
    if used+slot['expected_recording_bytes'] > cap:
        raise ValueError('registered charged-byte budget exhausted')
    save(marker, {'slot': slot['slot'], 'charged_bytes': slot['expected_recording_bytes'],
                  'admitted_at_utc': datetime.now(UTC).isoformat()}, exclusive=True)


def retained_files(directory):
    return {str(p): digest(p) for p in sorted(directory.iterdir()) if p.is_file() and
            (p.name.endswith('-headers.txt') or p.name.endswith('-body.bin') or p.name in ('attempt.yaml', 'GET-attempt.yaml'))}


def validate_receipt(path, slot):
    receipt = yaml.safe_load(path.read_text())
    same(receipt['slot'], slot)
    if any(digest(p) != value for p, value in receipt['retained_file_hashes'].items()):
        raise ValueError('saved attempt/receipt evidence changed')
    return receipt


def run(root, reg, slots, pins, worker, now=time.time, available=lambda p: shutil.disk_usage(p).free):
    with lock(root):
        if any(digest(p) != value for p, value in pins.items()):
            raise ValueError('sealed source changed')
        manifest = root/'manifest.yaml'
        if manifest.exists():
            state = yaml.safe_load(manifest.read_text())
            same(state['source_hashes'], {str(p): value for p, value in pins.items()})
            same(state['selected_slots'], slots)
            for item in state['receipts']:
                if digest(item['path']) != item['sha256']:
                    raise ValueError('saved terminal receipt changed')
                validate_receipt(Path(item['path']), slots[item['slot']-1])
            if state['status'].startswith('terminal-'):
                if state['status'] == 'terminal-all-selected-slots-accounted-independent-verification-pending' and {r['slot'] for r in state['receipts']} != {s['slot'] for s in slots}:
                    raise ValueError('terminal completion accounting incomplete')
                return state
        else:
            started = now()
            state = {'status': 'acquisition-in-progress', 'started_epoch': started,
                'deadline_epoch': started+reg['execution']['deadline_seconds_from_first_start'],
                'source_hashes': {str(p): value for p, value in pins.items()}, 'selected_slots': slots,
                'receipts': [], 'physiological_arrays_interpreted': False, 'specimen_eligibility_established': False}
            save(manifest, state)
            needed = 2*reg['transport']['expected_sealed_total_recording_bytes']+536870912
            if available(root) < needed:
                state.update(status='terminal-storage-preflight-stop', available_bytes=available(root), required_bytes=needed)
                save(manifest, state)
                return state
        directories = [root/f'slot-{s["slot"]:03d}' for s in slots]
        known = set(directories)
        if any(p not in known for p in root.glob('slot-*')):
            raise ValueError('unexpected slot evidence directory')
        for slot, directory in zip(slots, directories, strict=True):
            marker = directory/'GET-attempt.yaml'
            if marker.exists():
                admitted = yaml.safe_load(marker.read_text())
                same(admitted['slot'], slot['slot'])
                same(admitted['charged_bytes'], slot['expected_recording_bytes'])
        for slot, directory in zip(slots, directories, strict=True):
            receipt_path = directory/'receipt.yaml'
            if receipt_path.exists():
                validate_receipt(receipt_path, slot)
            elif directory.exists():
                if (directory/'attempt.yaml').exists():
                    attempt = yaml.safe_load((directory/'attempt.yaml').read_text())
                    same(attempt['slot'], slot)
                save(receipt_path, {'slot': slot, 'status': 'interrupted-attempt-retained-no-retry',
                    'admission_marker_missing': not (directory/'attempt.yaml').exists(),
                    'retained_file_hashes': retained_files(directory), 'HTTP_receipts': [],
                    'physiological_arrays_interpreted': False}, exclusive=True)
            else:
                if now() < state['started_epoch'] or now() >= state['deadline_epoch']:
                    state['status'] = 'terminal-absolute-deadline-stop'
                    break
                if available(root) < slot['expected_recording_bytes']+536870912:
                    state['status'] = 'terminal-storage-stop'
                    break
                if any(digest(p) != value for p, value in pins.items()):
                    raise ValueError('source changed before slot admission')
                directory.mkdir()
                save(directory/'attempt.yaml', {'slot': slot, 'started_at_utc': datetime.now(UTC).isoformat()}, exclusive=True)
                result = worker(directory, slot, state['deadline_epoch']-now(),
                                lambda d=directory, s=slot: charge(d, s, directories, reg['transport']['maximum_charged_recording_bytes']))
                result.update(slot=slot, retained_file_hashes=retained_files(directory))
                save(receipt_path, result, exclusive=True)
            state['receipts'] = [r for r in state['receipts'] if r['slot'] != slot['slot']]
            state['receipts'].append({'slot': slot['slot'], 'path': str(receipt_path), 'sha256': digest(receipt_path)})
            state['receipts'].sort(key=lambda r: r['slot'])
            save(manifest, state)
        else:
            state['status'] = 'terminal-all-selected-slots-accounted-independent-verification-pending'
        state.update(finished_at_utc=datetime.now(UTC).isoformat(),
            charged_recording_bytes=sum(yaml.safe_load((d/'GET-attempt.yaml').read_text())['charged_bytes'] for d in directories if (d/'GET-attempt.yaml').exists()),
            unattempted_slots=[s['slot'] for s, d in zip(slots, directories, strict=True) if not d.exists()])
        if any(digest(p) != value for p, value in pins.items()):
            raise ValueError('source changed during acquisition')
        save(manifest, state)
        return state


def acquire(directory, slot, remaining, charge_get, reg, locator):
    config = {'URL_resolution': {**reg['URL_resolution'], 'maximum_HEAD_requests': reg['transport']['maximum_HEAD_requests_per_slot'],
        'maximum_HEAD_redirects': reg['transport']['maximum_HEAD_redirects_per_slot']},
        'HTTP_range_limits': {'maximum_run_seconds': remaining, 'timeout_seconds_per_request': reg['transport']['timeout_seconds_per_request']}}
    transport = Transport(directory, locator, {'size': slot['expected_recording_bytes']}, config)
    result = {'status': 'acquisition-or-identity-failure-retained', 'HTTP_receipts': [],
              'physiological_arrays_interpreted': False, 'specimen_eligibility_established': False}
    try:
        transport.head()
        transport.check_deadline()
        charge_get()
        _, _, body = transport.request('RANGE', 0, slot['expected_recording_bytes']-1)
        value = digest(body)
        result.update(complete_body_path=str(body), complete_file_SHA256=value)
        if value != slot['expected_SHA256']:
            raise ValueError('complete recording author SHA256 mismatch')
        os.environ['HDF5_PLUGIN_PRELOAD'] = '::'
        import h5py
        inspection = {'root_attributes': {}, 'datasets': {}, 'logical_dataset_reads': []}
        result['inspection'] = inspection
        with h5py.File(body, 'r') as file:
            inspect_metadata(file, reg['logical_metadata_reads'], h5py, inspection)
        flags, accepted = identity(inspection, slot)
        if digest(body) != value:
            raise ValueError('complete binary changed during identity inspection')
        result.update(identity_matches=flags, status='content-and-internal-identity-verified-independent-check-pending'
                      if accepted else 'identity-unresolved-or-conflict-retained')
        transport.check_deadline()
    except (OSError, ValueError, TypeError, KeyError, TimeoutError, subprocess.TimeoutExpired) as exc:
        result.update(status='acquisition-or-identity-failure-retained', error={'type': type(exc).__name__, 'message': str(exc)})
    finally:
        result.update(HTTP_receipts=transport.records, finished_at_utc=datetime.now(UTC).isoformat())
    return result


def main():
    if digest(REG) != REG_HASH:
        raise ValueError('sealed acquisition registration changed')
    reg = yaml.safe_load(REG.read_text())
    pins = {REG: REG_HASH, Path(__file__): digest(__file__),
        Path('scripts/run_patchseq_NWB_metadata_pilot.py'): 'c50fc863b901db0c179314c90f9730e1fba0aee9f759788da002d85817d5fa66',
        Path('scripts/patchseq_bounded_NWB.py'): '722128c416bf640247e5eb9aac1284c16d570e3785452ed4037c1a7392019206',
        Path('scripts/verify_patchseq_whole_file_integrity.py'): '22e40726389126ff7a875794138145d222ed5b87515e87a2ec9bccadce78c057'}
    keys = ('subset', 'independent_subset', 'archive_acquisition', 'independent_archive')
    pins.update({Path(reg[k]): reg[k+'_sha256'] for k in keys})
    if any(digest(p) != value for p, value in pins.items()):
        raise ValueError('pinned prerequisite changed')
    data = {k: yaml.load(Path(reg[k]).read_text(), Loader=yaml.CSafeLoader) for k in keys}
    if (data['independent_subset']['status'] != 'finite-intrinsic-subset-and-author-preflight-independently-reconstructed'
        or data['independent_archive']['status'] != 'all-terminal-archive-metadata-receipts-independently-verified'
        or data['independent_archive']['terminal_manifest_sha256'] != reg['archive_acquisition_sha256']):
        raise ValueError('independent prerequisites missing')
    slots = data['subset']['slots']
    if ([s['slot'] for s in slots] != list(range(1, 49)) or sum(s['expected_recording_bytes'] for s in slots) != reg['transport']['expected_sealed_total_recording_bytes']
        or len({s['asset_id'] for s in slots}) != 48 or any(s['preflight_failures'] for s in slots)
        or any(s['expected_recording_bytes'] > reg['transport']['maximum_single_recording_bytes'] for s in slots)):
        raise ValueError('sealed slot scope or size preflight differs')
    locators = {}
    for slot in slots:
        matches = [r for r in data['archive_acquisition']['records'] if r['asset_id'] == slot['asset_id']]
        checks = [r for r in data['independent_archive']['records'] if r['asset_id'] == slot['asset_id']]
        if len(matches) != 1 or len(checks) != 1 or matches[0]['receipt_sha256'] != checks[0]['receipt_sha256']:
            raise ValueError('selected archive receipt not independently matched')
        path = Path(matches[0]['receipt'])
        pins[path] = matches[0]['receipt_sha256']
        if digest(path) != pins[path]:
            raise ValueError('selected archive receipt changed')
        receipt = yaml.safe_load(path.read_text())
        source = path.parent/'response.json'
        pins[source] = receipt['response_sha256']
        if checks[0]['raw_evidence_hashes'].get(str(source)) != pins[source] or digest(source) != pins[source]:
            raise ValueError('selected original archive JSON changed')
        raw = json.loads(source.read_bytes())
        if (raw['path'] != slot['file_path'] or type(raw['contentSize']) is not int
            or raw['contentSize'] != slot['expected_recording_bytes'] or raw['digest']['dandi:sha2-256'] != slot['expected_SHA256']):
            raise ValueError('selected author/archive source mismatch')
        locators[slot['slot']], _ = select_url(raw, reg['URL_resolution'])
    if version('h5py') != reg['logical_metadata_reads']['parser_dependency'].split('==')[1]:
        raise ValueError('registered HDF5 parser differs')
    result = run(Path(reg['output_directory']), reg, slots, pins,
                 lambda directory, slot, remaining, charged: acquire(directory, slot, remaining, charged, reg, locators[slot['slot']]))
    print(result['status'])


if __name__ == '__main__':
    main()
