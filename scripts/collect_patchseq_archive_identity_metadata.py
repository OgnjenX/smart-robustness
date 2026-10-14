"""Bounded, no-retry archive metadata acquisition with immutable checkpoints."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import subprocess
import uuid
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import yaml

REG = Path('docs/validation-results/post2008-patchseq-archive-identity-metadata-registration-1238.yaml')
REG_SHA = '5f94fe6a58ad551be3ea09251750bc713e77de7dec4c64ff910b486873d0ca3f'


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def save(path, data):
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('x') as handle:
        yaml.safe_dump(data, handle, sort_keys=False)
    temporary.replace(path)


def load_json(raw, limits):
    if len(raw) > limits['maximum_response_bytes']:
        raise ValueError('response byte bound exceeded')

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result

    def constant(value):
        raise ValueError('nonfinite JSON value: '+value)

    parsed = json.loads(raw.decode('utf-8'), object_pairs_hook=pairs, parse_constant=constant)
    stack, count = [(parsed, 0)], 0
    while stack:
        value, depth = stack.pop()
        count += 1
        if depth > limits['maximum_json_depth'] or count > limits['maximum_json_nodes']:
            raise ValueError('JSON shape bound exceeded')
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError('nonfinite JSON exponent')
        if isinstance(value, dict):
            stack.extend((v, depth+1) for v in value.values())
        elif isinstance(value, list):
            stack.extend((v, depth+1) for v in value)
    if not isinstance(parsed, dict):
        raise TypeError('metadata object required')
    return parsed


def project(source, reg):
    limits, allowed = reg['limits'], reg['permitted_projection']

    def snapshot(value):
        kind = type(value).__name__
        if isinstance(value, str) and len(value) > limits['maximum_projected_string_characters']:
            raise ValueError('projected string bound exceeded')
        if isinstance(value, (dict, list)):
            if len(value) > limits['maximum_projected_list_entries']:
                raise ValueError('projected container bound exceeded')
            return {'type': kind, 'keys' if isinstance(value, dict) else 'length': sorted(value) if isinstance(value, dict) else len(value)}
        return {'type': kind, 'value': value}

    def fields(obj, names):
        result = {}
        for name in names:
            record = {'present': name in obj}
            if name in obj:
                value = obj[name]
                record.update(snapshot(value))
                if name == 'digest' and isinstance(value, dict):
                    record['algorithm_values'] = {k: snapshot(v) for k, v in value.items()}
            result[name] = record
        return result

    def actors(obj, container, names):
        if container not in obj:
            return {'present': False}
        value = obj[container]
        record = {'present': True, **snapshot(value)}
        if isinstance(value, list):
            record['entries'] = [fields(v, names) if isinstance(v, dict) else {'unexpected_entry': snapshot(v)} for v in value]
        return record

    def branch(obj, names):
        return {'fields': fields(obj, names),
                'participants': actors(obj, allowed['participant_container'], allowed['participant_fields']),
                'activities': actors(obj, allowed['activity_container'], allowed['activity_fields'])}

    result = {'top_level': branch(source, allowed['top_level_fields']),
              'metadata_container': {'present': 'metadata' in source}}
    if 'metadata' in source:
        value = source['metadata']
        result['metadata_container'].update(type=type(value).__name__)
        if isinstance(value, dict):
            result['metadata_container']['projection'] = branch(value, allowed['metadata_container_fields'])
    return result


def endpoint(asset, reg):
    identity = asset['asset_id']
    if str(uuid.UUID(identity)) != identity:
        raise ValueError('canonical asset UUID required')
    return reg['endpoint_template'].format(asset_id=identity)


def headers(path):
    if path.stat().st_size > 65536:
        raise ValueError('HTTP headers exceed bound')
    sections = path.read_text().replace('\r\n', '\n').strip().split('\n\n')
    final = sections[-1].splitlines()
    if not final or final[0].split()[1:2] != ['200']:
        raise ValueError('HTTP 200 response required; redirects rejected')
    result = {}
    for line in final[1:]:
        key, value = line.split(':', 1)
        if key.lower() in ('content-type', 'content-length', 'content-encoding', 'etag', 'last-modified', 'date'):
            result.setdefault(key.lower(), []).append(value.strip())
    if result.get('content-encoding', ['identity']) != ['identity']:
        raise ValueError('identity content encoding required')
    if not any(v.split(';')[0].strip() == 'application/json' for v in result.get('content-type', [])):
        raise ValueError('JSON content type required')
    return result


def inspect_checkpoint(directory, asset, reg, collector_hash):
    receipt = yaml.load((directory/'receipt.yaml').read_text(), Loader=yaml.CSafeLoader)
    if (receipt['asset'] != asset or receipt['registration_sha256'] != REG_SHA
        or receipt['collector_sha256'] != collector_hash or receipt['endpoint'] != endpoint(asset, reg)):
        raise ValueError('checkpoint source mismatch')
    for name, hashkey in [('response.json', 'response_sha256'), ('headers.txt', 'headers_sha256')]:
        expected = receipt.get(hashkey)
        path = directory/name
        if expected is not None and (not path.exists() or sha(path) != expected):
            raise ValueError('checkpoint raw evidence changed')
        if expected is None and path.exists():
            raise ValueError('unreceipted checkpoint bytes')
    return receipt


def acquire(directory, asset, reg, collector_hash):
    directory.mkdir()
    url = endpoint(asset, reg)
    initial = {'asset': asset, 'endpoint': url, 'registration_sha256': REG_SHA,
               'collector_sha256': collector_hash, 'started_at_utc': datetime.now(UTC).isoformat()}
    save(directory/'attempt.yaml', initial)
    receipt = {**initial, 'status': 'acquisition-failure-retained', 'physiological_arrays_read': False}
    body, header = directory/'response.json', directory/'headers.txt'
    try:
        completed = subprocess.run(['curl', '--silent', '--show-error', '--proto', '=https',
            '--max-time', str(reg['limits']['timeout_seconds_per_request']),
            '--max-filesize', str(reg['limits']['maximum_response_bytes']), '--header', 'Accept: application/json',
            '--header', 'Accept-Encoding: identity', '--dump-header', str(header), '--output', str(body),
            '--write-out', '%{http_code}', url], capture_output=True, text=True,
            timeout=reg['limits']['timeout_seconds_per_request']+5, check=False)
        receipt.update(client_exit_code=completed.returncode, http_status=completed.stdout, client_error=completed.stderr[:4096])
        if completed.returncode != 0 or completed.stdout != '200':
            raise ValueError('single request failed; no retry authorized')
        receipt['headers'] = headers(header)
        if body.stat().st_size > reg['limits']['maximum_response_bytes']:
            raise ValueError('response byte bound exceeded')
        parsed = load_json(body.read_bytes(), reg['limits'])
        receipt['projection'] = project(parsed, reg)
        receipt['status'] = 'archive-metadata-projected-independent-verification-pending'
    except (OSError, ValueError, TypeError, KeyError, RecursionError, subprocess.TimeoutExpired) as exc:
        receipt.update(error_type=type(exc).__name__, error=str(exc)[:4096])
    for path, key in [(body, 'response_sha256'), (header, 'headers_sha256')]:
        if path.exists():
            receipt[key] = sha(path)
    receipt['finished_at_utc'] = datetime.now(UTC).isoformat()
    save(directory/'receipt.yaml', receipt)
    return receipt


def main():
    if sha(REG) != REG_SHA:
        raise ValueError('sealed metadata registration changed')
    reg = yaml.load(REG.read_text(), Loader=yaml.CSafeLoader)
    for path, expected in [(reg['catalog'], reg['catalog_sha256']),
                           (reg['prerequisite']['result'], reg['prerequisite']['result_sha256'])]:
        if sha(path) != expected:
            raise ValueError('metadata prerequisite changed')
    catalog = yaml.load(Path(reg['catalog']).read_text(), Loader=yaml.CSafeLoader)['assets']
    assets = sorted(catalog, key=lambda a: (a['path'], a['asset_id']))
    if len(assets) != reg['selection']['expected_assets'] or len({a['asset_id'] for a in assets}) != len(assets):
        raise ValueError('full catalog cohort required')
    root = Path(reg['output_directory'])
    root.mkdir(exist_ok=True)
    if (root/'manifest.yaml').exists():
        raise ValueError('terminal acquisition already exists; do not rerun')
    collector_hash = sha(__file__)
    with (root/'collector.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        records = []
        for asset in assets:
            directory = root/asset['asset_id']
            if (directory/'receipt.yaml').exists():
                receipt = inspect_checkpoint(directory, asset, reg, collector_hash)
            elif directory.exists():
                raise ValueError('interrupted attempt requires explicit assessment; no automatic retry')
            else:
                receipt = acquire(directory, asset, reg, collector_hash)
            records.append({'asset_id': asset['asset_id'], 'receipt': str(directory/'receipt.yaml'),
                            'receipt_sha256': sha(directory/'receipt.yaml'), 'status': receipt['status']})
            if len(records) % 100 == 0:
                print(f'{len(records)}/{len(assets)} metadata attempts receipted', flush=True)
        if sha(REG) != REG_SHA or sha(__file__) != collector_hash or sha(reg['catalog']) != reg['catalog_sha256']:
            raise ValueError('source changed during metadata acquisition')
        save(root/'manifest.yaml', {'status': 'full-archive-metadata-acquisition-terminal-independent-verification-pending',
             'registration_sha256': REG_SHA, 'collector_sha256': collector_hash, 'records': records,
             'status_counts': dict(Counter(r['status'] for r in records)), 'network_requests_maximum': len(assets),
             'physiological_arrays_read': False, 'specimen_qualification_performed': False,
             'finished_at_utc': datetime.now(UTC).isoformat()})
        print('Full metadata acquisition terminal; failures remain visible', flush=True)


if __name__ == '__main__':
    main()
