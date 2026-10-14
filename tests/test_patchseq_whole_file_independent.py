"""Independent complete binary verification rejects damaged or stale evidence."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

scripts = Path(__file__).resolve().parents[1]/'scripts'
spec = importlib.util.spec_from_file_location('whole_file_independent', scripts/'verify_patchseq_whole_file_integrity.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(tmp_path):
    def receipt(kind, fields, data, index):
        status = 200 if kind == 'HEAD' else 206
        headers, body = tmp_path/f'{index}-headers.txt', tmp_path/f'{index}-body.bin'
        headers.write_text(f'HTTP/1.1 {status} OK\n'+''.join(f'{k}: {v[0]}\n' for k, v in fields.items())+'\n')
        body.write_bytes(data)
        return {'kind': kind, 'URL_host': 'dandiarchive.s3.amazonaws.com', 'URL_path': '/blobs/synthetic',
            'start': None if kind == 'HEAD' else 0, 'end': None if kind == 'HEAD' else 7,
            'http_status': status, 'header_fields': fields, 'client_exit_code': 0, 'status': 'response-retained',
            'headers_path': str(headers), 'headers_bytes': headers.stat().st_size, 'headers_sha256': module.digest(headers),
            'body_path': str(body), 'body_bytes': len(data), 'body_sha256': module.digest(body)}
    report = {'HTTP_receipts': [receipt('HEAD', {'etag': ['"stable"'], 'content-length': ['8']}, b'', 0),
        receipt('RANGE', {'etag': ['"stable"'], 'content-range': ['bytes 0-7/8']}, b'abcdefgh', 1)],
        'started_at_utc': '2026-10-14T00:00:00+00:00', 'finished_at_utc': '2026-10-14T00:00:01+00:00'}
    reg = {'asset': {'size': 8}, 'URL_resolution': {'maximum_HEAD_requests': 2, 'maximum_HEAD_redirects': 1},
        'HTTP_range_limits': {'maximum_unique_downloaded_bytes': 8, 'maximum_run_seconds': 900}}
    return report, reg


def test_independent_headers_and_exact_full_range(tmp_path):
    report, reg = fixture(tmp_path)
    body, pins = module.verify_transport(report, reg, 'https://dandiarchive.s3.amazonaws.com/blobs/synthetic', '"stable"')
    assert body.read_bytes() == b'abcdefgh' and len(pins) == 4


@pytest.mark.parametrize('case', ['body', 'status', 'duplicate', 'etag', 'deadline'])
def test_bad_transport_evidence_rejected(tmp_path, case):
    report, reg = fixture(tmp_path)
    if case == 'body':
        Path(report['HTTP_receipts'][1]['body_path']).write_bytes(b'damaged!')
    elif case == 'status':
        report['HTTP_receipts'][1]['http_status'] = 200
    elif case == 'duplicate':
        report['HTTP_receipts'].append(copy.deepcopy(report['HTTP_receipts'][1]))
    elif case == 'etag':
        report['HTTP_receipts'][1]['header_fields']['etag'] = ['"changed"']
    else:
        report['finished_at_utc'] = '2026-10-14T00:16:00+00:00'
    with pytest.raises(ValueError):
        module.verify_transport(report, reg, 'https://dandiarchive.s3.amazonaws.com/blobs/synthetic', '"stable"')


def test_cached_blocks_require_exact_complete_file_offsets(tmp_path):
    body, block = tmp_path/'complete.bin', tmp_path/'cache.bin'
    body.write_bytes(b'abcdefgh')
    block.write_bytes(b'cdef')
    report = {'HTTP_receipts': [{'kind': 'RANGE', 'body_path': str(block), 'start': 2, 'end': 5}]}
    pins = {block: module.digest(block)}
    assert module.compare_cached_blocks(body, [report], pins) == 1
    body.write_bytes(b'abXXXXgh')
    with pytest.raises(ValueError, match='offsets'):
        module.compare_cached_blocks(body, [report], pins)
