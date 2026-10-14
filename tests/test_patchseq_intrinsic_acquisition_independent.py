"""Independent receipt validation rejects retries, changed evidence and identity conflicts."""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

scripts = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(scripts))
try:
    spec = importlib.util.spec_from_file_location('independent_acquisition', scripts/'verify_patchseq_intrinsic_acquisition.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
finally:
    sys.path.pop(0)


def fixture(tmp_path):
    def receipt(kind, data, fields, index):
        code = 200 if kind == 'HEAD' else 206
        header, body = tmp_path/f'{index}-headers.txt', tmp_path/f'{index}-body.bin'
        header.write_text(f'HTTP/1.1 {code} OK\n'+''.join(f'{k}: {v[0]}\n' for k, v in fields.items())+'\n')
        body.write_bytes(data)
        return {'kind': kind, 'URL_host': 'dandiarchive.s3.amazonaws.com', 'URL_path': '/blobs/synthetic',
            'status': 'response-retained', 'http_status': code, 'header_fields': fields, 'client_exit_code': 0,
            'start': None if kind == 'HEAD' else 0, 'end': None if kind == 'HEAD' else 7,
            'headers_path': str(header), 'headers_bytes': header.stat().st_size, 'headers_sha256': module.digest(header),
            'body_path': str(body), 'body_bytes': len(data), 'body_sha256': module.digest(body)}
    receipts = [receipt('HEAD', b'', {'etag': ['"stable"'], 'content-length': ['8']}, 0),
        receipt('RANGE', b'abcdefgh', {'etag': ['"stable"'], 'content-range': ['bytes 0-7/8']}, 1)]
    slot = {'expected_recording_bytes': 8}
    reg = {'transport': {'maximum_HEAD_requests_per_slot': 2, 'maximum_HEAD_redirects_per_slot': 1,
                         'maximum_GET_attempts_per_slot': 1}}
    return receipts, slot, reg, 'https://dandiarchive.s3.amazonaws.com/blobs/synthetic'


def test_independent_exact_complete_response_and_attempt_count(tmp_path):
    body, gets, pins = module.inspect_http(*fixture(tmp_path))
    assert body.read_bytes() == b'abcdefgh' and gets == 1 and len(pins) == 4


@pytest.mark.parametrize('case', ['retry', 'changed_body', 'header_status', 'extra_HEAD', 'wrong_range'])
def test_bad_retained_http_evidence_is_rejected(tmp_path, case):
    receipts, slot, reg, locator = fixture(tmp_path)
    if case == 'retry':
        receipts.append(copy.deepcopy(receipts[1]))
    elif case == 'changed_body':
        Path(receipts[1]['body_path']).write_bytes(b'changed!')
    elif case == 'header_status':
        receipts[1]['http_status'] = 200
    elif case == 'extra_HEAD':
        receipts.append(copy.deepcopy(receipts[0]))
    else:
        receipts[1]['end'] = 6
    with pytest.raises(ValueError):
        module.inspect_http(receipts, slot, reg, locator)


def test_failed_partial_response_remains_unaccepted_not_complete(tmp_path):
    receipts, slot, reg, locator = fixture(tmp_path)
    receipts[1]['status'] = 'failed'
    receipts[1]['client_exit_code'] = 28
    body, gets, _ = module.inspect_http(receipts, slot, reg, locator)
    assert body is None and gets == 1


@pytest.mark.parametrize('case', ['match', 'conflict', 'numeric', 'absent'])
def test_independent_identity_requires_every_present_session_to_agree(case):
    def text(value):
        return {'present': True, 'scalar': {'raw_type': 'bytes', 'raw_hex': value.encode().hex(), 'decoded_text': value}}
    data = {'/general/session_id': text('34'), '/session_id': text('34'), '/general/subject/subject_id': text('12')}
    if case == 'conflict':
        data['/session_id'] = text('35')
    elif case == 'numeric':
        data['/session_id'] = {'present': True, 'scalar': {'raw_type': 'int', 'value': 34}}
    elif case == 'absent':
        data['/session_id'] = data['/general/session_id'] = {'present': False}
    _, accepted = module.identity_flags({'datasets': data}, {'file_path': 'sub-12/sub-12_ses-34_icephys.nwb'})
    assert accepted is (case == 'match')


@pytest.mark.parametrize('case', ['valid', 'wrong_bytes', 'late_admission', 'early_GET', 'late_response'])
def test_independent_admission_and_absolute_deadline(tmp_path, case):
    slot = {'slot': 1, 'expected_recording_bytes': 8}
    (tmp_path/'attempt.yaml').write_text(yaml.safe_dump({'slot': slot, 'started_at_utc': '2026-10-14T00:00:01+00:00'}))
    admission = {'slot': 1, 'charged_bytes': 8, 'admitted_at_utc': '2026-10-14T00:00:02+00:00'}
    event = {'kind': 'RANGE', 'status': 'response-retained', 'started_at_utc': '2026-10-14T00:00:03+00:00',
             'finished_at_utc': '2026-10-14T00:00:04+00:00'}
    from datetime import datetime
    state = {'started_epoch': datetime.fromisoformat('2026-10-14T00:00:00+00:00').timestamp(),
             'deadline_epoch': datetime.fromisoformat('2026-10-14T00:00:10+00:00').timestamp()}
    if case == 'wrong_bytes':
        admission['charged_bytes'] = 7
    elif case == 'late_admission':
        admission['admitted_at_utc'] = '2026-10-14T00:00:11+00:00'
    elif case == 'early_GET':
        event['started_at_utc'] = '2026-10-14T00:00:01+00:00'
    elif case == 'late_response':
        event['finished_at_utc'] = '2026-10-14T00:00:11+00:00'
    (tmp_path/'GET-attempt.yaml').write_text(yaml.safe_dump(admission))
    if case == 'valid':
        assert module.check_admission_and_times(state, slot, tmp_path, {'HTTP_receipts': [event]}) == 8
    else:
        with pytest.raises(ValueError):
            module.check_admission_and_times(state, slot, tmp_path, {'HTTP_receipts': [event]})
