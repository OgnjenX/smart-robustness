"""Seeded range acquisition counts only new requests, without overwriting cache."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

scripts = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(scripts))
try:
    spec = importlib.util.spec_from_file_location('structure_ranges', scripts/'run_patchseq_NWB_structure_ranges.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
finally:
    sys.path.pop(0)


def fixture(tmp_path, data=b'abcd', start=0):
    body = tmp_path/'block.bin'
    body.write_bytes(data)
    return {'HTTP_receipts': [{'kind': 'RANGE', 'start': start, 'end': start+len(data)-1,
        'body_path': str(body), 'body_sha256': hashlib.sha256(data).hexdigest()}]}


def reader(fetch):
    return module.BoundedReader(12, {'cache_block_bytes': 4,
        'maximum_unique_downloaded_bytes': 4, 'maximum_range_requests': 1}, fetch)


def test_seed_cache_skips_network_and_new_blocks_charge_before_fetch(tmp_path):
    calls = []
    def fetch(start, end):
        calls.append((start, end))
        return b'efgh'
    stream = reader(fetch)
    module.seed(stream, fixture(tmp_path), {'blocks': 1, 'bytes': 4})
    assert stream.read(4) == b'abcd'
    assert calls == [] and stream.downloaded == 0 and stream.requests == 0
    assert stream.read(4) == b'efgh'
    assert calls == [(4, 7)] and stream.downloaded == 4 and stream.requests == 1
    with pytest.raises(ValueError, match='budget exhausted'):
        stream.read(1)
    assert calls == [(4, 7)]


@pytest.mark.parametrize('case', ['hash', 'alignment', 'duplicate', 'inventory'])
def test_bad_seed_is_rejected_before_any_network(tmp_path, case):
    pilot = fixture(tmp_path, start=1 if case == 'alignment' else 0)
    if case == 'hash':
        pilot['HTTP_receipts'][0]['body_sha256'] = '0'*64
    if case == 'duplicate':
        pilot['HTTP_receipts'] *= 2
    stream = reader(lambda *args: pytest.fail('network called while seeding'))
    with pytest.raises(ValueError):
        module.seed(stream, pilot, {'blocks': 2 if case == 'inventory' else 1, 'bytes': 4})
    assert stream.cache == {} and stream.requests == 0
