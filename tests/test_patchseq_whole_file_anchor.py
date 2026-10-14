"""Complete binary acquisition requires literal author/archive anchor agreement."""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import pytest

scripts = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(scripts))
try:
    spec = importlib.util.spec_from_file_location('whole_file_anchor', scripts/'acquire_patchseq_whole_file_integrity.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
finally:
    sys.path.pop(0)


def inputs():
    reg = {'asset': {'asset_id': 'synthetic', 'path': 'subject/file.nwb', 'size': 12},
           'author_source_row_number': 2, 'expected_SHA256': 'a'*64}
    cells = {k: {'formula_present': False, 'raw_value': value, 'literal_text': value}
             for k, value in [('file_size', '12'), ('file_name', 'file.nwb'), ('sha256_checksum', 'a'*64)]}
    manifest = {'rows': [{'source_row_number': 2, 'cells': cells}]}
    refs = {'assets': [{'asset_id': 'synthetic', 'path': 'subject/file.nwb', 'complete_external_file_reference_agreement': True,
        'manifest_candidates': [{'manifest_source_row_number': 2}]}]}
    raw = {'contentSize': 12, 'digest': {'dandi:sha2-256': 'a'*64}}
    return reg, manifest, refs, raw


def test_literal_author_anchor_passes():
    module.check_author_anchor(*inputs())


@pytest.mark.parametrize('case', ['formula', 'size', 'filename', 'digest', 'duplicate'])
def test_wrong_or_ambiguous_anchor_stops_before_acquisition(case):
    reg, manifest, refs, raw = inputs()
    if case == 'formula':
        manifest['rows'][0]['cells']['file_size']['formula_present'] = True
    elif case == 'size':
        manifest['rows'][0]['cells']['file_size']['raw_value'] = '012'
    elif case == 'filename':
        manifest['rows'][0]['cells']['file_name']['literal_text'] = 'different.nwb'
    elif case == 'digest':
        raw['digest']['dandi:sha2-256'] = 'b'*64
    else:
        refs['assets'].append(copy.deepcopy(refs['assets'][0]))
    with pytest.raises(ValueError):
        module.check_author_anchor(reg, manifest, refs, raw)
