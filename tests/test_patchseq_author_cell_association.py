"""External association requires unique content/session/cell reference chains."""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import pytest

scripts = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(scripts))
try:
    spec = importlib.util.spec_from_file_location('author_cell_association', scripts/'associate_patchseq_author_cell.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
finally:
    sys.path.pop(0)


def inputs():
    checksum = 'a'*64
    cells = {key: {'storage_type': None if key in ('file_size', 'cell_specimen_id') else 's',
        'formula_present': False, 'raw_value': value, 'literal_text': value}
        for key, value in [('file_size', '12'), ('cell_specimen_id', '42'),
                           ('file_name', 'sub-1_ses-7_icephys.nwb'), ('sha256_checksum', checksum)]}
    integrity = {'complete_file_SHA256': checksum, 'complete_body_bytes': 12}
    session = {'inspection': {'datasets': {'/general/session_id': {'present': True,
        'scalar': {'raw_type': 'bytes', 'decoded_text': '7'}}}}, 'literal_path_session_match': True}
    refs = {'assets': [{'asset_id': 'synthetic', 'path': 'sub-1/sub-1_ses-7_icephys.nwb',
        'manifest_candidates': [{'manifest_source_row_number': 2, 'July_cell_source_row_numbers': [3],
            'manifest_cell_agrees_with_unique_July_session': True, 'unique_July_cell_and_session_same_row': True}],
        'complete_external_file_reference_agreement': True, 'July_session_source_row_numbers': [3]}]}
    manifest = {'rows': [{'source_row_number': 2, 'cells': cells,
        'canonical_integers': {'cell_specimen_id': 42, 'file_size': 12}}]}
    july = {'July_rows': [{'source_row_number': 3, 'canonical_ids': {'cell_specimen_id': 42, 'ephys_session_id': 7},
        'raw_fields': {'cell_specimen_id': '42', 'ephys_session_id': '7'}}]}
    return ['synthetic', integrity, session, refs, manifest, july]


def test_unique_reference_chain_is_not_eligibility_or_RNA_alias_proof():
    result = module.associate(*inputs())
    assert result['status'].startswith('externally-author-linked')
    assert result['association']['source_cell_specimen_id'] == '42'
    assert all(result[key] is False for key in ('internal_cell_ID_proven', 'RNA_alias_equivalence_established',
        'donor_equivalence_established', 'mapping_confidence_assigned', 'specimen_eligibility_established'))


@pytest.mark.parametrize('case', ['duplicate_asset', 'duplicate_author', 'duplicate_July', 'wrong_hash',
                                'wrong_session', 'raw_whitespace', 'formula', 'float_canonical'])
def test_ambiguous_or_inconsistent_source_keeps_no_association(case):
    values = inputs()
    if case == 'duplicate_asset':
        values[3]['assets'] *= 2
    elif case == 'duplicate_author':
        values[4]['rows'] *= 2
    elif case == 'duplicate_July':
        values[5]['July_rows'].append(copy.deepcopy(values[5]['July_rows'][0]))
    elif case == 'wrong_hash':
        values[1]['complete_file_SHA256'] = 'b'*64
    elif case == 'wrong_session':
        values[2]['inspection']['datasets']['/general/session_id']['scalar']['decoded_text'] = '8'
    elif case == 'raw_whitespace':
        values[5]['July_rows'][0]['raw_fields']['cell_specimen_id'] = ' 42'
    elif case == 'formula':
        values[4]['rows'][0]['cells']['cell_specimen_id']['formula_present'] = True
    else:
        values[5]['July_rows'][0]['canonical_ids']['cell_specimen_id'] = 42.0
    result = module.associate(*values)
    assert result['status'] == 'unresolved-no-association'
    assert result['failures'] and 'association' not in result
    assert result['specimen_eligibility_established'] is False


@pytest.mark.parametrize('raw', ['042', '4.2e1', '42.0', ' 42', '0', 'True'])
def test_numeric_lexeme_cannot_be_cleaned_or_rounded(raw):
    cell = {'formula_present': False, 'storage_type': None, 'raw_value': raw, 'literal_text': None}
    assert module.canonical_typed(cell) is None
