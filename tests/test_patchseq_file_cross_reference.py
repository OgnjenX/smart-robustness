"""Source file cross-references never confer physiological eligibility."""

from __future__ import annotations

import copy
import csv
import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / 'scripts/cross_reference_patchseq_files.py'
spec = importlib.util.spec_from_file_location('file_cross_reference', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    raw = dict.fromkeys(module.FIELDS, '')
    raw.update(file_name='123.nwb', cell_specimen_id='456', technique='intracellular_electrophysiology')
    rows = [{'source_row_number': 2, 'raw_fields': raw, 'canonical_cell_specimen_id': 456,
             'recognized_file_session': 123}]
    assets = [{'asset_id': 'a', 'path': 'sub-9/sub-9_ses-123_icephys.nwb', 'path_session_token': 123,
               'annotation_candidates': [{'canonical_ids': {'cell_specimen_id': 456}}]}]
    return rows, assets


def test_exact_cell_reference_does_not_promote_identity_or_eligibility():
    result = module.cross_reference(*fixture())
    assert result['summary']['agreeing_cell_references'] == 1
    assert not result['assets'][0]['internal_recording_identity_confirmed']
    assert not result['assets'][0]['biological_eligibility_established']
    assert result['source_rows'][0]['raw_fields']['archive_uri'] == ''


@pytest.mark.parametrize('name', ['001.nwb', ' 123.nwb', '123.NWB', 'folder/123.nwb',
                                'abc123.nwb', '١.nwb', 'sub-9_ses-001_icephys.nwb'])
def test_filename_not_repaired(name):
    assert module.file_session(name) is None


@pytest.mark.parametrize('name', ['123.nwb', 'sub-9_ses-123_icephys.nwb'])
def test_registered_filename_shapes(name):
    assert module.file_session(name) == 123


@pytest.mark.parametrize('fault', ['duplicate', 'different-cell', 'other-modality', 'unknown-file', 'annotation-ambiguity'])
def test_conflicts_not_silently_resolved(fault):
    rows, assets = fixture()
    if fault == 'duplicate':
        other = copy.deepcopy(rows[0])
        other['source_row_number'] = 3
        rows.append(other)
    elif fault == 'different-cell':
        rows[0]['canonical_cell_specimen_id'] = 457
        rows[0]['raw_fields']['cell_specimen_id'] = '457'
    elif fault == 'other-modality':
        rows[0]['raw_fields']['technique'] = 'morphology'
    elif fault == 'unknown-file':
        rows[0]['recognized_file_session'] = None
        rows[0]['raw_fields']['file_name'] = 'unknown.nwb'
    else:
        assets[0]['annotation_candidates'].append(copy.deepcopy(assets[0]['annotation_candidates'][0]))
    result = module.cross_reference(rows, assets)
    assert not result['assets'][0]['source_cell_ID_agrees']
    assert result['source_rows'] == rows
    if fault == 'duplicate':
        assert result['assets'][0]['manifest_source_row_numbers'] == [2, 3]
        assert result['duplicate_source_file_sessions'] == [123]
    if fault in ('other-modality', 'unknown-file'):
        assert result['unreferenced_source_row_numbers'] == [2]


def test_raw_fields_and_noncanonical_ids_preserved(tmp_path):
    path = tmp_path / 'manifest.csv'
    raw = dict.fromkeys(module.FIELDS, '')
    raw.update(file_name='001.nwb', cell_specimen_id=' 456', archive_uri='unrecognized:source')
    with path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(module.FIELDS)
        writer.writerow([raw[k] for k in module.FIELDS])
    rows = module.read_rows(path)
    assert rows[0]['raw_fields'] == raw
    assert rows[0]['canonical_cell_specimen_id'] is None
    assert rows[0]['recognized_file_session'] is None
