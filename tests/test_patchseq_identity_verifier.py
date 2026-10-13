"""Independent relational verifier detects altered identity evidence."""

from __future__ import annotations

import copy
import csv
import importlib.util
import io
import tarfile
from pathlib import Path

import pytest


def load(name):
    path = Path(__file__).resolve().parents[1] / 'scripts' / (name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


verifier = load('verify_patchseq_identity')
collector = load('join_patchseq_identity')


def fixture():
    assets = [{'asset_id': str(i), 'path': f'unknown-{i}.nwb'} for i in range(4435)]
    assets[0]['path'] = 'sub-10/sub-10_ses-1000_icephys.nwb'
    raw = dict.fromkeys(verifier.FIELDS, '')
    raw.update(cell_specimen_id='100', donor_id='11', ephys_session_id='1000')
    rows = [{'source_row_number': 2, 'raw_fields': raw,
             'canonical_ids': {field: verifier.decimal(raw[field]) for field in verifier.IDS}}]
    old = [{'specimen_id': i, 'included': i <= 111} for i in range(1, 2334)]
    return assets, rows, old


@pytest.mark.parametrize('fault', ['none', 'duplicate-source', 'duplicate-catalog', 'unmatched', 'malformed'])
def test_independent_join_reconstructs_conflicts(fault):
    assets, rows, old = fixture()
    if fault == 'duplicate-source':
        other = copy.deepcopy(rows[0])
        other['source_row_number'] = 3
        rows.append(other)
    elif fault == 'duplicate-catalog':
        assets[1]['path'] = 'sub-12/sub-12_ses-1000_icephys.nwb'
    elif fault == 'unmatched':
        rows[0]['raw_fields']['ephys_session_id'] = '999'
        rows[0]['canonical_ids']['ephys_session_id'] = 999
    elif fault == 'malformed':
        assets[0]['path'] = 'sub-10/sub-11_ses-1000_icephys.nwb'
    expected = verifier.rebuild(assets, rows, old)
    verifier.compare(expected, collector.join_rows(assets, rows, old))
    assert expected['summary']['path_annotation_consistent_assets'] == 0
    assert not expected['summary']['biological_eligibility_established']


@pytest.mark.parametrize('value', ['', '001', '1.0', '+1', ' 1', '١', None, True, 1])
def test_decimal_does_not_repair_source(value):
    assert verifier.decimal(value) is None


@pytest.mark.parametrize('key', ['assets', 'annotation_rows', 'summary', 'prior_specimen_ID_equalities',
                               'unreferenced_source_row_numbers', 'duplicate_source_session_IDs'])
def test_tampered_projection_rejected(key):
    expected = verifier.rebuild(*fixture())
    altered = copy.deepcopy(expected)
    altered[key] = None
    with pytest.raises(ValueError, match=key):
        verifier.compare(expected, altered)


def test_false_identity_promotion_rejected():
    expected = verifier.rebuild(*fixture())
    altered = copy.deepcopy(expected)
    altered['assets'][0]['internal_recording_identity_confirmed'] = True
    with pytest.raises(ValueError, match='assets'):
        verifier.compare(expected, altered)


def test_bool_not_interchangeable_with_integer():
    expected = verifier.rebuild(*fixture())
    altered = copy.deepcopy(expected)
    altered['annotation_rows'][0]['canonical_ids']['cell_specimen_id'] = True
    with pytest.raises(ValueError):
        verifier.compare(expected, altered)


@pytest.mark.parametrize('fault', ['none', 'short-row', 'long-row', 'changed-header'])
def test_csv_reconstruction_projects_only_registered_columns(tmp_path, fault):
    header = list(verifier.FIELDS) + ['medical_conditions']
    values = [''] * len(header)
    values[header.index('cell_specimen_id')] = '001'
    values[-1] = 'irrelevant-sensitive-value'
    stream = io.StringIO(newline='')
    writer = csv.writer(stream)
    writer.writerow(header if fault != 'changed-header' else header[:-1])
    writer.writerow(values[:-1] if fault == 'short-row' else values + ['extra'] if fault == 'long-row' else values)
    data = stream.getvalue().encode()
    path = tmp_path / 'annotations.tar'
    with tarfile.open(path, 'w:') as archive:
        member = tarfile.TarInfo('annotations.csv')
        member.size = len(data)
        archive.addfile(member, io.BytesIO(data))
    schema = {'CSV_member': 'annotations.csv', 'columns': header}
    if fault != 'none':
        with pytest.raises(ValueError):
            verifier.raw_rows(path, schema)
    else:
        rows = verifier.raw_rows(path, schema)
        assert rows[0]['raw_fields']['cell_specimen_id'] == '001'
        assert rows[0]['canonical_ids']['cell_specimen_id'] is None
        assert set(rows[0]['raw_fields']) == set(verifier.FIELDS)
        assert 'medical_conditions' not in rows[0]['raw_fields']
