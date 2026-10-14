"""Structural discovery never accesses values or follows non-hard links."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

scripts = Path(__file__).resolve().parents[1]/'scripts'
spec = importlib.util.spec_from_file_location('verify_patchseq_NWB_metadata_pilot', scripts/'verify_patchseq_NWB_metadata_pilot.py')
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)
sys.modules.setdefault('verify_patchseq_NWB_metadata_pilot', reader)
spec = importlib.util.spec_from_file_location('structure', scripts/'inspect_patchseq_NWB_structure.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_names_only_no_dataset_or_attribute_values(tmp_path, monkeypatch):
    h5py = pytest.importorskip('h5py')
    source = tmp_path/'synthetic.nwb'
    with h5py.File(source, 'w') as file:
        file['general/subject/cell_id'] = 'never-read-this-value'
        file['general/subject'].attrs['identity_note'] = 'never-read-this-attribute'
        file['acquisition/trace/data'] = [1, 2, 3]
        file['general/link'] = h5py.SoftLink('/acquisition')
        file['general/external'] = h5py.ExternalLink('/never-open.h5', '/')
    def forbidden(*args, **kwargs):
        raise AssertionError('value access forbidden')
    monkeypatch.setattr(h5py.Dataset, '__getitem__', forbidden)
    monkeypatch.setattr(h5py.AttributeManager, '__getitem__', forbidden)
    records = []
    policy = yaml.safe_load(module.REG.read_text())['structure_policy']
    with h5py.File(source, 'r') as file:
        module.describe(file, policy, h5py, records)
    cell = next(r for r in records if r['path'] == '/general/subject/cell_id')
    assert cell['object_type'] == 'dataset' and cell['dataset_shape'] == []
    assert next(r for r in records if r['path'] == '/general/subject')['attribute_names'] == ['identity_note']
    assert not any(r['path'].startswith('/acquisition/') for r in records)
    assert next(r for r in records if r['path'] == '/general/external')['dereferenced'] is False
    assert 'never-read' not in str(records)


def test_structural_record_budget_is_terminal(tmp_path):
    h5py = pytest.importorskip('h5py')
    source = tmp_path/'synthetic.nwb'
    with h5py.File(source, 'w') as file:
        file['general/subject/cell_id'] = 1
    policy = yaml.safe_load(module.REG.read_text())['structure_policy']
    policy['maximum_unique_object_paths'] = 1
    records = []
    with h5py.File(source, 'r') as file, pytest.raises(ValueError, match='record bound'):
        module.describe(file, policy, h5py, records)
    assert len(records) == 1
