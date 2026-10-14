"""Column access guard and historical vector-last-row distinction."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

scripts = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(scripts))
try:
    spec = importlib.util.spec_from_file_location('numeric_notebook', scripts/'read_patchseq_notebook_numeric.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
finally:
    sys.path.pop(0)


def test_only_selected_columns_all_nine_pages_and_duplicates_retained(tmp_path, monkeypatch):
    h5py = pytest.importorskip('h5py')
    labels = ['SweepNum', 'unselected', 'Hold', 'Hold']
    header = {'path': '/general/labnotebook/device/numericalKeys', 'shape': [3, 4],
              'rows': [[{'decoded_text': label} for label in labels]]*3}
    data = np.arange(3*4*9., dtype='float64').reshape(3, 4, 9)
    data[1, 2, 0] = np.nan
    with h5py.File(tmp_path/'fixture.h5', 'w') as file:
        file['general/labnotebook/device/numericalValues'] = data
        original = h5py.Dataset.__getitem__
        def guard(node, key):
            assert node.name.endswith('/numericalValues')
            assert key[1] in (0, 2, 3) and key[0] == slice(None) and key[2] == slice(None)
            return original(node, key)
        monkeypatch.setattr(h5py.Dataset, '__getitem__', guard)
        arrays, info = module.extract(file, header, ['SweepNum', 'Hold', 'absent'], h5py, [0])
        assert info['column_candidates'] == {'SweepNum': [0], 'Hold': [2, 3], 'absent': []}
        assert arrays['column-0002'].shape == (3, 9)
        assert np.isnan(arrays['column-0002'][1, 0])


def test_vector_last_row_is_not_headstage_last_finite():
    values = np.full((3, 9), np.nan)
    values[0, 0] = 10
    values[1, 1] = 20
    values[2, 0] = 30
    row = module.last_explicit_mode_row(np.array([5, 5, 5]), np.array([0, 0, 1]), values, 5, 0)
    assert row == {'row': 1} and np.isnan(values[row['row'], 0])


def test_missing_mode_and_missing_sweep_never_filled():
    v = np.ones((2, 9))
    assert 'undefined' in module.last_explicit_mode_row(np.array([5, 5]), np.array([0, np.nan]), v, 5, 0)
    assert module.last_explicit_mode_row(np.array([5, 5]), np.array([0, 0]), v, 6, 0) == {'undefined': 'sweep-absent'}


def test_nonfinite_selected_vector_not_skipped_for_older_finite_values():
    v = np.full((2, 9), np.nan)
    v[0, 0], v[1, 0] = 10, np.inf
    assert module.last_explicit_mode_row(np.array([5, 5]), np.array([0, 0]), v, 5, 0) == {'row': 1}
