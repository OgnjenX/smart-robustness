"""Current-context archives preserve units, duplicates and raw value bits."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
try:
    from read_patchseq_current_context_numeric import extract_context
    from verify_patchseq_current_context_numeric import equal_bits
finally:
    sys.path.pop(0)


def header():
    names = ['SweepNum', 'unselected', 'I-Clamp Holding Level', 'Slow current injection',
             'Slow current injection level', 'I-Clamp Holding Level']
    return {'path': '/general/labnotebook/device/numericalKeys', 'shape': [3, len(names)],
            'rows': [[{'decoded_text': name} for name in names],
                     [{'decoded_text': unit} for unit in ('', '', 'pA', 'On/Off', 'V', 'pA')],
                     [{'decoded_text': ''} for _ in names]]}


def test_only_whitelisted_columns_all_source_rows_pages_and_unit_headers(tmp_path, monkeypatch):
    h5py = pytest.importorskip('h5py')
    data = np.arange(4*6*9., dtype='float64').reshape(4, 6, 9)
    data[0, 2, 0], data[1, 3, 1], data[2, 4, 2] = -0., np.nan, np.inf
    with h5py.File(tmp_path/'fixture.h5', 'w') as file:
        path = '/general/labnotebook/device/numericalValues'
        file[path] = data
        file['general/labnotebook/device/textualValues'] = 'not-authorized'
        original = h5py.Dataset.__getitem__
        def guard(node, index):
            assert node.name == path and index[0] == index[2] == slice(None)
            assert index[1] in (0, 2, 3, 4, 5)
            return original(node, index)
        monkeypatch.setattr(h5py.Dataset, '__getitem__', guard)
        budget = [0]
        arrays, metadata = extract_context(file, header(), ['SweepNum', 'I-Clamp Holding Level',
            'Slow current injection', 'Slow current injection level', 'missing'], h5py, budget)
        assert metadata['column_candidates']['I-Clamp Holding Level'] == [2, 5]
        assert metadata['column_candidates']['missing'] == []
        assert metadata['column_headers'][4][1]['decoded_text'] == 'V'
        assert budget[0] == 4*5*9*8
        for k in (0, 2, 3, 4, 5):
            equal_bits(arrays[f'column-{k:04d}'], data[:, k, :])


@pytest.mark.parametrize('case', ['float32', 'soft', 'external', 'plugin', 'budget'])
def test_unsupported_storage_and_budget_stop_before_values(tmp_path, monkeypatch, case):
    h5py = pytest.importorskip('h5py')
    with h5py.File(tmp_path/'fixture.h5', 'w') as file:
        path = '/general/labnotebook/device/numericalValues'
        if case == 'soft':
            file[path] = h5py.SoftLink('/never-read')
        elif case == 'external':
            file[path] = h5py.ExternalLink('/never-open', '/')
        else:
            file.create_dataset(path, shape=(4, 6, 9), dtype='float32' if case == 'float32' else 'float64',
                                compression='lzf' if case == 'plugin' else None)
        monkeypatch.setattr(h5py.Dataset, '__getitem__', lambda *a: pytest.fail('forbidden value read'))
        budget = [268435456] if case == 'budget' else [0]
        with pytest.raises(ValueError):
            extract_context(file, header(), ['SweepNum'], h5py, budget)


def test_independent_bit_check_rejects_signed_zero_and_filled_missing_values():
    for original, modified in [(np.array([-0.]), np.array([0.])),
                               (np.array([np.nan]), np.array([0.])),
                               (np.array([np.inf]), np.array([1.]))]:
        with pytest.raises(ValueError):
            equal_bits(modified, original)
