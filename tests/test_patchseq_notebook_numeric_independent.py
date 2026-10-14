"""Bitwise equality protects NaNs, sign bits, shape and exact raw values."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

scripts = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(scripts))
try:
    spec = importlib.util.spec_from_file_location('independent_numeric', scripts/'verify_patchseq_notebook_numeric.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
finally:
    sys.path.pop(0)


def test_nan_inf_and_negative_zero_preserved():
    values = np.array([[np.nan, np.inf, -np.inf, -0., 1.]])
    module.equal_bits(values.copy(), values)


@pytest.mark.parametrize('mutation', ['nan-fill', 'negative-zero', 'tiny-value', 'shape', 'dtype'])
def test_subtle_numeric_archive_corruption_rejected(mutation):
    source = np.array([[np.nan, -0., 1.]])
    saved = source.copy()
    if mutation == 'nan-fill':
        saved[0, 0] = 0
    elif mutation == 'negative-zero':
        saved[0, 1] = 0.
    elif mutation == 'tiny-value':
        saved[0, 2] = np.nextafter(1., 2.)
    elif mutation == 'shape':
        saved = saved.reshape(-1)
    else:
        saved = saved.astype('float32')
    with pytest.raises(ValueError):
        module.equal_bits(saved, source)
