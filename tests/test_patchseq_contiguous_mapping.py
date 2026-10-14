"""Published last-block selection without raw source mutation or earlier fallback."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
from test_patchseq_setting_mapping import fixture, run

scripts = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(scripts))
try:
    spec = importlib.util.spec_from_file_location('contiguous_mapping', scripts/'continue_patchseq_contiguous_mapping.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
finally:
    sys.path.pop(0)


def test_mixed_missing_modes_do_not_impute_and_raw_arrays_unchanged():
    table, data, metadata = fixture()
    data['column-0001'][1, 0] = np.nan
    before = {key: values.tobytes() for key, values in data.items()}
    assert run(table, data, metadata)['mapping_status'] == 'unknown'
    view, proof = module.lookup_view(table, data, 7)
    assert proof['selected_acquisition_rows'] == [0]
    assert run(table, view, metadata)['mapping_status'] == 'mapped'
    assert all(values.tobytes() == before[key] for key, values in data.items())
    assert np.isnan(view['column-0001'][1, 0])


def test_only_last_contiguous_block_no_favorable_older_field_fallback():
    table, data, metadata = fixture()
    data = {key: np.repeat(values[:1], 3, axis=0) for key, values in data.items()}
    data['column-0001'][:, 0] = [0, 1, 0]
    field = table['column_candidates']['Bridge Bal Value'][0]
    data[f'column-{field:04d}'][2, :] = np.nan
    view, proof = module.lookup_view(table, data, 7)
    assert proof['selected_acquisition_rows'] == [2]
    out = run(table, view, metadata)
    assert out['fields']['bridge_balance']['concordance'] == 'unknown'
    assert 'value' not in out['fields']['bridge_balance']['native_value']


def test_absent_explicit_mode_no_legacy_guess():
    table, data, _ = fixture()
    data['column-0001'][:, 0] = np.nan
    view, proof = module.lookup_view(table, data, 7)
    assert view is None and proof['undefined'] == 'no-explicit-acquisition-block-legacy-branch-not-ported'
