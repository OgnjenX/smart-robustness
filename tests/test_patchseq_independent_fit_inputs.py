"""Synthetic command-reader access and source identity controls."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

scripts = Path(__file__).parents[1]/'scripts'
for name in ('verify_patchseq_whole_file_integrity', 'verify_patchseq_fitting_targets',
             'patchseq_independent_fit_inputs'):
    spec = importlib.util.spec_from_file_location(name, scripts/(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
mod = sys.modules['patchseq_independent_fit_inputs']


def test_held_out_gate_precedes_metadata_access(monkeypatch):
    monkeypatch.setattr(mod, 'metadata', lambda path: pytest.fail('metadata opened'))
    with pytest.raises(ValueError, match='held-out'):
        mod.inputs_for_training([{'partition': 'held-out'}], {}, {})


def test_empty_scope_rejected():
    with pytest.raises(ValueError): mod.inputs_for_training([], {}, {})


def test_pooling_rejected_before_access():
    records = [{'partition': 'fitting', 'source_group': 'Sst', 'layer_label': layer}
               for layer in ('L23', 'L5')]
    with pytest.raises(ValueError, match='pooling'): mod.inputs_for_training(records, {}, {})


def test_response_descriptor_cannot_call_decoder(monkeypatch):
    monkeypatch.setattr(mod, 'read', lambda *args: pytest.fail('response decoder invoked'))
    with pytest.raises(ValueError, match='response'):
        mod.command_array(None, {'role': 'response'}, None)


def test_exact_source_conversion_and_bounded_descriptor(tmp_path):
    import h5py
    path = tmp_path/'synthetic.h5'
    with h5py.File(path, 'w') as file:
        file.create_dataset('command/data', data=np.array([0., -30., 50., 0.]))
    descriptor = {'role': 'stimulus', 'path': '/command', 'data_shape': [4], 'conversion': 1e-12}
    with h5py.File(path, 'r') as file:
        array, charge = mod.command_array(file, descriptor, h5py)
        np.testing.assert_array_equal(array, [0., -30., 50., 0.])
        assert charge == 32
        with pytest.raises(ValueError):
            mod.command_array(file, descriptor | {'data_shape': [3]}, h5py)


def test_external_link_rejected(tmp_path):
    import h5py
    path = tmp_path/'synthetic.h5'
    with h5py.File(path, 'w') as file:
        file['command'] = h5py.ExternalLink('unread.h5', '/command')
    with h5py.File(path, 'r') as file, pytest.raises(ValueError):
        mod.command_array(file, {'role': 'stimulus', 'path': '/command',
                                'data_shape': [4], 'conversion': 1e-12}, h5py)
