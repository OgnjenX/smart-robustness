"""Explicit compiled-vs-NumPy routing leaves original defaults unchanged."""

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

scripts = Path(__file__).parents[1]/'scripts'
spec = importlib.util.spec_from_file_location('patchseq_verification_routes', scripts/'patchseq_verification_routes.py')
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def test_original_route_without_new_requirement_unchanged():
    mod.require_backend({}, {})


def test_explicit_backend_required():
    config = {'expected_independent_backend': 'independent-compiled-RK4'}
    mod.require_backend(config, {'verification_backend': 'independent-compiled-RK4'})
    for wrong in ({}, {'verification_backend': 'NumPy'}, {'verification_backend': True}):
        with pytest.raises(ValueError): mod.require_backend(config, wrong)


@pytest.mark.parametrize('id,old_id,field', [(1459, 1447, 'output_directory'), (1461, 1450, 'output')])
def test_routes_are_separate_outputs_and_original_gates(id, old_id, field):
    directory = Path(__file__).parents[1]/'docs/validation-results'
    names = {1459: 'post2008-patchseq-selected-compiled-table-registration-1459.yaml',
             1447: 'post2008-patchseq-selected-fit-verification-registration-1447.yaml',
             1461: 'post2008-patchseq-compiled-training-assessment-registration-1461.yaml',
             1450: 'post2008-patchseq-terminal-training-assessment-registration-1450.yaml'}
    new, old = [yaml.safe_load((directory/names[i]).read_text()) for i in (id, old_id)]
    assert new[field] != old[field]
    assert new['protocol'] == old['protocol'] and new['seal'] == old['seal']
    assert new['expected_independent_backend'] == 'independent-compiled-RK4'
    assert new['held_out_access_authorized'] is False
    assert 'expected_independent_backend' not in old
