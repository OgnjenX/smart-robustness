"""Versioned append transport cannot drift from the original sealed controls."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def functions(path):
    tree = ast.parse(path.read_text())
    return {node.name: ast.dump(node, include_attributes=False) for node in tree.body
            if isinstance(node, ast.FunctionDef)}


@pytest.mark.parametrize(('original', 'append', 'exclude'), [
    ('acquire_patchseq_intrinsic_recordings.py', 'acquire_patchseq_append_recordings.py', set()),
    ('verify_patchseq_intrinsic_acquisition.py', 'verify_patchseq_append_acquisition.py', {'main'}),
])
def test_transport_and_independent_check_functions_are_unchanged(original, append, exclude):
    before = functions(ROOT/'scripts'/original)
    after = functions(ROOT/'scripts'/append)
    assert before.keys() == after.keys()
    assert {k: v for k, v in before.items() if k not in exclude} == {
        k: v for k, v in after.items() if k not in exclude}


def test_append_has_separate_evidence_but_identical_transport_and_identity_rules():
    prefix = ROOT/'docs/validation-results'
    original = yaml.safe_load((prefix/'post2008-patchseq-intrinsic-acquisition-registration-1313.yaml').read_text())
    append = yaml.safe_load((prefix/'post2008-patchseq-append-acquisition-registration-1375.yaml').read_text())
    for key in ('execution', 'URL_resolution', 'storage_preflight', 'logical_metadata_reads',
                'identity_acceptance'):
        assert append[key] == original[key]
    for key in original['transport']:
        if key != 'expected_sealed_total_recording_bytes':
            assert append['transport'][key] == original['transport'][key]
    assert append['transport']['expected_sealed_total_recording_bytes'] == 1739316630
    assert append['output_directory'] != original['output_directory']
    for key in ('subset', 'independent_subset'):
        assert append[key] != original[key]
    assert append['physiological_arrays_or_descriptive_text_interpretation_authorized'] is False
