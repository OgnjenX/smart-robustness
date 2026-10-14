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
    ('inventory_patchseq_trace_schema.py', 'inventory_patchseq_append_trace_schema.py', set()),
    ('verify_patchseq_trace_schema.py', 'verify_patchseq_append_trace_schema.py', {'main'}),
    ('check_patchseq_pairing.py', 'check_patchseq_append_pairing.py', set()),
    ('verify_patchseq_pairing.py', 'verify_patchseq_append_pairing.py', {'main'}),
    ('run_patchseq_waveform_QC.py', 'run_patchseq_append_waveform_QC.py', set()),
    ('verify_patchseq_waveform_QC.py', 'verify_patchseq_append_waveform_QC.py', {'main'}),
    ('audit_patchseq_health_protocols.py', 'audit_patchseq_append_health_protocols.py', set()),
    ('verify_patchseq_health_protocols.py', 'verify_patchseq_append_health_protocols.py', {'main'}),
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


def test_append_metadata_scope_and_interpretation_are_unchanged():
    prefix = ROOT/'docs/validation-results'
    original = yaml.safe_load((prefix/'post2008-patchseq-trace-schema-registration-1318.yaml').read_text())
    append = yaml.safe_load((prefix/'post2008-patchseq-append-trace-schema-registration-1380.yaml').read_text())
    for key in ('scope', 'interpretation', 'parser_dependency', 'HDF5_plugin_preload'):
        assert append[key] == original[key]
    assert append['waveform_or_timestamp_vector_access_authorized'] is False


def test_append_waveform_rules_and_physical_conventions_are_unchanged():
    prefix = ROOT/'docs/validation-results'
    original = yaml.safe_load((prefix/'post2008-patchseq-waveform-QC-registration-1327.yaml').read_text())
    append = yaml.safe_load((prefix/'post2008-patchseq-append-waveform-QC-registration-1388.yaml').read_text())
    for key in ('read_policy', 'conventions', 'waveform_rules', 'scientific_claims'):
        assert append[key] == original[key]
    assert append['scope']['pairs'] == 'all-2550-verified-current-clamp-pairs-retained'


def test_append_health_ontology_codes_and_pairing_rules_are_unchanged():
    prefix = ROOT/'docs/validation-results'
    original = yaml.safe_load((prefix/'post2008-patchseq-health-protocol-registration-1360.yaml').read_text())
    append = yaml.safe_load((prefix/'post2008-patchseq-append-health-protocol-registration-1393.yaml').read_text())
    for key in ('primary_ontology', 'codes', 'rules', 'scope_limit'):
        assert append[key] == original[key]
    assert append['new_HDF5_or_waveform_reads_authorized'] is False


def test_append_recording_quality_measurement_and_gates_are_identical():
    for original, append in [('measure_patchseq_recording_quality.py', 'measure_patchseq_append_recording_quality.py'),
                              ('verify_patchseq_recording_quality.py', 'verify_patchseq_append_recording_quality.py')]:
        before, after = functions(ROOT/'scripts'/original), functions(ROOT/'scripts'/append)
        for key in before:
            if key != 'main':
                assert before[key] == after[key]
    prefix = ROOT/'docs/validation-results'
    original = yaml.safe_load((prefix/'post2008-patchseq-recording-quality-registration-1364.yaml').read_text())
    append = yaml.safe_load((prefix/'post2008-patchseq-append-recording-quality-registration-1399.yaml').read_text())
    for key in ('measurement_rules', 'mandatory_measurement_gate', 'blowout', 'primary_reference'):
        assert append[key] == original[key]
    for key in original['scope']:
        if key != 'trials':
            assert append['scope'][key] == original['scope'][key]
