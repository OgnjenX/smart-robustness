"""Local continuation preserves the sealed names-only scope and offline limits."""

from pathlib import Path

import yaml


def test_local_continuation_preserves_original_scope_and_failed_outcomes():
    reg = yaml.safe_load(Path('docs/validation-results/post2008-patchseq-local-structure-registration-1289.yaml').read_text())
    original = yaml.safe_load(Path(reg['structural_registration']).read_text())
    assert reg['structure_scope'] == 'exactly-1271-without-added-fields-depth-or-count-limits'
    assert original['structure_policy']['maximum_unique_object_paths'] == 256
    assert original['structure_policy']['maximum_descendant_depth_per_root'] == 2
    assert reg['network_requests_allowed'] == reg['new_download_bytes_allowed'] == 0
    assert reg['dataset_values_allowed'] is reg['attribute_values_allowed'] is False
    assert reg['cohort_qualification_authorized'] is False
    assert reg['output'] != original['output']
    assert '1290' in reg['output']
