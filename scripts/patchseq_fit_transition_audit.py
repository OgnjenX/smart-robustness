"""Independent checkpoint-origin accounting; not a forward validation claim."""

from __future__ import annotations

import json
from pathlib import Path

from verify_patchseq_whole_file_integrity import digest


def audit_transition(state, protocol):
    output = Path(protocol['output'])
    archive = output/'predecessor-checkpoint.json'
    sha = digest(archive)
    if sha != state['predecessor_checkpoint_sha256']:
        raise ValueError('archived predecessor hash differs')
    old = json.loads(archive.read_text())
    if (old.get('status') is not None or state['fitting_backend'] != 'compiled-RK4-with-retained-Brian2-prefix'
        or old['candidate_table'] != state['candidate_table']
        or set(state['transition_checks']) != set(protocol['strata'])):
        raise ValueError('transition provenance scope differs')
    if state['source_hashes'].get(str(archive)) != sha:
        raise ValueError('predecessor archive not pinned')
    for stratum in protocol['strata']:
        rows = state['strata'][stratum]
        prefix = old['strata'].get(stratum, {}).get('candidates', {})
        checks = state['transition_checks'][stratum]
        if set(checks) != set(prefix) or any(c['passed'] is not True for c in checks.values()):
            raise ValueError('retained original verification incomplete')
        if set(rows['candidate_backends']) != set(rows['candidates']):
            raise ValueError('candidate origin support differs')
        for key, row in rows['candidates'].items():
            expected = 'retained-Brian2' if key in prefix else 'compiled-RK4'
            if rows['candidate_backends'][key] != expected:
                raise ValueError('candidate backend origin differs')
            if key in prefix and row != prefix[key]:
                raise ValueError('retained original candidate changed')
    build = state['compiled_build']
    if state['source_hashes'].get(build['binary_path']) != build['binary_sha256']:
        raise ValueError('fitting binary provenance not pinned')
    return {'predecessor_checkpoint_sha256': sha, 'retained_original_rows_verified_and_unchanged': True,
            'explicit_candidate_backend_coverage_verified': True}
