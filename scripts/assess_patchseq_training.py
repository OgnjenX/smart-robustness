"""Pinned terminal training assessment; no held-out response or network authority."""

from __future__ import annotations

import fcntl
import json
from contextlib import contextmanager
from pathlib import Path

import yaml
from audit_patchseq_fit_checkpoint import audit
from patchseq_training_assessment_gate import assess_training
from patchseq_verification_routes import require_backend
from verify_patchseq_fit_checkpoint_structure import (
    digest,
    read_yaml,
    terminal_checkpoint,
    verify_source_pins,
    verify_training_bodies,
)

REG = Path('docs/validation-results/post2008-patchseq-terminal-training-assessment-registration-1450.yaml')


@contextmanager
def completed_verification(path, status):
    path = Path(path)
    with (path.parent/'verifier.lock').open('r') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('verification process is live') from exc
        sha = digest(path)
        state = json.loads(path.read_text())
        if state.get('status') != status:
            raise ValueError('verification snapshot is incomplete')
        yield state, sha
        if digest(path) != sha:
            raise ValueError('verification snapshot changed during assessment')


def replay_disagreement(original, independent, protocol):
    """Retain terminal verification disagreement without needing selected-stage access."""
    if set(independent['strata']) != set(protocol['strata']):
        raise ValueError('independent stratum scope differs')
    keys = {str(i) for i in range(protocol['search']['candidates_per_stratum'])}
    rows = []
    for stratum in protocol['strata']:
        cases = independent['strata'][stratum]['candidates']
        if set(cases) != keys or set(original['strata'][stratum]['candidates']) != keys:
            raise ValueError('independent table coverage incomplete')
        if any(type(row['passed']) is not bool for row in cases.values()):
            raise ValueError('independent agreement claim not boolean')
        failures = [key for key, row in cases.items() if not row['passed']]
        rows.append({'stratum': stratum, 'disagreeing_candidate_ids': failures,
                     'classification': 'verification-disagreement-not-biological-failure' if failures
                     else 'table-reconstructed-selected-stage-not-assessed',
                     'eligible_to_seal_held_out_design': False})
    if not any(r['disagreeing_candidate_ids'] for r in rows) or independent['all_reconstructions_agree'] is not False:
        raise ValueError('no verified disagreement or inconsistent aggregate')
    return {'status': 'verification-disagreement-training-interpretation-unresolved',
            'strata': rows, 'complete_map_training_eligible': False,
            'held_out_access_authorized': False, 'network_promotion': False,
            'biological_validation_complete': False, 'scientific_goal_complete': False}


def main(registration=REG):
    reg = read_yaml(registration)
    protocol, seal = read_yaml(reg['protocol']), read_yaml(reg['seal'])
    catalog = read_yaml(protocol['training_catalog'])
    structure = read_yaml(reg['structure_result'])
    pins = {str(registration): digest(registration), str(Path(__file__)): digest(__file__),
            'scripts/patchseq_training_assessment_gate.py': digest('scripts/patchseq_training_assessment_gate.py'),
            'scripts/patchseq_verification_routes.py': digest('scripts/patchseq_verification_routes.py'),
            reg['structure_result']: digest(reg['structure_result'])}
    with terminal_checkpoint(protocol['output']) as (original, original_sha), completed_verification(
        reg['independent_checkpoint'], 'all-candidate-forward-feature-loss-reconstruction-retained') as (independent, independent_sha):
        require_backend(reg, independent)
        if structure['checkpoint_sha256'] != original_sha or independent['input_checkpoint_sha256'] != original_sha:
            raise ValueError('assessment snapshots refer to different main results')
        audit(original, protocol, catalog, seal)
        for evidence in (original, structure, independent):
            verify_source_pins(evidence['source_hashes'])
            pins.update(evidence['source_hashes'])
        pins[reg['independent_checkpoint']] = independent_sha
        if independent['held_out_access_authorized'] is not False or independent['network_promotion'] is not False:
            raise ValueError('independent result exceeds access/claim boundary')
        if independent['all_reconstructions_agree'] is False:
            report = replay_disagreement(original, independent, protocol)
        elif independent['all_reconstructions_agree'] is True:
            with completed_verification(reg['selected_checkpoint'],
                'selected-training-waveform-feature-and-numerical-checks-retained') as (selected, selected_sha):
                verify_source_pins(selected['source_hashes'])
                pins.update(selected['source_hashes'])
                pins[reg['selected_checkpoint']] = selected_sha
                report = assess_training(original, independent, selected, protocol)
        else:
            raise ValueError('aggregate replay agreement not explicit boolean')
        if verify_training_bodies(original, protocol, catalog) != structure['training_body_hashes']:
            raise ValueError('recording bodies changed since source audit')
        verify_source_pins(pins)
        report.update(source_hashes=pins, original_checkpoint_sha256=original_sha,
                      terminal_training_source_integrity_verified=True,
                      frozen_baseline_modified=False,
                      interpretation='bounded-training-study-under-sealed-input-model-and-search-assumptions-not-ART-or-all-neuron-family-conclusion')
    # Lock-context snapshot checks must succeed before publication.
    destination = Path(reg['output'])
    destination.parent.mkdir()
    with destination.open('x') as file:
        yaml.safe_dump(report, file, sort_keys=False)
    print('Pinned training assessment retained; held-out and network access remain closed')


if __name__ == '__main__':
    main()
