"""Terminal-only checkpoint/source audit; forward validation remains separate."""

from __future__ import annotations

import fcntl
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

import yaml
from audit_patchseq_fit_checkpoint import audit
from patchseq_fit_transition_audit import audit_transition

REG = Path('docs/validation-results/post2008-patchseq-amended-fit-structure-registration-1469.yaml')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read_yaml(path):
    return yaml.load(Path(path).read_text(), Loader=yaml.CSafeLoader)


@contextmanager
def terminal_checkpoint(directory):
    """Refuse a live runner; retain lock so it cannot resume during inspection."""
    directory = Path(directory)
    path, lock_path = directory/'checkpoint.json', directory/'runner.lock'
    if not path.is_file() or not lock_path.is_file():
        raise ValueError('terminal checkpoint or runner lock missing')
    with lock_path.open('r') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('runner is live; verification must wait') from exc
        before = digest(path)
        state = json.loads(path.read_text())
        if state.get('status') != 'training-search-and-selected-numerical-controls-retained-independent-verification-pending':
            raise ValueError('incomplete or unrecognized checkpoint')
        yield state, before
        if digest(path) != before:
            raise ValueError('checkpoint changed during audit')


def verify_source_pins(pins):
    if not pins:
        raise ValueError('source pins missing')
    for path, expected in pins.items():
        if digest(path) != expected:
            raise ValueError('source hash differs: '+str(path))


def verify_training_bodies(state, protocol, catalog):
    """Whole-file digests only for opened training cases; no response decoding."""
    source = read_yaml('docs/validation-results/post2008-patchseq-fitting-target-registration-1429.yaml')
    batches = {b['name']: b for b in source['batches']}
    records = {(r['source_batch'], r['slot']): r for r in catalog['records']}
    expected = set()
    for stratum in protocol['strata']:
        for row in state['strata'][stratum]['opened_training_files']:
            key = (row['source_batch'], row['slot'])
            if row['partition'] != 'fitting' or records[key]['partition'] != 'fitting':
                raise ValueError('held-out body access forbidden')
            expected.add(key)
    receipts, checked = {}, []
    for batch, slot in sorted(expected):
        if batch not in receipts:
            receipts[batch] = {r['slot']: r for r in read_yaml(batches[batch]['acquisition'])['receipts']}
        item = receipts[batch][slot]
        if digest(item['path']) != item['sha256']:
            raise ValueError('training receipt differs')
        receipt = read_yaml(item['path'])
        selection = next(s for s in read_yaml(batches[batch]['selection'])['slots'] if s['slot'] == slot)
        if (selection['partition'] != 'fitting'
            or selection['source_cell_specimen_id'] != records[(batch, slot)]['source_cell_specimen_id']):
            raise ValueError('training body identity differs')
        actual = digest(receipt['complete_body_path'])
        if actual != selection['expected_SHA256'] or actual != receipt['complete_file_SHA256']:
            raise ValueError('training body changed')
        checked.append({'source_batch': batch, 'slot': slot, 'sha256': actual})
    return checked


def main():
    reg = read_yaml(REG)
    protocol, seal = read_yaml(reg['protocol']), read_yaml(reg['seal'])
    if Path(reg['input_directory']) != Path(protocol['output']):
        raise ValueError('verification output scope differs')
    verify_source_pins(seal['source_sha256'])
    pins = {str(REG): digest(REG), str(Path(__file__)): digest(__file__),
            'scripts/audit_patchseq_fit_checkpoint.py': digest('scripts/audit_patchseq_fit_checkpoint.py')}
    pins.update({reg[key]: digest(reg[key]) for key in ('protocol', 'seal')})
    for key in ('training_catalog', 'verified_catalog'):
        if digest(protocol[key]) != protocol[key+'_sha256']:
            raise ValueError('catalog changed')
        pins[protocol[key]] = protocol[key+'_sha256']
    catalog = read_yaml(protocol['training_catalog'])
    verified = read_yaml(protocol['verified_catalog'])
    if verified['status'] != 'all-training-response-features-and-held-out-no-access-boundaries-independently-reconstructed':
        raise ValueError('independent catalog prerequisite missing')
    with terminal_checkpoint(reg['input_directory']) as (state, before):
        verify_source_pins(state['source_hashes'])
        verify_source_pins(verified['source_hashes'])
        transition = audit_transition(state, protocol)
        report = audit(state, protocol, catalog, seal)
        report.update(transition)
        bodies = verify_training_bodies(state, protocol, catalog)
        verify_source_pins(state['source_hashes'])
        verify_source_pins(pins)
        report.update(checkpoint_sha256=before, source_hashes=pins,
                      training_body_hashes=bodies, runner_lock_held_during_verification=True,
                      response_arrays_decoded=False, forward_simulations_verified=False)
    destination = Path(reg['output'])
    destination.parent.mkdir()
    with destination.open('x') as file:
        yaml.safe_dump(report, file, sort_keys=False)
    print('Terminal structure and training body hashes checked; forward validation remains pending')


if __name__ == '__main__':
    main()
