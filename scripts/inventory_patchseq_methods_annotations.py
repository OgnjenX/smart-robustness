"""Whitelisted source annotations only; confidence and eligibility remain unset."""

from __future__ import annotations

import csv
import hashlib
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import yaml

REG = Path('docs/validation-results/post2008-patchseq-methods-annotation-inventory-registration-1246.yaml')
REG_HASH = 'bf705326b456bf58ba00fee96be60f456bd8f69ad2f4b3aa275fd938f51bcb76'


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def project(source, header, fields):
    limit = csv.field_size_limit(4096)
    try:
        with Path(source).open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.reader(handle, strict=True)
            if next(reader) != header or not set(fields) <= set(header):
                raise ValueError('exact verified schema required')
            positions = {k: header.index(k) for k in fields}
            rows = []
            for number, values in enumerate(reader, 2):
                if len(values) != len(header) or len(rows) >= 50000:
                    raise ValueError('bounded source row shape required')
                raw = {k: values[position] for k, position in positions.items()}
                identity = raw['cell_id']
                rows.append({'source_row_number': number, 'raw_fields': raw,
                             'canonical_cell_id': int(identity) if re.fullmatch('[1-9][0-9]*', identity) else None})
    finally:
        csv.field_size_limit(limit)
    counts = Counter(r['canonical_cell_id'] for r in rows if r['canonical_cell_id'] is not None)
    samples = Counter(r['raw_fields']['sample_id'] for r in rows if r['raw_fields']['sample_id'])
    return {'rows': rows, 'source_rows': len(rows),
            'duplicate_canonical_cell_ids': sorted(k for k, n in counts.items() if n > 1),
            'duplicate_nonempty_sample_ids': sorted(k for k, n in samples.items() if n > 1),
            'noncanonical_cell_ids': sum(r['canonical_cell_id'] is None for r in rows),
            'missing_field_counts': {k: sum(r['raw_fields'][k] == '' for r in rows) for k in fields},
            'raw_value_counts': {k: dict(sorted(Counter(r['raw_fields'][k] for r in rows).items()))
                                 for k in ('mapping_quality', 'Gouwens_etal_2020', 'dataset', 'class')}}


def main():
    if digest(REG) != REG_HASH:
        raise ValueError('sealed annotation inventory registration changed')
    reg = yaml.load(REG.read_text(), Loader=yaml.CSafeLoader)
    for key in ('source', 'schema'):
        if digest(reg[key]) != reg[key+'_sha256']:
            raise ValueError('pinned source changed')
    header = yaml.load(Path(reg['schema']).read_text(), Loader=yaml.CSafeLoader)['schema']['columns']
    root = Path(reg['output_directory'])
    root.mkdir()
    report = {'status': 'methods-annotation-inventory-failure-retained',
              'registration_sha256': REG_HASH, 'collector_sha256': digest(__file__),
              'source_hashes': {reg[k]: reg[k+'_sha256'] for k in ('source', 'schema')},
              'started_at_utc': datetime.now(UTC).isoformat(), 'network_requests': 0,
              'physiological_features_interpreted': False, 'mapping_confidence_assigned': False,
              'specimen_eligibility_established': False}
    try:
        report.update(project(reg['source'], header, reg['fields']))
        if any(digest(reg[k]) != reg[k+'_sha256'] for k in ('source', 'schema')):
            raise ValueError('source changed during annotation projection')
        report['status'] = 'methods-annotation-inventory-independent-verification-pending'
    except (OSError, ValueError, csv.Error, UnicodeError) as exc:
        report['error'] = str(exc)
    report['finished_at_utc'] = datetime.now(UTC).isoformat()
    with (root/'manifest.yaml').open('x') as handle:
        yaml.safe_dump(report, handle, sort_keys=False)
    print(report['status'])
    print(yaml.safe_dump({k: report[k] for k in ('source_rows', 'noncanonical_cell_ids', 'raw_value_counts') if k in report}, sort_keys=False))
    raise SystemExit(0 if report['status'] == 'methods-annotation-inventory-independent-verification-pending' else 1)


if __name__ == '__main__':
    main()
