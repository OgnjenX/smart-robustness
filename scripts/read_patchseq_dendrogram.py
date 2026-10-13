"""Bounded source-tree projection; no marker interpretation or class assignment."""

from __future__ import annotations

import hashlib
import json
import math
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import yaml

REG = Path('docs/validation-results/post2008-patchseq-dendrogram-read-registration-1220.yaml')
FIELDS = ('label', 'cell_set_accession', 'cell_set_preferred_alias', 'cell_set_aligned_alias',
          'taxonomy_id', 'cell_set_label')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('duplicate JSON object key')
        value[key] = item
    return value


def invalid_constant(value):
    raise ValueError('nonfinite JSON constant: ' + value)


def finite_float(value):
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError('nonfinite JSON exponent')
    return parsed


def project(data):
    if len(data) > 1048576:
        raise ValueError('reference JSON byte bound exceeded')
    root = json.loads(data.decode('utf-8'), object_pairs_hook=unique_object,
                      parse_constant=invalid_constant, parse_float=finite_float)
    stack, outputs = [(root, 'root', 0)], []
    while stack:
        node, path, depth = stack.pop()
        if depth > 64 or len(outputs) >= 5000 or type(node) is not dict:
            raise ValueError('tree depth, count or node type differs')
        if set(node) == {'leaf_attributes'}:
            leaf, key, children = True, 'leaf_attributes', []
        elif set(node) == {'node_attributes', 'children'}:
            leaf, key = False, 'node_attributes'
            children = node['children']
            if type(children) is not list or len(children) < 2:
                raise ValueError('nonempty branching child list required')
        else:
            raise ValueError('unsupported serializer node shape')
        attributes = node[key]
        if type(attributes) is list:
            if len(attributes) != 1 or type(attributes[0]) is not dict:
                raise ValueError('one-record attribute array required')
            attributes = attributes[0]
            encoding = 'one-record-array'
        elif type(attributes) is dict:
            encoding = 'object'
        else:
            raise ValueError('attribute record shape differs')
        projected = {field: {'present': field in attributes, 'raw_value': attributes.get(field),
                            'JSON_type': type(attributes[field]).__name__ if field in attributes else None}
                     for field in FIELDS}
        outputs.append({'path': path, 'depth': depth, 'leaf': leaf, 'attribute_encoding': encoding,
                        'source_attribute_keys': sorted(attributes), 'attributes': projected,
                        'children': [path + '/children/' + str(i) for i in range(len(children))]})
        stack.extend((child, path + '/children/' + str(i), depth + 1)
                     for i, child in reversed(list(enumerate(children))))
    return {'nodes': outputs, 'summary': {'nodes': len(outputs), 'leaves': sum(n['leaf'] for n in outputs),
                                        'maximum_depth': max(n['depth'] for n in outputs)},
            'class_membership_established': False, 'biological_eligibility_established': False}


def main():
    if digest(REG) != 'a5cf1562e92df7b5599005729968c68a314f24e697d8e092bf3ce109d5c3c719':
        raise ValueError('sealed source-tree registration changed')
    reg = yaml.safe_load(REG.read_text())
    archive = Path(reg['archive'])
    if digest(archive) != reg['archive_sha256']:
        raise ValueError('pinned source archive changed')
    root = Path(reg['output_directory'])
    root.mkdir()
    report = {'status': 'reference-dendrogram-read-failure-retained', 'registration_sha256': digest(REG),
              'collector_sha256': digest(__file__), 'started_at_utc': datetime.now(UTC).isoformat(),
              'network_requests': 0, 'archive_extracted': False, 'R_objects_executed': False,
              'physiological_arrays_read': False, 'biological_eligibility_established': False}
    try:
        with zipfile.ZipFile(archive) as source:
            candidates = [m for m in source.infolist() if m.filename == reg['member']]
            if (len(candidates) != 1 or candidates[0].is_dir() or candidates[0].file_size != reg['expected_bytes']
                or candidates[0].file_size > 1048576 or candidates[0].flag_bits & 1
                or (candidates[0].external_attr >> 16) & 0o170000 == 0o120000):
                raise ValueError('exact bounded regular source JSON required')
            data = source.read(candidates[0])
        if len(data) != reg['expected_bytes']:
            raise ValueError('actual source JSON byte extent differs')
        report.update(source_member_sha256=hashlib.sha256(data).hexdigest(), source_member_bytes=len(data))
        report.update(project(data))
        if digest(archive) != reg['archive_sha256']:
            raise ValueError('source changed during tree projection')
        report['status'] = 'reference-dendrogram-projected-independent-verification-pending'
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile, RecursionError) as exc:
        report['error'] = str(exc)
    report['finished_at_utc'] = datetime.now(UTC).isoformat()
    with (root / 'manifest.yaml').open('x') as handle:
        yaml.safe_dump(report, handle, sort_keys=False)
    print(report['status'])
    raise SystemExit(0 if report['status'] == 'reference-dendrogram-projected-independent-verification-pending' else 1)


if __name__ == '__main__':
    main()
