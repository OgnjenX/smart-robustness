"""Exact source leaf memberships through explicit reference-tree edges."""

from __future__ import annotations

import hashlib
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import yaml

REG = Path('docs/validation-results/post2008-patchseq-source-group-membership-registration-1223.yaml')
SUBGROUPS = ('Lamp5', 'Sncg', 'Vip', 'Sst', 'Pvalb')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def literal(node, key):
    value = node['attributes'][key]
    if value['present'] is not True or value['JSON_type'] != 'str' or type(value['raw_value']) is not str:
        return None
    return value['raw_value']


def trace(nodes, alignments, taxonomy, groups):
    paths, ids, parents = {}, {}, {}
    for node in nodes:
        path = node['path']
        accession = literal(node, 'cell_set_accession')
        if (path in paths or not accession or accession in ids or literal(node, 'taxonomy_id') != taxonomy
            or type(node['leaf']) is not bool or type(node['children']) is not list
            or bool(node['children']) == node['leaf']):
            raise ValueError('unique typed taxonomy nodes required')
        paths[path], ids[accession] = node, node
        for child in node['children']:
            if type(child) is not str or child in parents:
                raise ValueError('unique literal child-parent references required')
            parents[child] = path
    if 'root' not in paths or 'root' in parents or set(parents) != set(paths) - {'root'}:
        raise ValueError('one connected root and valid child references required')
    ancestors = {}
    for path in paths:
        chain, current = set(), path
        while True:
            if current in chain:
                raise ValueError('cycle in source reference tree')
            chain.add(current)
            if current == 'root':
                break
            current = parents[current]
        ancestors[path] = chain
    group_paths = {}
    for name, accession in groups.items():
        node = ids.get(accession)
        if node is None or node['leaf'] or literal(node, 'cell_set_preferred_alias') != name:
            raise ValueError('exact named source group node required')
        group_paths[name] = node['path']
    outputs = []
    for alignment in alignments:
        row = {'legacy_accession': alignment['legacy_accession'], 'legacy_label': alignment['legacy_label'],
               'reference_accession': alignment['reference_accession'], 'source_leaf_identity_agrees': False,
               'source_groups': [], 'named_inhibitory_subgroups': [], 'exclusive_named_inhibitory_subgroup': None,
               'specimen_qualification_established': False, 'historical_identifier_equivalence_confirmed': False}
        if alignment['exact_name_alignment_supported'] is True and alignment['reference_taxonomy'] == taxonomy:
            node = ids.get(alignment['reference_accession'])
            if node is not None and node['leaf'] and literal(node, 'cell_set_preferred_alias') == alignment['legacy_label']:
                row['source_leaf_identity_agrees'] = True
                row['source_groups'] = [name for name, path in group_paths.items() if path in ancestors[node['path']]]
                row['named_inhibitory_subgroups'] = [name for name in SUBGROUPS if name in row['source_groups']]
                if 'GABAergic' in row['source_groups'] and len(row['named_inhibitory_subgroups']) == 1:
                    row['exclusive_named_inhibitory_subgroup'] = row['named_inhibitory_subgroups'][0]
        outputs.append(row)
    return {'memberships': outputs, 'source_group_paths': group_paths,
            'summary': {'alignment_pairs_including_empty': len(outputs),
                        'source_leaf_identity_agreements': sum(r['source_leaf_identity_agrees'] for r in outputs),
                        'group_type_counts': {name: sum(name in r['source_groups'] for r in outputs) for name in groups},
                        'exclusive_inhibitory_subgroup_counts': dict(sorted(Counter(r['exclusive_named_inhibitory_subgroup'] for r in outputs if r['exclusive_named_inhibitory_subgroup'] is not None).items())),
                        'GABAergic_types_outside_named_subgroups': sum('GABAergic' in r['source_groups'] and not r['named_inhibitory_subgroups'] for r in outputs),
                        'types_with_overlapping_named_subgroups': sum(len(r['named_inhibitory_subgroups']) > 1 for r in outputs)}}


def main():
    if digest(REG) != '9d1e2fba67df4d7bb2861614afc6efa67e4d7f89bf8b12414708f979f4c15eed':
        raise ValueError('sealed source-membership registration changed')
    reg = yaml.safe_load(REG.read_text())
    pins = {Path(reg[k]): reg[k + '_sha256'] for k in ('tree', 'crosswalk')}
    if any(digest(p) != h for p, h in pins.items()):
        raise ValueError('pinned membership sources changed')
    root = Path(reg['output_directory'])
    root.mkdir()
    report = {'status': 'source-membership-failure-retained', 'registration_sha256': digest(REG),
              'collector_sha256': digest(__file__), 'source_hashes': {str(p): h for p, h in pins.items()},
              'started_at_utc': datetime.now(UTC).isoformat(), 'network_requests': 0,
              'physiological_arrays_read': False, 'biological_eligibility_established': False}
    try:
        tree = yaml.safe_load(Path(reg['tree']).read_text())
        crosswalk = yaml.safe_load(Path(reg['crosswalk']).read_text())
        report.update(trace(tree['nodes'], crosswalk['alignments'], reg['taxonomy_id'], reg['source_groups']))
        if any(digest(p) != h for p, h in pins.items()):
            raise ValueError('sources changed during membership tracing')
        report['status'] = 'source-type-memberships-traced-independent-verification-pending'
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report['error'] = str(exc)
    report['finished_at_utc'] = datetime.now(UTC).isoformat()
    with (root / 'manifest.yaml').open('x') as handle:
        yaml.safe_dump(report, handle, sort_keys=False)
    print(report['status'])
    raise SystemExit(0 if report['status'] == 'source-type-memberships-traced-independent-verification-pending' else 1)


if __name__ == '__main__':
    main()
