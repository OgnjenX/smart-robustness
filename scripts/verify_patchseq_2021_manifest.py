"""Independently rebuild typed manifest cells using DOM, not the collector."""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from xml.dom import minidom

import yaml

NS = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
FIELDS = ['file_id', 'file_name', 'file_size', 'sha256_checksum', 'md5_checksum',
          'technique', 'file_type', 'archive', 'archive_uri', 'project', 'cell_specimen_id']
REG = Path('docs/validation-results/post2008-patchseq-2021-manifest-row-registration-1229.yaml')
RESULT = Path('results/patchseq-2021-manifest-records-1230/manifest.yaml')
OUTPUT = Path('results/patchseq-2021-manifest-independent-1231/manifest.yaml')


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def elements(node, name):
    return [c for c in node.childNodes if c.nodeType == c.ELEMENT_NODE
            and c.namespaceURI == NS and c.localName == name]


def text(node):
    return ''.join(c.data if c.nodeType in (c.TEXT_NODE, c.CDATA_SECTION_NODE)
                   else text(c) for c in node.childNodes)


def xml(raw):
    if len(raw) > 32 * 1024 * 1024 or b'<!DOCTYPE' in raw.upper() or b'<!ENTITY' in raw.upper():
        raise ValueError('bounded declaration-free XML required')
    return minidom.parseString(raw)


def integer(cell):
    if cell is None or cell['formula_present']:
        return None
    kind = cell['storage_type']
    value = cell['literal_text'] if kind in ('s', 'inlineStr') else cell['raw_value']
    if kind not in (None, 'n', 's', 'inlineStr') or not isinstance(value, str):
        return None
    if kind in (None, 'n') and len(value) > 15:
        return None
    if not value or value[0] not in '123456789' or any(c not in '0123456789' for c in value):
        return None
    return int(value)


def rebuild(workbook):
    with zipfile.ZipFile(workbook) as archive:
        worksheet = xml(archive.read('xl/worksheets/sheet1.xml'))
        shared = xml(archive.read('xl/sharedStrings.xml'))
    strings = [''.join(text(t) for t in item.getElementsByTagNameNS(NS, 't'))
               for item in elements(shared.documentElement, 'si')]
    containers = elements(worksheet.documentElement, 'sheetData')
    if len(containers) != 1:
        raise ValueError('one sheetData required')
    source_rows = elements(containers[0], 'row')
    if not source_rows or source_rows[0].getAttribute('r') != '1':
        raise ValueError('explicit header row required')
    header = []
    for cell in elements(source_rows[0], 'c'):
        values = elements(cell, 'v')
        if cell.getAttribute('t') != 's' or len(values) != 1 or elements(cell, 'f'):
            raise ValueError('literal shared-string header required')
        header.append((cell.getAttribute('r'), strings[int(text(values[0]))]))
    if header != [(chr(65+i)+'1', name) for i, name in enumerate(FIELDS)]:
        raise ValueError('source header mismatch')
    records, gaps, previous = [], [], 1
    for row in source_rows[1:]:
        number = int(row.getAttribute('r'))
        if not previous < number <= 50001 or len(records) >= 50000:
            raise ValueError('row sequence out of bounds')
        gaps.extend(range(previous+1, number))
        previous = number
        cells = dict.fromkeys(FIELDS)
        for source in elements(row, 'c'):
            coord = source.getAttribute('r')
            match = re.fullmatch(r'([A-K])([1-9][0-9]*)', coord)
            if match is None or int(match[2]) != number:
                raise ValueError('invalid cell coordinate')
            name = FIELDS[ord(match[1])-65]
            if cells[name] is not None:
                raise ValueError('duplicate cell coordinate')
            values = elements(source, 'v')
            if len(values) > 1:
                raise ValueError('duplicate cell value')
            raw = text(values[0]) or None if values else None
            kind = source.getAttribute('t') if source.hasAttribute('t') else None
            cell = {'coordinate': coord, 'storage_type': kind, 'raw_value': raw,
                    'formula_present': bool(elements(source, 'f')), 'literal_text': None}
            if kind == 's' and raw is not None:
                if not raw.isascii() or not raw.isdecimal() or int(raw) > 200000:
                    raise ValueError('invalid shared-string index')
                index = int(raw)
                cell.update(shared_string_index=index, decoded_shared_string=strings[index])
                if not cell['formula_present']:
                    cell['literal_text'] = strings[index]
            elif kind == 'inlineStr':
                literal = ''.join(text(t) for t in source.getElementsByTagNameNS(NS, 't'))
                cell['raw_inline_text'] = literal
                if not cell['formula_present']:
                    cell['literal_text'] = literal
            if any(isinstance(v, str) and len(v) > 4096 for v in cell.values()):
                raise ValueError('cell value exceeds registered bound')
            cells[name] = cell
        records.append({'source_row_number': number, 'cells': cells,
                        'canonical_integers': {k: integer(cells[k])
                                               for k in ('file_id', 'file_size', 'cell_specimen_id')}})
    worksheet.unlink()
    shared.unlink()
    return records, gaps


def check(expected, gaps, result):
    if len(expected) != result['explicit_rows'] or len(expected) != len(result['rows']):
        raise ValueError('row count mismatch')
    if gaps != result['missing_row_numbers_between_explicit_rows']:
        raise ValueError('missing-row evidence mismatch')
    for original, observed in zip(expected, result['rows'], strict=True):
        if json.dumps(original, sort_keys=True) != json.dumps(observed, sort_keys=True):
            raise ValueError(f"typed row mismatch at {original['source_row_number']}")
    if result['workbook_modified'] is not False or result['formulas_evaluated'] is not False:
        raise ValueError('source-preservation flags mismatch')


def main():
    reg_hash = sha(REG)
    if reg_hash != 'b3250d23e6bb27a098a6e33aa852f77756202596246d9c1f2b295dc7bd546171':
        raise ValueError('registration changed')
    reg = yaml.load(REG.read_text(), Loader=yaml.CSafeLoader)
    pins = {Path(reg[k]): reg[k+'_sha256'] for k in ('workbook', 'schema', 'header_reader')}
    result_hash = sha(RESULT)
    if any(sha(p) != h for p, h in pins.items()):
        raise ValueError('source pin mismatch')
    result = yaml.load(RESULT.read_text(), Loader=yaml.CSafeLoader)
    if result['registration_sha256'] != reg_hash or result['source_hashes'] != {str(p): h for p, h in pins.items()}:
        raise ValueError('result provenance mismatch')
    if result['collector_sha256'] != sha('scripts/read_patchseq_2021_manifest.py'):
        raise ValueError('collector changed')
    if result['status'] != '2021-manifest-typed-records-read-independent-verification-pending':
        raise ValueError('successful terminal collector required')
    rows, gaps = rebuild(Path(reg['workbook']))
    check(rows, gaps, result)
    if any(sha(p) != h for p, h in pins.items()) or sha(RESULT) != result_hash:
        raise ValueError('evidence changed during verification')
    report = {'status': 'all-typed-manifest-records-independently-reconstructed',
              'verified_at_utc': datetime.now(UTC).isoformat(), 'verifier_sha256': sha(__file__),
              'result_sha256': result_hash, 'registration_sha256': reg_hash,
              'source_hashes': {str(p): h for p, h in pins.items()}, 'explicit_rows': len(rows),
              'missing_row_numbers_between_explicit_rows': gaps,
              'canonical_integer_unresolved_counts': {k: sum(r['canonical_integers'][k] is None for r in rows)
                                                       for k in ('file_id', 'file_size', 'cell_specimen_id')},
              'technique_counts': dict(Counter(r['cells']['technique']['literal_text']
                                              if r['cells']['technique'] else None for r in rows)),
              'storage_type_counts': dict(Counter(c['storage_type'] for r in rows for c in r['cells'].values() if c)),
              'formula_cells': sum(c['formula_present'] for r in rows for c in r['cells'].values() if c),
              'physiological_arrays_read': False, 'specimen_qualification_performed': False,
              'network_requests': 0, 'workbook_modified': False}
    OUTPUT.parent.mkdir()
    with OUTPUT.open('x') as handle:
        yaml.safe_dump(report, handle, sort_keys=False)
    print(yaml.safe_dump(report, sort_keys=False))


if __name__ == '__main__':
    main()
