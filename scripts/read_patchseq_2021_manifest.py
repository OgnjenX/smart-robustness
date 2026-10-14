"""Preserve typed official manifest records without recording downloads."""

from __future__ import annotations

import hashlib
import importlib.util
import re
import xml.etree.ElementTree as ET
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import yaml

REG = Path('docs/validation-results/post2008-patchseq-2021-manifest-row-registration-1229.yaml')
NS = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(65536), b''):
            h.update(block)
    return h.hexdigest()


def canonical_integer(cell):
    if cell is None or cell['formula_present']:
        return None
    if cell['storage_type'] in ('s', 'inlineStr'):
        value = cell['literal_text']
    elif cell['storage_type'] in (None, 'n'):
        value = cell['raw_value']
        if type(value) is not str or len(value) > 15:
            return None
    else:
        return None
    return int(value) if type(value) is str and re.fullmatch('[1-9][0-9]*', value) else None


def project(path, definition, fields, guard):
    if [c['text'] for c in definition['cells']] != fields:
        raise ValueError('exact manifest header required')
    columns = {c['cell'][:-1]: c['text'] for c in definition['cells']}
    rows, needed, gaps, previous, header_seen = [], set(), [], 1, False
    with zipfile.ZipFile(path) as archive:
        with archive.open(definition['path']) as handle:
            guard(handle, 'worksheet')
            for _, row in ET.iterparse(handle, events=('end',)):
                if row.tag != NS + 'row':
                    continue
                raw = row.get('r', '')
                if not raw.isascii() or not raw.isdecimal():
                    raise ValueError('explicit numeric row coordinate required')
                number = int(raw)
                if number == 1 and not header_seen:
                    header_seen = True
                    row.clear()
                    continue
                if not header_seen or number <= previous or number > 50001 or len(rows) >= 50000:
                    raise ValueError('bounded ordered source rows required')
                gaps.extend(range(previous + 1, number))
                previous = number
                values = dict.fromkeys(fields)
                for cell in row.findall(NS + 'c'):
                    match = re.fullmatch(r'([A-Z]{1,3})([1-9][0-9]*)', cell.get('r', ''))
                    if not match or int(match[2]) != number or match[1] not in columns:
                        raise ValueError('cell outside exact source header fields')
                    field = columns[match[1]]
                    if values[field] is not None:
                        raise ValueError('duplicate source cell coordinate')
                    kind, raw_value = cell.get('t'), cell.findtext(NS + 'v')
                    if raw_value is not None and len(raw_value) > 4096:
                        raise ValueError('raw cell lexeme bound exceeded')
                    item = {'coordinate': cell.get('r'), 'storage_type': kind, 'raw_value': raw_value,
                            'formula_present': cell.find(NS + 'f') is not None, 'literal_text': None}
                    if kind == 's' and raw_value is not None:
                        if not raw_value.isascii() or not raw_value.isdecimal() or int(raw_value) > 200000:
                            raise ValueError('shared-string index bound exceeded')
                        item['shared_string_index'] = int(raw_value)
                        needed.add(int(raw_value))
                    elif kind == 'inlineStr':
                        text = ''.join(t.text or '' for t in cell.iter(NS + 't'))
                        if len(text) > 4096:
                            raise ValueError('inline text bound exceeded')
                        item['raw_inline_text'] = text
                        if not item['formula_present']:
                            item['literal_text'] = text
                    values[field] = item
                rows.append({'source_row_number': number, 'cells': values})
                row.clear()
        strings = {}
        if needed:
            highest = max(needed)
            with archive.open('xl/sharedStrings.xml') as handle:
                guard(handle, 'sst')
                index = 0
                for _, item in ET.iterparse(handle, events=('end',)):
                    if item.tag == NS + 'si':
                        if index in needed:
                            text = ''.join(t.text or '' for t in item.iter(NS + 't'))
                            if len(text) > 4096:
                                raise ValueError('shared string field bound exceeded')
                            strings[index] = text
                        item.clear()
                        index += 1
                        if index > highest:
                            break
            if set(strings) != needed:
                raise ValueError('source referenced shared string absent')
    for row in rows:
        for cell in row['cells'].values():
            if cell is not None and 'shared_string_index' in cell:
                cell['decoded_shared_string'] = strings[cell['shared_string_index']]
                if not cell['formula_present']:
                    cell['literal_text'] = cell['decoded_shared_string']
        row['canonical_integers'] = {k: canonical_integer(row['cells'][k]) for k in ('file_id', 'file_size', 'cell_specimen_id')}
    return {'rows': rows, 'missing_row_numbers_between_explicit_rows': gaps,
            'explicit_rows': len(rows), 'workbook_modified': False, 'formulas_evaluated': False}


def main():
    if digest(REG) != 'b3250d23e6bb27a098a6e33aa852f77756202596246d9c1f2b295dc7bd546171':
        raise ValueError('sealed typed-manifest registration changed')
    reg = yaml.safe_load(REG.read_text())
    pins = {Path(reg[k]): reg[k + '_sha256'] for k in ('workbook', 'schema', 'header_reader')}
    if any(digest(p) != h for p, h in pins.items()):
        raise ValueError('pinned manifest evidence changed')
    root = Path(reg['output_directory'])
    root.mkdir()
    report = {'status': '2021-manifest-record-read-failure-retained', 'registration_sha256': digest(REG),
              'collector_sha256': digest(__file__), 'source_hashes': {str(p): h for p, h in pins.items()},
              'started_at_utc': datetime.now(UTC).isoformat(), 'network_requests': 0,
              'physiological_arrays_read': False, 'specimen_qualification_performed': False}
    try:
        schema = yaml.safe_load(Path(reg['schema']).read_text())['schema']
        matches = [s for s in schema['sheets'] if s['name'] == reg['sheet_name']]
        if len(matches) != 1:
            raise ValueError('one registered sheet required')
        spec = importlib.util.spec_from_file_location('sealed_manifest_guard', reg['header_reader'])
        reader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reader)
        report.update(project(Path(reg['workbook']), matches[0], reg['fields'], reader.guard))
        if any(digest(p) != h for p, h in pins.items()):
            raise ValueError('source changed during typed manifest read')
        report['status'] = '2021-manifest-typed-records-read-independent-verification-pending'
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile, ET.ParseError) as exc:
        report['error'] = str(exc)
    report['finished_at_utc'] = datetime.now(UTC).isoformat()
    with (root / 'manifest.yaml').open('x') as handle:
        yaml.safe_dump(report, handle, sort_keys=False)
    print(report['status'])
    raise SystemExit(0 if report['status'] == '2021-manifest-typed-records-read-independent-verification-pending' else 1)


if __name__ == '__main__':
    main()
