"""Typed reference records and exact lookups, with no inferred class membership."""

from __future__ import annotations

import hashlib
import importlib.util
import re
import xml.etree.ElementTree as ET
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import yaml

REG = Path('docs/validation-results/post2008-patchseq-CCN-reference-record-registration-1214.yaml')
NS = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(65536), b''):
            h.update(block)
    return h.hexdigest()


def cell_snapshot(cell):
    raw = cell.findtext(NS + 'v')
    kind = cell.get('t')
    formula = cell.find(NS + 'f') is not None
    if raw is not None and len(raw) > 4096:
        raise ValueError('cell lexeme bound exceeded')
    result = {'coordinate': cell.get('r'), 'storage_type': kind, 'raw_value': raw,
              'formula_present': formula, 'literal_text': None}
    if kind == 's' and raw is not None:
        if not raw.isascii() or not raw.isdecimal() or int(raw) > 10000:
            raise ValueError('shared-string index outside registered bounds')
        result['shared_string_index'] = int(raw)
    elif kind == 'inlineStr':
        text = ''.join(t.text or '' for t in cell.iter(NS + 't'))
        if len(text) > 4096:
            raise ValueError('inline string bound exceeded')
        result['raw_inline_text'] = text
        if not formula:
            result['literal_text'] = text
    return result


def project(workbook, schema, selected, reader):
    outputs, needed = {}, set()
    with zipfile.ZipFile(workbook) as archive:
        for name, fields in selected.items():
            matches = [s for s in schema['sheets'] if s['name'] == name]
            if len(matches) != 1 or [c['text'] for c in matches[0]['cells']] != fields:
                raise ValueError('exact registered sheet header required')
            definition = matches[0]
            columns = {c['cell'][:-1]: c['text'] for c in definition['cells']}
            rows, previous, gaps, header_seen = [], 1, [], False
            with archive.open(definition['path']) as handle:
                reader.guard(handle, 'worksheet')
                for _, element in ET.iterparse(handle, events=('end',)):
                    if element.tag != NS + 'row':
                        continue
                    raw_number = element.get('r', '')
                    if not raw_number.isascii() or not raw_number.isdecimal():
                        raise ValueError('explicit numeric row coordinate required')
                    number = int(raw_number)
                    if number == 1 and not header_seen:
                        header_seen = True
                        element.clear()
                        continue
                    if not header_seen or number <= previous or number > 10000 or len(rows) >= 5000:
                        raise ValueError('source row order or extent differs')
                    gaps.extend(range(previous + 1, number))
                    previous = number
                    values = dict.fromkeys(fields)
                    for cell in element.findall(NS + 'c'):
                        match = re.fullmatch(r'([A-Z]{1,3})([1-9][0-9]*)', cell.get('r', ''))
                        if not match or int(match[2]) != number or match[1] not in columns:
                            raise ValueError('cell outside registered field coordinates')
                        field = columns[match[1]]
                        if values[field] is not None:
                            raise ValueError('duplicate source cell coordinate')
                        values[field] = cell_snapshot(cell)
                        if 'shared_string_index' in values[field]:
                            needed.add(values[field]['shared_string_index'])
                    rows.append({'source_row_number': number, 'cells': values})
                    element.clear()
            outputs[name] = {'rows': rows, 'missing_row_numbers_between_explicit_rows': gaps}
        strings = reader.shared_headers(archive, needed)
    for sheet in outputs.values():
        for row in sheet['rows']:
            for cell in row['cells'].values():
                if cell is not None and 'shared_string_index' in cell:
                    text = strings[cell['shared_string_index']]
                    if len(text) > 4096:
                        raise ValueError('reference string bound exceeded')
                    cell['decoded_shared_string'] = text
                    if not cell['formula_present']:
                        cell['literal_text'] = text
    return outputs


def literal(row, field):
    cell = row['cells'][field]
    return None if cell is None else cell['literal_text']


def lookup(records, july_rows):
    leaves = records['cell set nomenclature']['rows']
    pairs = sorted({(r['raw_fields']['T-type Accession ID'], r['raw_fields']['T-type Label']) for r in july_rows})
    outputs = []
    for accession, label in pairs:
        by_id = [r for r in leaves if accession and literal(r, 'cell_set_ accession') == accession]
        by_alias = [r for r in leaves if label and literal(r, 'cell_set_preferred_alias') == label]
        outputs.append({'raw_July_accession': accession, 'raw_July_label': label,
                        'exact_accession_candidates': [r['source_row_number'] for r in by_id],
                        'exact_preferred_alias_candidates': [r['source_row_number'] for r in by_alias],
                        'accession_candidate_alias_agreement': [literal(r, 'cell_set_preferred_alias') == label for r in by_id],
                        'class_membership_established': False})
    return outputs


def main():
    if digest(REG) != 'c649794fd689d78d3163fadd82d62563ae80f88d18f42e2469d6935cb0a0a19a':
        raise ValueError('sealed reference-record registration changed')
    reg = yaml.safe_load(REG.read_text())
    pins = {Path(reg[k]): reg[k + '_sha256'] for k in ('workbook', 'schema', 'July_comparison', 'header_reader')}
    if any(digest(p) != h for p, h in pins.items()):
        raise ValueError('pinned reference evidence changed')
    root = Path(reg['output_directory'])
    root.mkdir()
    report = {'status': 'CCN-reference-read-failure-retained', 'registration_sha256': digest(REG),
              'collector_sha256': digest(__file__), 'source_hashes': {str(p): h for p, h in pins.items()},
              'started_at_utc': datetime.now(UTC).isoformat(), 'network_requests': 0,
              'workbook_modified': False, 'formulas_evaluated': False, 'external_links_followed': False,
              'physiological_arrays_read': False, 'biological_eligibility_established': False}
    try:
        spec = importlib.util.spec_from_file_location('pinned_reference_header_reader', reg['header_reader'])
        reader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reader)
        schema = yaml.safe_load(Path(reg['schema']).read_text())['schema']
        records = project(Path(reg['workbook']), schema, reg['sheets'], reader)
        july = yaml.load(Path(reg['July_comparison']).read_text(), Loader=yaml.CSafeLoader)['July_rows']
        report.update(reference_sheets=records, accession_lookup=lookup(records, july))
        if any(digest(p) != h for p, h in pins.items()):
            raise ValueError('source changed during reference read')
        report['status'] = 'CCN-typed-reference-records-read-independent-verification-pending'
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile, ET.ParseError) as exc:
        report['error'] = str(exc)
    report['finished_at_utc'] = datetime.now(UTC).isoformat()
    with (root / 'manifest.yaml').open('x') as handle:
        yaml.safe_dump(report, handle, sort_keys=False)
    print(report['status'])
    raise SystemExit(0 if report['status'] == 'CCN-typed-reference-records-read-independent-verification-pending' else 1)


if __name__ == '__main__':
    main()
