"""Conditional manifest acquisition and bounded first-row-only OOXML inspection."""

from __future__ import annotations

import hashlib
import posixpath
import re
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import yaml

REG = Path('docs/validation-results/post2008-patchseq-2021-workbook-acquisition-registration-1201.yaml')
NS = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
REL = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(65536), b''):
            h.update(block)
    return h.hexdigest()


def guard(handle, root):
    prefix = handle.read(65536)
    if (b'<!DOCTYPE' in prefix or b'<!ENTITY' in prefix
        or not re.search(rb'<(?:[A-Za-z_][\w.-]*:)?' + root.encode() + rb'(?:\s|>)', prefix)):
        raise ValueError('bounded XML root without DTD/entities required')
    handle.seek(0)


def header_cells(archive, path):
    with archive.open(path) as handle:
        guard(handle, 'worksheet')
        dimension = None
        for _, element in ET.iterparse(handle, events=('end',)):
            if element.tag == NS + 'dimension':
                dimension = element.get('ref')
            if element.tag == NS + 'row':
                if element.get('r') != '1':
                    raise ValueError('first worksheet row must be registered header row 1')
                cells = []
                for cell in element.findall(NS + 'c'):
                    if cell.find(NS + 'f') is not None:
                        raise ValueError('formula header unsupported; never evaluate')
                    ref = cell.get('r', '')
                    if not re.fullmatch('[A-Z]{1,3}1', ref):
                        raise ValueError('explicit first-row header cell coordinates required')
                    kind = cell.get('t')
                    if kind == 's':
                        raw = cell.findtext(NS + 'v', '')
                        if not raw.isascii() or not raw.isdecimal() or int(raw) > 10000:
                            raise ValueError('bounded shared-string header index required')
                        cells.append({'cell': ref, 'shared_string_index': int(raw)})
                    elif kind == 'inlineStr':
                        cells.append({'cell': ref, 'text': ''.join(t.text or '' for t in cell.iter(NS + 't'))})
                    else:
                        raise ValueError('literal string header required')
                if not 0 < len(cells) <= 256 or len({c['cell'] for c in cells}) != len(cells):
                    raise ValueError('bounded unique nonempty header required')
                return dimension, cells  # Do not iterate or interpret any later data row.
    raise ValueError('header row missing')


def shared_headers(archive, needed):
    if not needed:
        return {}
    values = {}
    with archive.open('xl/sharedStrings.xml') as handle:
        guard(handle, 'sst')
        index = 0
        for _, element in ET.iterparse(handle, events=('end',)):
            if element.tag == NS + 'si':
                if index in needed:
                    values[index] = ''.join(t.text or '' for t in element.iter(NS + 't'))
                element.clear()
                index += 1
                if index > max(needed):
                    break
    if set(values) != needed:
        raise ValueError('referenced header shared string absent')
    return values


def inspect(path):
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        names = [m.filename for m in members]
        if (len(members) > 100 or len(set(names)) != len(names)
            or sum(m.file_size for m in members) > 134217728
            or any(n.startswith('/') or '\\' in n or '..' in n.split('/') for n in names)
            or any(m.flag_bits & 1 for m in members)
            or any('vbaProject' in n or n.startswith('xl/externalLinks/') for n in names)):
            raise ValueError('bounded safe nonmacro workbook required')
        def small_xml(name, root):
            if archive.getinfo(name).file_size > 1048576:
                raise ValueError('workbook structural XML bound exceeded')
            with archive.open(name) as handle:
                guard(handle, root)
                return ET.parse(handle).getroot()
        book = small_xml('xl/workbook.xml', 'workbook')
        relations = {e.get('Id'): e for e in small_xml('xl/_rels/workbook.xml.rels', 'Relationships')}
        sheets, needed = [], set()
        definitions = book.findall(NS + 'sheets/' + NS + 'sheet')
        if not 0 < len(definitions) <= 10:
            raise ValueError('bounded worksheet count required')
        for sheet in definitions:
            relation = relations[sheet.get(REL)]
            if relation.get('TargetMode') == 'External' or not relation.get('Type', '').endswith('/worksheet'):
                raise ValueError('local worksheet relationship required')
            target = relation.get('Target', '')
            if '\\' in target or '..' in target.split('/'):
                raise ValueError('unsafe worksheet target')
            target = target.lstrip('/') if target.startswith('/') else posixpath.join('xl', target)
            if target not in names:
                raise ValueError('worksheet target absent')
            dimension, cells = header_cells(archive, target)
            needed.update(c['shared_string_index'] for c in cells if 'shared_string_index' in c)
            sheets.append({'name': sheet.get('name'), 'path': target, 'declared_dimension': dimension, 'cells': cells})
        if not 0 < len(sheets) <= 10:
            raise ValueError('bounded worksheet count required')
        strings = shared_headers(archive, needed)
        for sheet in sheets:
            for cell in sheet['cells']:
                if 'shared_string_index' in cell:
                    cell['text'] = strings[cell['shared_string_index']]
            labels = [c['text'] for c in sheet['cells']]
            if any(not t or len(t) > 4096 for t in labels) or len(set(labels)) != len(labels):
                raise ValueError('unique bounded header labels required')
        return {'ZIP_members': [{'name': m.filename, 'bytes': m.file_size} for m in members],
                'sheets': sheets, 'data_rows_interpreted': False, 'formulas_evaluated': False,
                'external_links_followed': False, 'workbook_modified': False}


def main():
    if digest(REG) != '09888cc4044d0bc115c7109dfcc27ccf3d8eb2dbdd91de90bdc5c8bed58b44bc':
        raise ValueError('sealed workbook acquisition registration changed')
    reg = yaml.safe_load(REG.read_text())
    root = Path(reg['output_directory'])
    root.mkdir()
    body, headers = root / 'manifest.xlsx', root / 'response-headers.txt'
    report = {'status': 'workbook-schema-acquisition-failure-retained', 'registration_sha256': digest(REG),
              'collector_sha256': digest(__file__), 'started_at_utc': datetime.now(UTC).isoformat(),
              'source_url': reg['source_url'], 'physiological_arrays_read': False, 'data_rows_interpreted': False}
    try:
        result = subprocess.run(['curl', '--fail', '--silent', '--show-error', '--proto', '=https',
                                 '--max-time', '120', '--max-filesize', '2889514', '--header', 'If-Match: ' + reg['expected_etag'],
                                 '--dump-header', str(headers), '--output', str(body), '--write-out', '%{http_code}', reg['source_url']],
                                capture_output=True, text=True, timeout=130, check=False)
        report.update(client_exit_code=result.returncode, http_status=result.stdout, client_error=result.stderr[:4096])
        if result.returncode != 0 or result.stdout != '200' or body.stat().st_size != reg['expected_bytes']:
            raise ValueError('conditional request identity differs')
        if headers.stat().st_size > 65536:
            raise ValueError('HTTP header byte bound exceeded')
        fields = {}
        for line in headers.read_text().splitlines():
            if ':' in line:
                key, value = line.split(':', 1)
                fields.setdefault(key.lower(), []).append(value.strip())
        if (fields.get('etag') != [reg['expected_etag']] or fields.get('x-amz-version-id') != [reg['expected_version_id']]
            or fields.get('content-length') != [str(reg['expected_bytes'])] or fields.get('content-encoding', ['identity']) != ['identity']):
            raise ValueError('conditional HTTP object headers differ')
        report.update(body_sha256=digest(body), bytes=body.stat().st_size, schema=inspect(body),
                      status='workbook-acquired-header-only-independent-verification-pending')
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile, ET.ParseError, subprocess.TimeoutExpired) as exc:
        report['error'] = str(exc)
    report['finished_at_utc'] = datetime.now(UTC).isoformat()
    with (root / 'manifest.yaml').open('x') as handle:
        yaml.safe_dump(report, handle, sort_keys=False)
    print(report['status'])
    raise SystemExit(0 if report['status'] == 'workbook-acquired-header-only-independent-verification-pending' else 1)


if __name__ == '__main__':
    main()
