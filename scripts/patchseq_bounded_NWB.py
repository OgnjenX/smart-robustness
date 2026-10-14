"""Read-only bounded HDF5 backing I/O and metadata whitelist enforcement."""

from __future__ import annotations

import io
from urllib.parse import unquote, urlsplit


def permitted_url(url, policy):
    if type(url) is not str:
        return False
    parsed = urlsplit(url)
    try:
        port = parsed.port
    except ValueError:
        return False
    return (parsed.scheme == policy['allowed_scheme'] and parsed.hostname in policy['allowed_hosts']
            and parsed.username is None and parsed.password is None and port in (None, 443)
            and not parsed.fragment and parsed.path.startswith(policy['required_path_prefix'])
            and not any(p in ('.', '..') for p in unquote(parsed.path).split('/')))


def url_candidates(raw, policy):
    containers = [('top_level', raw)]
    if type(raw.get('metadata')) is dict:
        containers.append(('literal_metadata_object', raw['metadata']))
    candidates = []
    for label, obj in containers:
        value = obj.get('contentUrl')
        if type(value) is list:
            candidates.extend({'container': label, 'value': v, 'permitted': permitted_url(v, policy)} for v in value)
        elif 'contentUrl' in obj:
            candidates.append({'container': label, 'value': value, 'permitted': False})
    return candidates


def select_url(raw, policy):
    candidates = url_candidates(raw, policy)
    urls = {r['value'] for r in candidates if r['permitted']}
    if len(urls) != 1:
        raise ValueError('one unique exact permitted object locator required')
    return urls.pop(), candidates


class BoundedReader(io.RawIOBase):
    def __init__(self, size, limits, fetch, check_deadline=lambda: None):
        super().__init__()
        self.size, self.limits, self.fetch = size, limits, fetch
        self.position, self.downloaded, self.requests = 0, 0, 0
        self.cache = {}
        self.check_deadline = check_deadline

    def readable(self):
        return True

    def seekable(self):
        return True

    def writable(self):
        return False

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset + (0 if whence == 0 else self.position if whence == 1 else self.size if whence == 2 else -self.size-abs(offset)-1)
        if position < 0:
            raise ValueError('negative or unsupported seek')
        self.position = position
        return position

    def read(self, size=-1):
        self._checkClosed()
        self.check_deadline()
        remaining = max(0, self.size-self.position)
        length = remaining if size < 0 else min(size, remaining)
        if length > self.limits['maximum_unique_downloaded_bytes']:
            raise ValueError('unbounded read rejected')
        chunks = []
        while length:
            block = self.position // self.limits['cache_block_bytes']
            start = block*self.limits['cache_block_bytes']
            end = min(self.size, start+self.limits['cache_block_bytes'])
            if block not in self.cache:
                if (self.requests >= self.limits['maximum_range_requests']
                    or self.downloaded+end-start > self.limits['maximum_unique_downloaded_bytes']):
                    raise ValueError('registered range budget exhausted')
                self.requests += 1
                self.downloaded += end-start
                data = self.fetch(start, end-1)
                if len(data) != end-start:
                    raise ValueError('range response length mismatch')
                self.cache[block] = data
            offset = self.position-start
            count = min(length, end-self.position)
            chunks.append(self.cache[block][offset:offset+count])
            self.position += count
            length -= count
        return b''.join(chunks)

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)

    def write(self, data):
        raise io.UnsupportedOperation('read-only backing file')

    def truncate(self, size=None):
        raise io.UnsupportedOperation('read-only backing file')


def scalar(value, maximum):
    if hasattr(value, 'shape'):
        if value.shape not in ((), (1,)):
            raise ValueError('scalar or one-element metadata required')
        value = value.item()
    if isinstance(value, bytes):
        if len(value) > maximum:
            raise ValueError('metadata byte bound exceeded')
        return {'raw_type': 'bytes', 'raw_hex': value.hex(), 'decoded_text': value.decode('utf-8')}
    if type(value) is str:
        if len(value.encode('utf-8')) > maximum:
            raise ValueError('metadata text bound exceeded')
        return {'raw_type': 'str', 'value': value}
    if type(value) in (int, bool, float):
        import math
        if type(value) is float and not math.isfinite(value):
            raise ValueError('nonfinite metadata')
        return {'raw_type': type(value).__name__, 'value': value}
    raise TypeError('unsupported metadata scalar type')


def inspect_metadata(root, policy, h5py, report=None):
    if report is None:
        report = {'root_attributes': {}, 'datasets': {}, 'logical_dataset_reads': []}
    maximum = policy['maximum_decoded_value_bytes']
    for name in policy['root_attributes']:
        entry = {'present': name in root.attrs}
        report['root_attributes'][name] = entry
        if entry['present']:
            attr = root.attrs.get_id(name)
            if attr.shape not in ((), (1,)) or attr.get_storage_size() > maximum:
                raise ValueError('root metadata attribute exceeds bounds')
            entry['scalar'] = scalar(root.attrs[name], maximum)
        report['root_attributes'][name] = entry
    for path in policy['datasets']:
        node = root
        present = True
        for part in path.strip('/').split('/'):
            if not isinstance(node, h5py.Group):
                raise TypeError('metadata path parent must be a group')
            link = node.get(part, getlink=True)
            if link is None:
                present = False
                break
            if not isinstance(link, h5py.HardLink):
                raise TypeError('external and soft links rejected before traversal')
            node = node[part]
        entry = {'present': present}
        if present:
            if not isinstance(node, h5py.Dataset):
                raise ValueError('whitelisted metadata path is not a dataset')
            if (node.shape not in ((), (1,)) or node.is_virtual or node.external
                or node.compression is not None or node.id.get_storage_size() > maximum):
                raise ValueError('unsupported or unbounded metadata dataset')
            report['logical_dataset_reads'].append(path)
            entry.update(shape=list(node.shape), dtype=str(node.dtype), scalar=scalar(node[()], maximum))
        report['datasets'][path] = entry
    return report
