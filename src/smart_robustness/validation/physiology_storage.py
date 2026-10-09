"""Lossless content-addressed arrays, immutable receipts and conservative disk guards."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            result.update(block)
    return result.hexdigest()


def array_identity(values):
    array = np.asarray(values)
    if array.dtype.kind not in "biufc":
        raise ValueError("only non-object numeric arrays are supported")
    array = np.ascontiguousarray(array) if array.ndim else array.copy()
    metadata = {"dtype": array.dtype.str, "shape": list(array.shape)}
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode() + b"\0")
    raw = memoryview(array.reshape(-1).view(np.uint8))
    for start in range(0, len(raw), 1048576):
        digest.update(raw[start : start + 1048576])
    return array, metadata, digest.hexdigest()


def require_space(directory, additional_bytes, reserve_bytes):
    if additional_bytes < 0 or reserve_bytes < 0:
        raise ValueError("negative storage budget")
    free = shutil.disk_usage(directory).free
    if free < additional_bytes + reserve_bytes:
        raise OSError(
            "insufficient space; preserve checkpoints and resume after capacity increases"
        )


def publish(temp_path, destination):
    # Same-filesystem hard-link publication never replaces an existing result.
    os.link(temp_path, destination)
    descriptor = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class ArrayStore:
    """Single writer per directory; every duplicate is independently bit-hashed."""

    def __init__(self, directory, *, reserve_bytes=100 * 2**30):
        self.directory = Path(directory)
        self.reserve_bytes = reserve_bytes
        self.lock = None

    def __enter__(self):
        if self.lock is not None:
            raise ValueError("store context cannot be entered twice")
        self.directory.mkdir(exist_ok=True)
        self.lock = (self.directory / "writer.lock").open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.lock.close()
            self.lock = None
            raise
        return self

    def __exit__(self, *_):
        self.lock.close()
        self.lock = None

    def path(self, key):
        if not isinstance(key, str) or re.fullmatch(r"[0-9a-f]{64}", key) is None:
            raise ValueError("invalid content key")
        return self.directory / (key + ".npz")

    def get(self, receipt):
        path = self.path(receipt["content_sha256"])
        if file_digest(path) != receipt["file_sha256"]:
            raise ValueError("stored array file changed")
        with np.load(path, allow_pickle=False) as archive:
            if archive.files != ["values"]:
                raise ValueError("unexpected array archive fields")
            array = archive["values"]
        _, metadata, key = array_identity(array)
        if key != receipt["content_sha256"] or metadata != receipt["metadata"]:
            raise ValueError("stored array content differs from receipt")
        return array

    def put(self, values):
        if self.lock is None:
            raise ValueError("writes require exclusive store context")
        array, metadata, key = array_identity(values)
        destination = self.path(key)
        if not destination.exists():
            # Includes conservative deflate overhead and NPY/ZIP headers, no assumed compression.
            require_space(
                self.directory, array.nbytes + array.nbytes // 100 + 65536, self.reserve_bytes
            )
            descriptor, name = tempfile.mkstemp(prefix="pending-array-", dir=self.directory)
            temporary = Path(name)
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    np.savez_compressed(handle, values=array)
                    handle.flush()
                    os.fsync(handle.fileno())
                publish(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
        receipt = {
            "content_sha256": key,
            "file_sha256": file_digest(destination),
            "metadata": metadata,
            "uncompressed_bytes": array.nbytes,
            "stored_bytes": destination.stat().st_size,
        }
        # Verify new and existing blobs, never trust existence as proof of valid content.
        restored = self.get(receipt)
        original_bytes = memoryview(array.reshape(-1).view(np.uint8))
        restored_bytes = memoryview(restored.reshape(-1).view(np.uint8))
        for start in range(0, len(original_bytes), 1048576):
            if original_bytes[start : start + 1048576] != restored_bytes[start : start + 1048576]:
                raise ValueError("bitwise duplicate verification failed")
        return receipt

    def checkpoint(self, name, payload):
        if self.lock is None or re.fullmatch(r"[a-z0-9-]+", name) is None:
            raise ValueError("checkpoint requires store context and safe name")
        encoded = json.dumps(payload, sort_keys=True, allow_nan=False).encode()
        destination = self.directory / (name + ".json")
        if destination.exists():
            if destination.read_bytes() != encoded:
                raise ValueError("immutable checkpoint differs")
            return destination
        require_space(self.directory, len(encoded) + 65536, self.reserve_bytes)
        descriptor, name = tempfile.mkstemp(prefix="pending-receipt-", dir=self.directory)
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            publish(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination
