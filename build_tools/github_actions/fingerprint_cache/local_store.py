# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A local, directory-backed object store for local builds.

Phase 0 of the roadmap runs entirely on one machine with no CI and no network:
a local build populates a cache, a later local build reuses from it. This module
is the local fallback for S3 "immutable artifact storage" — it copies artifact
files into a content directory, records real SHA-256 digests, and satisfies the
``ObjectStore`` protocol used by ``restore`` (``exists`` / ``fetch_digest``).

Everything here is deliberately simple and provisional: the goal is to flush
design issues cheaply before any contract is frozen.
"""


from dataclasses import dataclass
import hashlib
from pathlib import Path
import shutil

from .manifest import OutputObject

_URI_SCHEME = "local://"


def sha256_file(path: Path, _chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(_chunk), b""):
            h.update(block)
    return h.hexdigest()


@dataclass
class LocalObjectStore:
    """Stores artifact objects under ``root`` keyed by their filename.

    URIs are of the form ``local://<key>`` and resolve to ``root/<key>``. This
    mirrors how the production store will hand back an S3 pointer, so the rest
    of the pipeline (manifest, restore) is unchanged between local and prod.
    """

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    # -- path helpers ------------------------------------------------------
    def uri_for(self, key: str) -> str:
        return f"{_URI_SCHEME}{key}"

    def path_for(self, uri: str) -> Path:
        key = uri[len(_URI_SCHEME) :] if uri.startswith(_URI_SCHEME) else uri
        return self.root / key

    # -- writes ------------------------------------------------------------
    def upload_file(self, src: Path, key: str | None = None) -> OutputObject:
        """Copy a built artifact file into the store; return its OutputObject.

        The returned object carries the real digest and size, so a later
        ``restore`` can verify integrity exactly as it will against S3.
        """
        src = Path(src)
        key = key or src.name
        dest = self.root / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        return OutputObject(
            key=key,
            uri=self.uri_for(key),
            digest=sha256_file(dest),
            size_bytes=dest.stat().st_size,
        )

    # -- ObjectStore protocol (reads) -------------------------------------
    def exists(self, uri: str) -> bool:
        return self.path_for(uri).exists()

    def fetch_digest(self, uri: str) -> str | None:
        path = self.path_for(uri)
        if not path.exists():
            return None
        return sha256_file(path)

    def download_file(self, uri: str, dest: Path) -> Path:
        """Copy an object out of the store (the 'download' in flowchart 2)."""
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.path_for(uri), dest)
        return dest
