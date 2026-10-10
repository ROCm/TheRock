# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Reads and rewrites the kpack archives that hold kpack-split device code.

A kpack-split artifact moves every GPU code object out of its host binaries
into one archive per artifact and GPU target: .kpack/rand_lib_gfx942.kpack
holds everything rocRAND and hipRAND build for gfx942. The host binary keeps a
.rocm_kpack_ref marker, and the runtime finds its code objects in the archive
under "<stage prefix>/<binary>#<n>", one entry per translation unit.

This mirrors what rocm_kpack.kpack.PackedKernelArchive (rocm-systems
shared/kpack) writes, since that package is not installed where these scripts
run:

  header  b"KPAK", uint32 format version, uint64 TOC offset
  blob    "zstd-per-kernel": uint32 count, then a uint32 size and a zstd
          frame per kernel. "none": the code objects themselves, located by
          the TOC's "blobs" list of absolute offsets.
  TOC     MessagePack {"toc": {key: {arch: {"ordinal": n, ...}}}, ...}
"""

import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Callable, Iterator

MAGIC = b"KPAK"
FORMAT_VERSION = 1
SCHEME_ZSTD = "zstd-per-kernel"
SCHEME_NONE = "none"

_HEADER = struct.Struct("<4sIQ")
_TOC_OFFSET = struct.Struct("<Q")
_UINT32 = struct.Struct("<I")
_BLOB_ALIGNMENT = 64
# TOC keys that describe the blob layout, rewritten along with the blob.
_LAYOUT_KEYS = ("toc", "compression_scheme", "zstd_offset", "zstd_size", "blobs")


def _get_msgpack():
    """Lazy import msgpack with helpful error message."""
    try:
        import msgpack

        return msgpack
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "msgpack is required to read kpack archives. "
            "Install it with: pip install msgpack"
        )


@dataclass
class KpackArchive:
    """One .kpack file: its TOC, and where each kernel's stored bytes sit."""

    path: Path
    toc_data: dict[str, Any]
    # (file offset, size) per ordinal: a zstd frame under "zstd-per-kernel",
    # the code object itself under "none".
    spans: list[tuple[int, int]]

    @property
    def scheme(self) -> str:
        return self.toc_data.get("compression_scheme", SCHEME_NONE)

    def entries(self) -> Iterator[tuple[str, str, dict[str, Any]]]:
        """Yields (key, arch, entry) for every kernel in the archive."""
        for key, arches in self.toc_data["toc"].items():
            for arch, entry in arches.items():
                yield key, arch, entry

    def read_stored(self, f: BinaryIO, ordinal: int) -> bytes:
        offset, size = self.spans[ordinal]
        f.seek(offset)
        return f.read(size)

    def code_object(self, f: BinaryIO, entry: dict[str, Any]) -> bytes:
        """Returns the code object behind entry, decompressed."""
        data = self.read_stored(f, entry["ordinal"])
        if self.scheme == SCHEME_ZSTD:
            import pyzstd

            return pyzstd.decompress(data)
        return data


def read_kpack(path: Path) -> KpackArchive:
    msgpack = _get_msgpack()
    with open(path, "rb") as f:
        magic, version, toc_offset = _HEADER.unpack(f.read(_HEADER.size))
        if magic != MAGIC:
            raise ValueError(f"{path} is not a kpack archive (magic {magic!r})")
        if version != FORMAT_VERSION:
            raise ValueError(f"{path}: unsupported kpack format version {version}")
        f.seek(toc_offset)
        toc_data = msgpack.unpack(f, raw=False)

        scheme = toc_data.get("compression_scheme", SCHEME_NONE)
        if scheme == SCHEME_ZSTD:
            offset = toc_data["zstd_offset"]
            f.seek(offset)
            (count,) = _UINT32.unpack(f.read(_UINT32.size))
            offset += _UINT32.size
            spans = []
            for _ in range(count):
                f.seek(offset)
                (size,) = _UINT32.unpack(f.read(_UINT32.size))
                spans.append((offset + _UINT32.size, size))
                offset += _UINT32.size + size
        elif scheme == SCHEME_NONE:
            spans = [(blob["offset"], blob["size"]) for blob in toc_data["blobs"]]
        else:
            raise ValueError(f"{path}: unknown kpack compression scheme {scheme!r}")
    return KpackArchive(path=Path(path), toc_data=toc_data, spans=spans)


def write_kpack(
    output: Path,
    template: KpackArchive,
    entries: list[tuple[str, str, dict[str, Any], KpackArchive]],
) -> None:
    """Writes entries, (key, arch, entry, source archive), as one archive.

    Stored bytes are copied as they are, so every source has to share the
    template's compression scheme. The header fields come from template.
    Written next to output and renamed over it, so output may be one of the
    sources.
    """
    msgpack = _get_msgpack()
    scheme = template.scheme
    for _, _, _, source in entries:
        if source.scheme != scheme:
            raise ValueError(
                f"cannot combine {source.path} ({source.scheme}) with "
                f"{template.path} ({scheme}): kpack compression schemes differ"
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.name + ".tmp")
    sources: dict[Path, BinaryIO] = {}
    try:
        with open(temp, "wb") as out:
            out.write(_HEADER.pack(MAGIC, FORMAT_VERSION, 0))
            out.write(b"\0" * (-out.tell() % _BLOB_ALIGNMENT))
            blob_start = out.tell()
            if scheme == SCHEME_ZSTD:
                out.write(_UINT32.pack(len(entries)))

            toc: dict[str, dict[str, dict[str, Any]]] = {}
            blobs = []
            for ordinal, (key, arch, entry, source) in enumerate(entries):
                if source.path not in sources:
                    sources[source.path] = open(source.path, "rb")
                data = source.read_stored(sources[source.path], entry["ordinal"])
                if scheme == SCHEME_ZSTD:
                    out.write(_UINT32.pack(len(data)))
                else:
                    blobs.append({"offset": out.tell(), "size": len(data)})
                out.write(data)
                toc.setdefault(key, {})[arch] = {**entry, "ordinal": ordinal}
            blob_size = out.tell() - blob_start

            toc_data = {
                k: v for k, v in template.toc_data.items() if k not in _LAYOUT_KEYS
            }
            toc_data["toc"] = toc
            toc_data["compression_scheme"] = scheme
            if scheme == SCHEME_ZSTD:
                toc_data["zstd_offset"] = blob_start
                toc_data["zstd_size"] = blob_size
            else:
                toc_data["blobs"] = blobs

            toc_offset = out.tell()
            msgpack.pack(toc_data, out, use_bin_type=True)
            out.seek(len(MAGIC) + _UINT32.size)
            out.write(_TOC_OFFSET.pack(toc_offset))
    finally:
        for f in sources.values():
            f.close()
    os.replace(temp, output)


def replace_entries(
    baseline: KpackArchive,
    replacement: KpackArchive,
    owned: Callable[[str], bool],
    output: Path,
) -> int:
    """Writes baseline with every key owned() accepts taken from replacement.

    Owned keys are dropped from baseline even where replacement has no
    counterpart, so the result holds replacement's code objects for them and
    nothing else. Returns how many entries came from replacement.
    """
    kept = [(k, a, e, baseline) for k, a, e in baseline.entries() if not owned(k)]
    swapped = [(k, a, e, replacement) for k, a, e in replacement.entries() if owned(k)]
    write_kpack(output, baseline, kept + swapped)
    return len(swapped)
