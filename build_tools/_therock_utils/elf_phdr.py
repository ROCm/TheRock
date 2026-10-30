# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Keeps an instrumented binary's program headers where its profile runtime looks.

The kpack split rewrites every host binary that carries device code, and when
the program header table outgrows its slot it moves the table to a trailing
PT_LOAD whose p_vaddr is only page-congruent with its p_offset. rocm_kpack's
normalize_phdr_vaddr() then pins p_vaddr to p_offset, but only for executables:
old kernels compute AT_PHDR as load_bias + e_phoff, while ld.so maps shared
libraries correctly either way, so those are left as they are.

The LLVM profile runtime linked into an instrumented library makes the same
assumption as those kernels. Its __llvm_write_binary_ids() reads the headers at
&__ehdr_start + e_phoff, so in a split library it reads a file offset as if it
were an address: past the end of the image the process faults at exit, before
the library's profile is written (librocsparse.so, librocsolver.so); inside the
image it parses whatever is mapped there (librocrand.so).

normalize_phdr_vaddr() below applies the kpack normalization to any 64-bit
little-endian ELF file, executable or not.
"""

import os
import struct
from pathlib import Path
from typing import BinaryIO, Iterator

PT_LOAD = 1
PT_PHDR = 6
EM_X86_64 = 62
PAGE_SIZE = 0x1000  # x86-64

_EHDR_SIZE = 64
_PHDR = struct.Struct("<IIQQQQQQ")
_SHDR = struct.Struct("<IIQQQQIIQQ")
# Present in the host object only when its own code is instrumented, i.e. when
# it carries a profile runtime.
_PROFILE_COUNTERS_SECTION = "__llvm_prf_cnts"


def _round_up_to_page(value: int) -> int:
    return (value + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1)


def _read_ehdr(f: BinaryIO) -> bytes | None:
    """Returns the ELF header of a 64-bit little-endian x86-64 file, else None."""
    f.seek(0)
    ehdr = f.read(_EHDR_SIZE)
    if len(ehdr) < _EHDR_SIZE or ehdr[:4] != b"\x7fELF" or ehdr[4:6] != b"\x02\x01":
        return None
    (machine,) = struct.unpack_from("<H", ehdr, 0x12)
    return ehdr if machine == EM_X86_64 else None


def _read_phdrs(f: BinaryIO, ehdr: bytes) -> list[list[int]]:
    (phoff,) = struct.unpack_from("<Q", ehdr, 0x20)
    phentsize, phnum = struct.unpack_from("<HH", ehdr, 0x36)
    f.seek(phoff)
    table = f.read(phentsize * phnum)
    return [
        list(_PHDR.unpack_from(table, i * phentsize))
        for i in range(phnum)
        if (i + 1) * phentsize <= len(table)
    ]


def _write_phdr(f: BinaryIO, phoff: int, phentsize: int, index: int, phdr) -> None:
    f.seek(phoff + index * phentsize)
    f.write(_PHDR.pack(*phdr))


def has_profile_counters(path: Path) -> bool:
    """Whether the host object's own code is instrumented for profiling."""
    try:
        with open(path, "rb") as f:
            ehdr = _read_ehdr(f)
            if ehdr is None:
                return False
            (shoff,) = struct.unpack_from("<Q", ehdr, 0x28)
            shentsize, shnum, shstrndx = struct.unpack_from("<HHH", ehdr, 0x3A)
            if not shoff or shstrndx >= shnum:
                return False
            f.seek(shoff)
            table = f.read(shentsize * shnum)
            if len(table) < shentsize * shnum:
                return False
            sections = [_SHDR.unpack_from(table, i * shentsize) for i in range(shnum)]
            strtab = sections[shstrndx]
            f.seek(strtab[4])
            names = f.read(strtab[5])
    except OSError:
        return False
    wanted = _PROFILE_COUNTERS_SECTION.encode() + b"\0"
    return any(names[s[0] : s[0] + len(wanted)] == wanted for s in sections)


def normalize_phdr_vaddr(path: Path) -> bool:
    """Pins a relocated program header PT_LOAD to p_vaddr == p_offset.

    Mirrors rocm_kpack.elf.phdr_manager.normalize_phdr_vaddr: a pure metadata
    edit when the table already sits above every other segment's addresses,
    otherwise the table is copied to a fresh end-of-file offset at or above
    them, which grows the file. Returns True if the file was modified.
    """
    with open(path, "r+b") as f:
        ehdr = _read_ehdr(f)
        if ehdr is None:
            return False
        (phoff,) = struct.unpack_from("<Q", ehdr, 0x20)
        phentsize, _ = struct.unpack_from("<HH", ehdr, 0x36)
        phdrs = _read_phdrs(f, ehdr)

        # p_offset, p_vaddr and p_filesz sit at fields 2, 3 and 5.
        cover = next(
            (
                i
                for i, ph in enumerate(phdrs)
                if ph[0] == PT_LOAD and ph[2] <= phoff < ph[2] + ph[5]
            ),
            None,
        )
        # Only a relocated table, which kpack gives a PT_LOAD of its own that
        # starts exactly at e_phoff, and only one not already pinned.
        if cover is None or phdrs[cover][2] != phoff or phdrs[cover][3] == phoff:
            return False
        ceiling = max(
            (
                ph[3] + ph[6]
                for i, ph in enumerate(phdrs)
                if ph[0] == PT_LOAD and i != cover
            ),
            default=0,
        )

        new_phoff = phoff
        if phoff < _round_up_to_page(ceiling):
            # Pinning the address to the current offset would overlap another
            # segment, so copy the whole covering segment past the ceiling.
            f.seek(phoff)
            table = f.read(phdrs[cover][5])
            end = f.seek(0, os.SEEK_END)
            new_phoff = _round_up_to_page(max(end, _round_up_to_page(ceiling)))
            f.write(b"\0" * (new_phoff - end))
            f.write(table)
            f.seek(0x20)
            f.write(struct.pack("<Q", new_phoff))

        for i, ph in enumerate(phdrs):
            if i == cover or ph[0] == PT_PHDR:
                ph[2] = ph[3] = ph[4] = new_phoff
                _write_phdr(f, new_phoff, phentsize, i, ph)
    return True


def iter_regular_files(root: Path) -> Iterator[Path]:
    """Yields the regular files under root, skipping symlinks."""
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            path = Path(dirpath) / name
            if path.is_symlink() or not path.is_file():
                continue
            yield path


def normalize_instrumented_binaries(roots: list[Path]) -> list[Path]:
    """Normalizes every profile-instrumented x86-64 ELF file under roots.

    Uninstrumented files never run __llvm_write_binary_ids, so they are left
    exactly as the split wrote them.
    """
    changed = []
    for root in roots:
        if not root.is_dir():
            continue
        for path in iter_regular_files(root):
            if has_profile_counters(path) and normalize_phdr_vaddr(path):
                changed.append(path)
    return changed
