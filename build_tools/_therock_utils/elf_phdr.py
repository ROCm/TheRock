# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Puts a relocated ELF program header table where in-process readers look.

kpack moves the program header table of a binary it rewrites to the end of the
file, mapped by a PT_LOAD of its own whose p_vaddr is only page-congruent with
the table's file offset. ld.so copes, but code that finds the table at
__ehdr_start + e_phoff reads memory that is not mapped. compiler-rt's profile
writer does exactly that to record binary IDs, so an instrumented library that
kpack has rewritten segfaults at exit before writing a single counter.

kpack pins p_vaddr == p_offset for executables (normalize_phdr_vaddr in
rocm_kpack/elf/phdr_manager.py) and leaves shared libraries alone.
pin_phdr_table() applies the same edit to a shared library.
"""

import os
import struct
from pathlib import Path

_EHDR = struct.Struct("<16sHHIQQQIHHHHHH")
_PHDR = struct.Struct("<IIQQQQQQ")
_SHDR = struct.Struct("<IIQQQQIIQQ")
_E_PHOFF_OFFSET = 0x20

_ET_DYN = 3
_EM_X86_64 = 62
_PN_XNUM = 0xFFFF
_SHN_XINDEX = 0xFFFF
_PT_LOAD = 1
_PT_INTERP = 3
_PT_PHDR = 6
_PAGE_SIZE = 0x1000

# Program header fields, in _PHDR order.
_TYPE, _FLAGS, _OFFSET, _VADDR, _PADDR, _FILESZ, _MEMSZ, _ALIGN = range(8)


def _read_at(f, offset: int, size: int) -> bytes:
    f.seek(offset)
    data = f.read(size)
    if len(data) != size:
        raise ValueError(f"truncated ELF: wanted {size} bytes at {offset:#x}")
    return data


def _round_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _section_names(f, e_shoff: int, e_shentsize: int, e_shnum: int, e_shstrndx: int):
    if not e_shoff or e_shentsize < _SHDR.size:
        return set()
    first = _SHDR.unpack(_read_at(f, e_shoff, _SHDR.size))
    if e_shnum == 0:
        e_shnum = first[5]
    if e_shstrndx == _SHN_XINDEX:
        e_shstrndx = first[6]
    if e_shstrndx >= e_shnum:
        return set()
    table = _read_at(f, e_shoff, e_shnum * e_shentsize)
    sections = [_SHDR.unpack_from(table, i * e_shentsize) for i in range(e_shnum)]
    strtab = sections[e_shstrndx]
    names = _read_at(f, strtab[4], strtab[5])
    return {
        names[s[0] : names.find(b"\0", s[0])].decode(errors="replace")
        for s in sections
        if s[0] < len(names)
    }


def pin_phdr_table(path: Path, *, require_section: str | None = None) -> bool:
    """Pins a shared library's relocated program header table to p_vaddr == p_offset.

    When the table already sits above every other PT_LOAD's address range this
    only rewrites two program headers; otherwise the table is copied to a new
    end-of-file offset above that range and e_phoff follows it, as kpack does
    for executables.

    Leaves the file untouched, returning False, unless it is a little-endian
    ELF64 x86-64 shared library without a PT_INTERP whose table lives in a
    PT_LOAD of its own, the last one, starting at e_phoff and mapped somewhere
    else, and, when require_section is given, it has a section of that name.
    Returns True if the file was modified.
    """
    with open(path, "r+b") as f:
        header = f.read(_EHDR.size)
        if len(header) < _EHDR.size or header[:4] != b"\x7fELF":
            return False
        if header[4] != 2 or header[5] != 1:  # ELFCLASS64, ELFDATA2LSB
            return False
        (
            _,
            e_type,
            e_machine,
            _,
            _,
            e_phoff,
            e_shoff,
            _,
            _,
            e_phentsize,
            e_phnum,
            e_shentsize,
            e_shnum,
            e_shstrndx,
        ) = _EHDR.unpack(header)
        if e_type != _ET_DYN or e_machine != _EM_X86_64:
            return False
        if e_phnum in (0, _PN_XNUM) or e_phentsize < _PHDR.size:
            return False

        table = bytearray(_read_at(f, e_phoff, e_phnum * e_phentsize))
        phdrs = [
            list(_PHDR.unpack_from(table, i * e_phentsize)) for i in range(e_phnum)
        ]
        if any(p[_TYPE] == _PT_INTERP for p in phdrs):
            return False
        cover_index = next(
            (
                i
                for i, p in enumerate(phdrs)
                if p[_TYPE] == _PT_LOAD
                and p[_OFFSET] <= e_phoff < p[_OFFSET] + p[_FILESZ]
            ),
            None,
        )
        if cover_index is None:
            return False
        cover = phdrs[cover_index]
        if cover[_OFFSET] != e_phoff or cover[_VADDR] == cover[_OFFSET]:
            return False
        # The pinned table ends up above every other segment, so only the last
        # PT_LOAD can take it without breaking their ascending p_vaddr order.
        if any(p[_TYPE] == _PT_LOAD for p in phdrs[cover_index + 1 :]):
            return False
        if require_section is not None and require_section not in _section_names(
            f, e_shoff, e_shentsize, e_shnum, e_shstrndx
        ):
            return False

        alignment = max(cover[_ALIGN], _PAGE_SIZE)
        ceiling = _round_up(
            max(
                (
                    p[_VADDR] + p[_MEMSZ]
                    for i, p in enumerate(phdrs)
                    if p[_TYPE] == _PT_LOAD and i != cover_index
                ),
                default=0,
            ),
            alignment,
        )
        new_phoff = e_phoff
        if e_phoff < ceiling:
            # Pinning the table where it is would overlap another segment.
            # Copy the whole covering segment, spare slots included, so the
            # file holds every byte the PT_LOAD claims.
            cover_bytes = _read_at(f, e_phoff, cover[_FILESZ])
            file_size = f.seek(0, os.SEEK_END)
            new_phoff = _round_up(max(file_size, ceiling), alignment)
            f.write(b"\0" * (new_phoff - file_size))
            f.write(cover_bytes)

        for i, p in enumerate(phdrs):
            if i == cover_index or p[_TYPE] == _PT_PHDR:
                p[_OFFSET] = p[_VADDR] = p[_PADDR] = new_phoff
            _PHDR.pack_into(table, i * e_phentsize, *p)
        f.seek(new_phoff)
        f.write(table)
        if new_phoff != e_phoff:
            f.seek(_E_PHOFF_OFFSET)
            f.write(struct.pack("<Q", new_phoff))
    return True
