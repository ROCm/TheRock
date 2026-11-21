#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import collections
import struct
import tempfile
import unittest
from pathlib import Path

from _therock_utils.elf_phdr import pin_phdr_table

_EHDR = struct.Struct("<16sHHIQQQIHHHHHH")
_PHDR = struct.Struct("<IIQQQQQQ")
_SHDR = struct.Struct("<IIQQQQIIQQ")

ET_EXEC = 2
ET_DYN = 3
EM_X86_64 = 62
PT_LOAD = 1
PT_INTERP = 3
PT_PHDR = 6
SHT_PROGBITS = 1
SHT_STRTAB = 3

Phdr = collections.namedtuple(
    "Phdr", "type flags offset vaddr paddr filesz memsz align"
)

# Where the libraries below keep their (relocated) program header table, and
# the address kpack would have mapped it at: page-congruent, nothing more.
TABLE_OFFSET = 0x3000
TABLE_VADDR = 0x9000


def _write_library(
    path: Path,
    *,
    elf_class: int = 2,
    e_type: int = ET_DYN,
    data_end: int = 0x2000,
    relocated: bool = True,
    table_vaddr: int = TABLE_VADDR,
    table_segment_last: bool = True,
    interp: bool = False,
    sections=("__llvm_prf_cnts",),
):
    """Writes a shared library laid out the way kpack leaves one.

    Two PT_LOADs map the first 0x2000 bytes of the file, the second reaching
    data_end in memory. With relocated, the program header table sits at
    TABLE_OFFSET, mapped at table_vaddr by a PT_LOAD of its own; otherwise it
    follows the ELF header inside the first PT_LOAD, as the linker left it.
    """
    table_offset = TABLE_OFFSET if relocated else 0x40
    table_vaddr = table_vaddr if relocated else table_offset
    count = 3 + interp + relocated
    table_size = count * _PHDR.size

    phdrs = [
        Phdr(
            PT_PHDR,
            4,
            table_offset,
            table_vaddr,
            table_vaddr,
            table_size,
            table_size,
            8,
        ),
    ]
    if interp:
        phdrs.append(Phdr(PT_INTERP, 4, 0x200, 0x200, 0x200, 0x1C, 0x1C, 1))
    phdrs.append(Phdr(PT_LOAD, 5, 0, 0, 0, 0x1000, 0x1000, 0x1000))
    phdrs.append(
        Phdr(PT_LOAD, 6, 0x1000, 0x1000, 0x1000, 0x1000, data_end - 0x1000, 0x1000)
    )
    if relocated:
        table_segment = Phdr(
            PT_LOAD,
            4,
            table_offset,
            table_vaddr,
            table_vaddr,
            table_size,
            table_size,
            0x1000,
        )
        phdrs.insert(
            len(phdrs) if table_segment_last else len(phdrs) - 1, table_segment
        )

    names = [".shstrtab", *sections]
    strtab = b"\0" + b"".join(name.encode() + b"\0" for name in names)
    shoff = 0x2000
    shnum = 1 + len(names)
    strtab_offset = shoff + shnum * _SHDR.size
    shdrs = [_SHDR.pack(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)]
    for name in names:
        name_offset = strtab.index(b"\0" + name.encode() + b"\0") + 1
        if name == ".shstrtab":
            shdrs.append(
                _SHDR.pack(
                    name_offset,
                    SHT_STRTAB,
                    0,
                    0,
                    strtab_offset,
                    len(strtab),
                    0,
                    0,
                    1,
                    0,
                )
            )
        else:
            shdrs.append(
                _SHDR.pack(
                    name_offset, SHT_PROGBITS, 3, 0x1000, 0x1000, 0x10, 0, 0, 8, 0
                )
            )

    image = bytearray(max(strtab_offset + len(strtab), table_offset + table_size))
    ident = b"\x7fELF" + bytes([elf_class, 1, 1]) + bytes(9)
    image[: _EHDR.size] = _EHDR.pack(
        ident,
        e_type,
        EM_X86_64,
        1,
        0,
        table_offset,
        shoff,
        0,
        _EHDR.size,
        _PHDR.size,
        count,
        _SHDR.size,
        shnum,
        1,
    )
    image[0x1000:0x2000] = bytes(range(256)) * 16
    image[shoff:strtab_offset] = b"".join(shdrs)
    image[strtab_offset : strtab_offset + len(strtab)] = strtab
    for i, phdr in enumerate(phdrs):
        _PHDR.pack_into(image, table_offset + i * _PHDR.size, *phdr)
    path.write_bytes(image)


def _read_phdrs(path: Path):
    data = path.read_bytes()
    fields = _EHDR.unpack_from(data)
    e_phoff, e_phentsize, e_phnum = fields[5], fields[9], fields[10]
    return e_phoff, [
        Phdr(*_PHDR.unpack_from(data, e_phoff + i * e_phentsize))
        for i in range(e_phnum)
    ]


class PinPhdrTableTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "libfoo.so"

    def _assert_pinned(self, phdrs, address):
        table_segment = next(
            p for p in phdrs if p.type == PT_LOAD and p.offset == p.vaddr == address
        )
        self.assertEqual(table_segment.paddr, address)
        self.assertEqual(phdrs[0].type, PT_PHDR)
        self.assertEqual(
            (phdrs[0].offset, phdrs[0].vaddr, phdrs[0].paddr),
            (address, address, address),
        )

    def test_pins_table_in_place_when_it_clears_other_segments(self):
        _write_library(self.path)
        before = self.path.read_bytes()
        _, phdrs_before = _read_phdrs(self.path)

        self.assertTrue(pin_phdr_table(self.path, require_section="__llvm_prf_cnts"))

        e_phoff, phdrs = _read_phdrs(self.path)
        self.assertEqual(e_phoff, TABLE_OFFSET)
        self.assertEqual(self.path.stat().st_size, len(before))
        self._assert_pinned(phdrs, TABLE_OFFSET)
        self.assertEqual(phdrs[1:3], phdrs_before[1:3])
        self.assertEqual(self.path.read_bytes()[:TABLE_OFFSET], before[:TABLE_OFFSET])

    def test_moves_table_above_address_range_it_would_overlap(self):
        # The second segment reaches 0x5800 in memory, past the table's offset.
        _write_library(self.path, data_end=0x5800)
        before = self.path.read_bytes()
        _, phdrs_before = _read_phdrs(self.path)

        self.assertTrue(pin_phdr_table(self.path))

        e_phoff, phdrs = _read_phdrs(self.path)
        self.assertEqual(e_phoff, 0x6000)
        self._assert_pinned(phdrs, 0x6000)
        self.assertEqual(phdrs[1:3], phdrs_before[1:3])
        after = self.path.read_bytes()
        table_size = len(phdrs) * _PHDR.size
        self.assertEqual(len(after), 0x6000 + table_size)
        self.assertEqual(after[len(before) : 0x6000], bytes(0x6000 - len(before)))
        # Apart from e_phoff, the original bytes stay as they were.
        self.assertEqual(after[:0x20], before[:0x20])
        self.assertEqual(after[0x28 : len(before)], before[0x28:])

    def test_second_call_changes_nothing(self):
        for data_end in (0x2000, 0x5800):
            with self.subTest(data_end=hex(data_end)):
                _write_library(self.path, data_end=data_end)
                self.assertTrue(pin_phdr_table(self.path))
                pinned = self.path.read_bytes()
                self.assertFalse(pin_phdr_table(self.path))
                self.assertEqual(self.path.read_bytes(), pinned)

    def test_leaves_other_files_untouched(self):
        cases = {
            "executable": dict(interp=True),
            "ET_EXEC": dict(e_type=ET_EXEC),
            "ELF32": dict(elf_class=1),
            "table where the linker put it": dict(relocated=False),
            "table already pinned": dict(table_vaddr=TABLE_OFFSET),
            "table segment not the last PT_LOAD": dict(table_segment_last=False),
            "required section missing": dict(sections=(".text",)),
        }
        for name, kwargs in cases.items():
            with self.subTest(name):
                _write_library(self.path, **kwargs)
                before = self.path.read_bytes()
                self.assertFalse(
                    pin_phdr_table(self.path, require_section="__llvm_prf_cnts")
                )
                self.assertEqual(self.path.read_bytes(), before)

    def test_leaves_non_elf_files_untouched(self):
        for content in (b"", b"\x7fELF", b"TensileLibrary data" * 8):
            with self.subTest(content=content[:8]):
                self.path.write_bytes(content)
                self.assertFalse(pin_phdr_table(self.path))
                self.assertEqual(self.path.read_bytes(), content)


if __name__ == "__main__":
    unittest.main()
