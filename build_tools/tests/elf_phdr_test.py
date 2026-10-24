# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

from _therock_utils.elf_phdr import (
    PT_LOAD,
    PT_PHDR,
    has_profile_counters,
    normalize_instrumented_binaries,
    normalize_phdr_vaddr,
)

EHDR = struct.Struct("<16sHHIQQQIHHHHHH")
PHDR = struct.Struct("<IIQQQQQQ")
SHDR = struct.Struct("<IIQQQQIIQQ")
EM_AMDGPU = 224
TABLE_SIZE = 4 * PHDR.size


def write_split_library(
    path: Path,
    *,
    phoff: int,
    phdr_vaddr: int,
    data_memsz: int = 0x1000,
    sections: tuple[str, ...] = (),
    machine: int = 62,
) -> None:
    """Writes a shared object laid out the way the kpack split leaves one.

    The program header table sits at phoff, covered by a PT_LOAD of its own
    mapped at phdr_vaddr. data_memsz sizes the data segment, which sets how far
    the other segments' addresses reach.
    """
    names = ["", *sections, ".shstrtab"]
    shstrtab = b"".join(name.encode() + b"\0" for name in names)
    shstrtab_off, shoff = 0x400, 0x500
    data = bytearray(max(phoff + TABLE_SIZE, shoff + len(names) * SHDR.size))
    data[: EHDR.size] = EHDR.pack(
        b"\x7fELF\x02\x01\x01" + b"\0" * 9,
        3,
        machine,
        1,
        0,
        phoff,
        shoff,
        0,
        EHDR.size,
        PHDR.size,
        4,
        SHDR.size,
        len(names),
        len(names) - 1,
    )
    data[shstrtab_off : shstrtab_off + len(shstrtab)] = shstrtab
    name_offset = 0
    for i, name in enumerate(names):
        is_strtab = name == ".shstrtab"
        data[shoff + i * SHDR.size : shoff + (i + 1) * SHDR.size] = SHDR.pack(
            name_offset,
            3 if is_strtab else int(bool(name)),
            0,
            0,
            shstrtab_off if is_strtab else 0,
            len(shstrtab) if is_strtab else 0,
            0,
            0,
            1,
            0,
        )
        name_offset += len(name) + 1
    phdrs = [
        (PT_PHDR, 4, phoff, phdr_vaddr, phdr_vaddr, TABLE_SIZE, TABLE_SIZE, 8),
        (PT_LOAD, 5, 0, 0, 0, 0x1000, 0x1000, 0x1000),
        (PT_LOAD, 6, 0x1000, 0x1000, 0x1000, 0x1000, data_memsz, 0x1000),
        (PT_LOAD, 4, phoff, phdr_vaddr, phdr_vaddr, TABLE_SIZE, TABLE_SIZE, 0x1000),
    ]
    for i, phdr in enumerate(phdrs):
        data[phoff + i * PHDR.size : phoff + (i + 1) * PHDR.size] = PHDR.pack(*phdr)
    path.write_bytes(bytes(data))


def read_layout(path: Path) -> tuple[int, list[tuple]]:
    """Returns (e_phoff, program headers read from the table e_phoff names)."""
    data = path.read_bytes()
    phoff = EHDR.unpack_from(data)[5]
    return phoff, [PHDR.unpack_from(data, phoff + i * PHDR.size) for i in range(4)]


class NormalizePhdrVaddrTest(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp_dir.cleanup)
        self.root = Path(self._temp_dir.name)
        self.lib = self.root / "libfoo.so"

    def assert_pinned(self, phoff: int, phdrs: list[tuple]) -> None:
        # What __llvm_write_binary_ids needs: base + e_phoff is the table.
        for index in (0, 3):
            self.assertEqual(phdrs[index][2:5], (phoff, phoff, phoff))

    def test_table_above_the_other_segments_is_pinned_in_place(self):
        # librocsparse.so and librocsolver.so: a pure metadata edit.
        write_split_library(self.lib, phoff=0x5000, phdr_vaddr=0x3000)
        size = self.lib.stat().st_size

        self.assertTrue(normalize_phdr_vaddr(self.lib))

        phoff, phdrs = read_layout(self.lib)
        self.assertEqual(phoff, 0x5000)
        self.assert_pinned(phoff, phdrs)
        self.assertEqual(self.lib.stat().st_size, size)

    def test_table_below_the_address_ceiling_is_copied_past_it(self):
        # librocrand.so: pinning in place would overlap the data segment, which
        # reaches 0x9000, so the table moves to a fresh offset at the ceiling.
        write_split_library(
            self.lib, phoff=0x4000, phdr_vaddr=0xA000, data_memsz=0x8000
        )
        before = self.lib.read_bytes()

        self.assertTrue(normalize_phdr_vaddr(self.lib))

        phoff, phdrs = read_layout(self.lib)
        self.assertEqual(phoff, 0x9000)
        self.assert_pinned(phoff, phdrs)
        self.assertEqual(self.lib.stat().st_size, 0x9000 + TABLE_SIZE)
        # The data segment is untouched; the old table just goes unreferenced.
        self.assertEqual(phdrs[2], PHDR.unpack_from(before, 0x4000 + 2 * PHDR.size))

    def test_table_never_relocated_is_left_alone(self):
        # The table still lives in the first PT_LOAD, as the linker wrote it.
        write_split_library(self.lib, phoff=0x40, phdr_vaddr=0x40)
        before = self.lib.read_bytes()

        self.assertFalse(normalize_phdr_vaddr(self.lib))
        self.assertEqual(self.lib.read_bytes(), before)

    def test_table_already_pinned_is_left_alone(self):
        write_split_library(self.lib, phoff=0x5000, phdr_vaddr=0x5000)
        before = self.lib.read_bytes()

        self.assertFalse(normalize_phdr_vaddr(self.lib))
        self.assertEqual(self.lib.read_bytes(), before)

    def test_files_that_are_not_host_elf_objects_are_left_alone(self):
        text = self.root / "notes.txt"
        text.write_text("not an ELF file")
        code_object = self.root / "kernel.co"
        write_split_library(
            code_object, phoff=0x5000, phdr_vaddr=0x3000, machine=EM_AMDGPU
        )
        before = code_object.read_bytes()

        self.assertFalse(normalize_phdr_vaddr(text))
        self.assertFalse(normalize_phdr_vaddr(code_object))
        self.assertEqual(code_object.read_bytes(), before)


class InstrumentedBinariesTest(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp_dir.cleanup)
        self.root = Path(self._temp_dir.name)

    def test_profile_counters_mark_an_instrumented_object(self):
        instrumented = self.root / "libyes.so"
        plain = self.root / "libno.so"
        write_split_library(
            instrumented, phoff=0x5000, phdr_vaddr=0x3000, sections=("__llvm_prf_cnts",)
        )
        write_split_library(plain, phoff=0x5000, phdr_vaddr=0x3000, sections=(".text",))

        self.assertTrue(has_profile_counters(instrumented))
        self.assertFalse(has_profile_counters(plain))
        self.assertFalse(has_profile_counters(self.root / "missing.so"))

    @unittest.skipIf(sys.platform == "win32", "symlinks need privileges on Windows")
    def test_only_instrumented_objects_are_normalized(self):
        # An uninstrumented library never runs __llvm_write_binary_ids, so it
        # keeps exactly the layout the split gave it.
        lib = self.root / "lib"
        lib.mkdir()
        instrumented = lib / "librocsparse.so.1.0"
        plain = lib / "libhipsparse.so.4.8.0"
        write_split_library(
            instrumented, phoff=0x5000, phdr_vaddr=0x3000, sections=("__llvm_prf_cnts",)
        )
        write_split_library(plain, phoff=0x5000, phdr_vaddr=0x3000)
        (lib / "librocsparse.so").symlink_to(instrumented.name)
        plain_before = plain.read_bytes()

        changed = normalize_instrumented_binaries([lib, self.root / "bin"])

        self.assertEqual(changed, [instrumented])
        self.assertEqual(plain.read_bytes(), plain_before)


if __name__ == "__main__":
    unittest.main()
