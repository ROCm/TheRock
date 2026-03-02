"""Minimal ELF parser for AMDGPU code objects (.co / .hsaco files)."""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import BinaryIO

# ELF constants
ELF_MAGIC = b"\x7fELF"

# ELF class
ELFCLASS64 = 2

# ELF data encoding
ELFDATA2LSB = 1  # little-endian

# ELF machine types
EM_AMDGPU = 0xE0

# Section header types
SHT_NULL = 0
SHT_PROGBITS = 1
SHT_SYMTAB = 2
SHT_STRTAB = 3
SHT_NOTE = 7

# Section flags
SHF_ALLOC = 0x2
SHF_EXECINSTR = 0x4

# Symbol binding
STB_LOCAL = 0
STB_GLOBAL = 1

# Symbol types
STT_NOTYPE = 0
STT_OBJECT = 1
STT_FUNC = 2
STT_SECTION = 3
STT_AMDGPU_HSA_KERNEL = 10


@dataclass
class ELFHeader:
    """64-bit ELF header."""

    ei_class: int = 0
    ei_data: int = 0
    ei_version: int = 0
    ei_osabi: int = 0
    e_type: int = 0
    e_machine: int = 0
    e_version: int = 0
    e_entry: int = 0
    e_phoff: int = 0
    e_shoff: int = 0
    e_flags: int = 0
    e_ehsize: int = 0
    e_phentsize: int = 0
    e_phnum: int = 0
    e_shentsize: int = 0
    e_shnum: int = 0
    e_shstrndx: int = 0

    STRUCT_FMT = "<16sHHIQQQIHHHHHH"

    @classmethod
    def from_bytes(cls, data: bytes) -> ELFHeader:
        fields = struct.unpack(cls.STRUCT_FMT, data[: struct.calcsize(cls.STRUCT_FMT)])
        ident = fields[0]
        return cls(
            ei_class=ident[4],
            ei_data=ident[5],
            ei_version=ident[6],
            ei_osabi=ident[7],
            e_type=fields[1],
            e_machine=fields[2],
            e_version=fields[3],
            e_entry=fields[4],
            e_phoff=fields[5],
            e_shoff=fields[6],
            e_flags=fields[7],
            e_ehsize=fields[8],
            e_phentsize=fields[9],
            e_phnum=fields[10],
            e_shentsize=fields[11],
            e_shnum=fields[12],
            e_shstrndx=fields[13],
        )


@dataclass
class SectionHeader:
    """64-bit ELF section header."""

    sh_name: int = 0
    sh_type: int = 0
    sh_flags: int = 0
    sh_addr: int = 0
    sh_offset: int = 0
    sh_size: int = 0
    sh_link: int = 0
    sh_info: int = 0
    sh_addralign: int = 0
    sh_entsize: int = 0
    name: str = ""

    STRUCT_FMT = "<IIQQQQIIQQ"
    SIZE = struct.calcsize(STRUCT_FMT)

    @classmethod
    def from_bytes(cls, data: bytes, offset: int = 0) -> SectionHeader:
        fields = struct.unpack(cls.STRUCT_FMT, data[offset : offset + cls.SIZE])
        return cls(
            sh_name=fields[0],
            sh_type=fields[1],
            sh_flags=fields[2],
            sh_addr=fields[3],
            sh_offset=fields[4],
            sh_size=fields[5],
            sh_link=fields[6],
            sh_info=fields[7],
            sh_addralign=fields[8],
            sh_entsize=fields[9],
        )


@dataclass
class Symbol:
    """64-bit ELF symbol."""

    st_name: int = 0
    st_info: int = 0
    st_other: int = 0
    st_shndx: int = 0
    st_value: int = 0
    st_size: int = 0
    name: str = ""

    STRUCT_FMT = "<IBBHQQ"
    SIZE = struct.calcsize(STRUCT_FMT)

    @classmethod
    def from_bytes(cls, data: bytes, offset: int = 0) -> Symbol:
        fields = struct.unpack(cls.STRUCT_FMT, data[offset : offset + cls.SIZE])
        return cls(
            st_name=fields[0],
            st_info=fields[1],
            st_other=fields[2],
            st_shndx=fields[3],
            st_value=fields[4],
            st_size=fields[5],
        )

    @property
    def binding(self) -> int:
        return self.st_info >> 4

    @property
    def type(self) -> int:
        return self.st_info & 0xF


@dataclass
class AMDGPUCodeObject:
    """Parsed AMDGPU ELF code object."""

    header: ELFHeader
    sections: list[SectionHeader]
    symbols: list[Symbol]
    raw_data: bytes
    text_section: SectionHeader | None = None
    rodata_section: SectionHeader | None = None
    note_section: SectionHeader | None = None

    @property
    def code(self) -> bytes:
        """Return the .text section (executable code)."""
        if self.text_section is None:
            return b""
        off = self.text_section.sh_offset
        sz = self.text_section.sh_size
        return self.raw_data[off : off + sz]

    def kernel_symbols(self) -> list[Symbol]:
        """Return symbols that are kernel entry points."""
        return [
            s
            for s in self.symbols
            if s.type == STT_AMDGPU_HSA_KERNEL or (
                s.type == STT_FUNC and s.binding == STB_GLOBAL
            )
        ]

    def get_section_data(self, section: SectionHeader) -> bytes:
        """Return raw data for a section."""
        return self.raw_data[section.sh_offset : section.sh_offset + section.sh_size]


def _resolve_string(strtab: bytes, offset: int) -> str:
    """Read a null-terminated string from a string table."""
    end = strtab.index(b"\x00", offset) if b"\x00" in strtab[offset:] else len(strtab)
    return strtab[offset:end].decode("utf-8", errors="replace")


def parse_elf(data: bytes) -> AMDGPUCodeObject:
    """Parse an AMDGPU ELF code object from raw bytes."""
    if data[:4] != ELF_MAGIC:
        from amd_gpu_driver.errors import KernelLoadError
        raise KernelLoadError("Not a valid ELF file (bad magic)")

    header = ELFHeader.from_bytes(data)

    if header.ei_class != ELFCLASS64:
        from amd_gpu_driver.errors import KernelLoadError
        raise KernelLoadError("Only 64-bit ELF files are supported")

    # Parse section headers
    sections: list[SectionHeader] = []
    for i in range(header.e_shnum):
        offset = header.e_shoff + i * header.e_shentsize
        sh = SectionHeader.from_bytes(data, offset)
        sections.append(sh)

    # Resolve section names from shstrtab
    if header.e_shstrndx < len(sections):
        shstrtab_hdr = sections[header.e_shstrndx]
        shstrtab = data[shstrtab_hdr.sh_offset : shstrtab_hdr.sh_offset + shstrtab_hdr.sh_size]
        for sh in sections:
            sh.name = _resolve_string(shstrtab, sh.sh_name)

    # Find key sections
    text_section = None
    rodata_section = None
    note_section = None
    symtab_section = None
    strtab_section = None

    for sh in sections:
        if sh.name == ".text":
            text_section = sh
        elif sh.name == ".rodata":
            rodata_section = sh
        elif sh.sh_type == SHT_NOTE:
            note_section = sh
        elif sh.sh_type == SHT_SYMTAB:
            symtab_section = sh
        elif sh.name == ".strtab":
            strtab_section = sh

    # Parse symbols
    symbols: list[Symbol] = []
    if symtab_section is not None:
        strtab_data = b""
        if strtab_section is not None:
            strtab_data = data[
                strtab_section.sh_offset : strtab_section.sh_offset + strtab_section.sh_size
            ]
        elif symtab_section.sh_link < len(sections):
            linked = sections[symtab_section.sh_link]
            strtab_data = data[linked.sh_offset : linked.sh_offset + linked.sh_size]

        num_syms = symtab_section.sh_size // Symbol.SIZE
        for i in range(num_syms):
            offset = symtab_section.sh_offset + i * Symbol.SIZE
            sym = Symbol.from_bytes(data, offset)
            if strtab_data and sym.st_name < len(strtab_data):
                sym.name = _resolve_string(strtab_data, sym.st_name)
            symbols.append(sym)

    return AMDGPUCodeObject(
        header=header,
        sections=sections,
        symbols=symbols,
        raw_data=data,
        text_section=text_section,
        rodata_section=rodata_section,
        note_section=note_section,
    )


def parse_elf_file(path: str) -> AMDGPUCodeObject:
    """Parse an AMDGPU ELF code object from a file path."""
    from pathlib import Path
    return parse_elf(Path(path).read_bytes())
