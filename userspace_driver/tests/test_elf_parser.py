"""Tests for ELF parser."""

import struct

import pytest

from amd_gpu_driver.kernel.elf_parser import (
    ELFHeader,
    SectionHeader,
    Symbol,
    parse_elf,
    ELF_MAGIC,
    ELFCLASS64,
    ELFDATA2LSB,
    EM_AMDGPU,
    SHT_PROGBITS,
    SHT_SYMTAB,
    SHT_STRTAB,
    STT_AMDGPU_HSA_KERNEL,
    STB_GLOBAL,
)
from amd_gpu_driver.kernel.descriptor import KernelDescriptor, KERNEL_DESCRIPTOR_SIZE


def _make_minimal_elf(
    *,
    code: bytes = b"\x00" * 64,
    kernel_name: str = "test_kernel",
) -> bytes:
    """Create a minimal valid AMDGPU ELF for testing."""
    # ELF header
    e_ident = ELF_MAGIC + bytes([
        ELFCLASS64,   # 64-bit
        ELFDATA2LSB,  # little-endian
        1,            # EV_CURRENT
        0,            # ELFOSABI_NONE
    ]) + b"\x00" * 8  # padding to 16 bytes

    # We'll build: .text, .strtab, .symtab, .shstrtab
    # Section names: \0.text\0.strtab\0.symtab\0.shstrtab\0
    shstrtab_data = b"\x00.text\x00.strtab\x00.symtab\x00.shstrtab\x00"
    # Offsets: .text=1, .strtab=7, .symtab=15, .shstrtab=23
    name_text = 1
    name_strtab = 7
    name_symtab = 15
    name_shstrtab = 23

    # Symbol name string table
    strtab_data = b"\x00" + kernel_name.encode() + b"\x00"
    sym_name_offset = 1

    # Symbol table entry
    sym_info = (STB_GLOBAL << 4) | STT_AMDGPU_HSA_KERNEL
    sym_entry = struct.pack("<IBBHQQ",
        sym_name_offset,  # st_name
        sym_info,         # st_info
        0,                # st_other
        1,                # st_shndx (.text)
        0,                # st_value (offset in .text)
        64,               # st_size
    )

    # Layout: ELF header | .text | strtab | symtab | shstrtab | section headers
    ehdr_size = 64
    text_offset = ehdr_size
    text_size = len(code)
    strtab_offset = text_offset + text_size
    strtab_size = len(strtab_data)
    symtab_offset = strtab_offset + strtab_size
    symtab_size = len(sym_entry)
    shstrtab_offset = symtab_offset + symtab_size
    shstrtab_size = len(shstrtab_data)
    shdr_offset = shstrtab_offset + shstrtab_size

    # Pad to 8-byte alignment
    shdr_offset = (shdr_offset + 7) & ~7

    # 5 section headers: null, .text, .strtab, .symtab, .shstrtab
    num_sections = 5
    shdr_size = SectionHeader.SIZE

    # ELF header
    elf_header = struct.pack("<16sHHIQQQIHHHHHH",
        e_ident,
        2,              # ET_EXEC
        EM_AMDGPU,
        1,              # EV_CURRENT
        0,              # e_entry
        0,              # e_phoff
        shdr_offset,    # e_shoff
        0,              # e_flags
        ehdr_size,      # e_ehsize
        0,              # e_phentsize
        0,              # e_phnum
        shdr_size,      # e_shentsize
        num_sections,   # e_shnum
        4,              # e_shstrndx (index of .shstrtab)
    )

    # Section headers
    def _sh(name, type_, flags, offset, size, link=0, info=0, addralign=1, entsize=0):
        return struct.pack("<IIQQQQIIQQ",
            name, type_, flags, 0, offset, size, link, info, addralign, entsize)

    sh_null = _sh(0, 0, 0, 0, 0)
    sh_text = _sh(name_text, SHT_PROGBITS, 0x6, text_offset, text_size)  # ALLOC|EXEC
    sh_strtab = _sh(name_strtab, SHT_STRTAB, 0, strtab_offset, strtab_size)
    sh_symtab = _sh(name_symtab, SHT_SYMTAB, 0, symtab_offset, symtab_size, link=2, entsize=Symbol.SIZE)
    sh_shstrtab = _sh(name_shstrtab, SHT_STRTAB, 0, shstrtab_offset, shstrtab_size)

    # Build the binary
    data = bytearray(shdr_offset + num_sections * shdr_size)
    data[:ehdr_size] = elf_header
    data[text_offset:text_offset + text_size] = code
    data[strtab_offset:strtab_offset + strtab_size] = strtab_data
    data[symtab_offset:symtab_offset + symtab_size] = sym_entry
    data[shstrtab_offset:shstrtab_offset + shstrtab_size] = shstrtab_data
    offset = shdr_offset
    for sh in [sh_null, sh_text, sh_strtab, sh_symtab, sh_shstrtab]:
        data[offset:offset + shdr_size] = sh
        offset += shdr_size

    return bytes(data)


class TestELFHeader:
    """Test ELF header parsing."""

    def test_parse_header(self):
        elf_data = _make_minimal_elf()
        header = ELFHeader.from_bytes(elf_data)
        assert header.ei_class == ELFCLASS64
        assert header.ei_data == ELFDATA2LSB
        assert header.e_machine == EM_AMDGPU
        assert header.e_shnum == 5

    def test_bad_magic_raises(self):
        with pytest.raises(Exception):
            parse_elf(b"NOT_ELF" + b"\x00" * 100)


class TestELFParsing:
    """Test full ELF parsing."""

    def test_parse_sections(self):
        elf_data = _make_minimal_elf()
        co = parse_elf(elf_data)
        assert len(co.sections) == 5

        section_names = [s.name for s in co.sections]
        assert ".text" in section_names
        assert ".strtab" in section_names
        assert ".symtab" in section_names

    def test_text_section(self):
        code = b"\xBE\xAD\xDE\x00" * 16  # 64 bytes
        elf_data = _make_minimal_elf(code=code)
        co = parse_elf(elf_data)
        assert co.text_section is not None
        assert co.code == code

    def test_kernel_symbols(self):
        elf_data = _make_minimal_elf(kernel_name="my_kernel")
        co = parse_elf(elf_data)
        kernels = co.kernel_symbols()
        assert len(kernels) >= 1
        assert kernels[0].name == "my_kernel"
        assert kernels[0].type == STT_AMDGPU_HSA_KERNEL

    def test_symbol_properties(self):
        elf_data = _make_minimal_elf(kernel_name="test_func")
        co = parse_elf(elf_data)
        sym = co.kernel_symbols()[0]
        assert sym.binding == STB_GLOBAL
        assert sym.st_shndx == 1  # .text section


class TestKernelDescriptor:
    """Test kernel descriptor parsing."""

    def test_parse_descriptor(self):
        # Create a descriptor with known values
        kd = KernelDescriptor(
            group_segment_fixed_size=256,
            private_segment_fixed_size=0,
            kernarg_size=64,
            kernel_code_entry_byte_offset=256,
            compute_pgm_rsrc1=0x002C0040,
            compute_pgm_rsrc2=0x00000090,
        )
        data = kd.to_bytes()
        parsed = KernelDescriptor.from_bytes(data)

        assert parsed.group_segment_fixed_size == 256
        assert parsed.private_segment_fixed_size == 0
        assert parsed.kernarg_size == 64
        assert parsed.kernel_code_entry_byte_offset == 256
        assert parsed.compute_pgm_rsrc1 == 0x002C0040
        assert parsed.compute_pgm_rsrc2 == 0x00000090

    def test_descriptor_roundtrip(self):
        kd = KernelDescriptor(
            group_segment_fixed_size=1024,
            private_segment_fixed_size=16,
            kernarg_size=128,
            kernel_code_entry_byte_offset=512,
            compute_pgm_rsrc1=0x12345678,
            compute_pgm_rsrc2=0x9ABCDEF0,
            compute_pgm_rsrc3=0x11223344,
            kernel_code_properties=0x1234,
        )
        data = kd.to_bytes()
        parsed = KernelDescriptor.from_bytes(data)

        assert parsed.group_segment_fixed_size == kd.group_segment_fixed_size
        assert parsed.private_segment_fixed_size == kd.private_segment_fixed_size
        assert parsed.kernarg_size == kd.kernarg_size
        assert parsed.kernel_code_entry_byte_offset == kd.kernel_code_entry_byte_offset
        assert parsed.compute_pgm_rsrc1 == kd.compute_pgm_rsrc1
        assert parsed.compute_pgm_rsrc2 == kd.compute_pgm_rsrc2
        assert parsed.compute_pgm_rsrc3 == kd.compute_pgm_rsrc3
        assert parsed.kernel_code_properties == kd.kernel_code_properties

    def test_descriptor_properties(self):
        kd = KernelDescriptor(
            compute_pgm_rsrc2=0x00000090,  # workgroup_id_x=1, workgroup_id_y=0, ...
            kernel_code_properties=0x08,   # enable_sgpr_kernarg_segment_ptr
        )
        assert kd.enable_sgpr_kernarg_segment_ptr is True
        assert kd.enable_sgpr_dispatch_ptr is False

    def test_insufficient_data(self):
        with pytest.raises(Exception):
            KernelDescriptor.from_bytes(b"\x00" * 10)
