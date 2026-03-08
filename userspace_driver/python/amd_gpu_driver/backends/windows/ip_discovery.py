"""IP Discovery table parser for AMD GPUs.

The IP discovery table is a binary structure stored at a fixed location
in GPU memory (typically VRAM_SIZE - 64KB). It describes all IP blocks
present on the GPU, their versions, and base register addresses.

This parser reads the table via register escapes (or BAR2 mapping) and
produces a structured view of the GPU's IP blocks.

Reference: drivers/gpu/drm/amd/include/discovery.h in Linux amdgpu
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from enum import IntEnum


# ============================================================================
# Constants from discovery.h
# ============================================================================

BINARY_SIGNATURE = 0x28211407
DISCOVERY_TABLE_SIGNATURE = 0x53445049  # "IPDS" in LE
GC_TABLE_ID = 0x4347                    # "GC" in LE
HARVEST_TABLE_SIGNATURE = 0x56524148    # "HARV" in LE
PSP_HEADER_SIZE = 256

# Table indices in binary_header.table_list
TABLE_IP_DISCOVERY = 0
TABLE_GC = 1
TABLE_HARVEST_INFO = 2
TABLE_VCN_INFO = 3
TABLE_MALL_INFO = 4
TABLE_NPS_INFO = 5
TOTAL_TABLES = 6


# ============================================================================
# Hardware IDs from soc15_hw_ip.h
# ============================================================================

class HardwareID(IntEnum):
    MP1 = 1
    MP2 = 2
    THM = 3
    SMUIO = 4
    FUSE = 5
    CLKA = 6
    PWR = 10
    GC = 11
    UVD = 12       # VCN uses same ID
    VCN = 12
    AUDIO_AZ = 13
    ACP = 14
    DCI = 15
    DMU = 271
    DCO = 16
    DIO = 272
    XDMA = 17
    DCEAZ = 18
    DAZ = 274
    SDPMUX = 19
    NTB = 20
    VPE = 21
    IOHC = 24
    L2IMU = 28
    VCE = 32
    MMHUB = 34
    ATHUB = 35
    DBGU_NBIO = 36
    DFX = 37
    DBGU0 = 38
    DBGU1 = 39
    OSSSYS = 40    # IH (interrupt handler)
    HDP = 41
    SDMA0 = 42
    SDMA1 = 43
    ISP = 44
    DBGU_IO = 45
    DF = 46
    CLKB = 47
    FCH = 48
    DFX_DAP = 49
    PCIE = 70
    PCS = 80
    LSDMA = 91
    IOAGR = 100
    NBIF = 108
    IOAPIC = 124
    SYSTEMHUB = 128
    UMC = 150
    SATA = 168
    USB = 170
    XGMI = 200
    XGBE = 216
    MP0 = 255      # PSP


# Human-readable names for hardware IDs
HW_ID_NAMES: dict[int, str] = {
    1: "MP1 (SMU)",
    2: "MP2",
    3: "THM",
    4: "SMUIO",
    5: "FUSE",
    6: "CLKA",
    10: "PWR",
    11: "GC (Graphics/Compute)",
    12: "VCN (Video)",
    13: "AUDIO_AZ",
    14: "ACP",
    15: "DCI",
    16: "DCO",
    17: "XDMA",
    18: "DCEAZ",
    19: "SDPMUX",
    20: "NTB",
    21: "VPE",
    24: "IOHC",
    28: "L2IMU",
    32: "VCE",
    34: "MMHUB",
    35: "ATHUB",
    36: "DBGU_NBIO",
    37: "DFX",
    38: "DBGU0",
    39: "DBGU1",
    40: "OSSSYS (IH)",
    41: "HDP",
    42: "SDMA0",
    43: "SDMA1",
    44: "ISP",
    45: "DBGU_IO",
    46: "DF",
    47: "CLKB",
    48: "FCH",
    49: "DFX_DAP",
    68: "SDMA2",
    69: "SDMA3",
    70: "PCIE",
    80: "PCS",
    91: "LSDMA",
    100: "IOAGR",
    108: "NBIF",
    124: "IOAPIC",
    128: "SYSTEMHUB",
    150: "UMC",
    168: "SATA",
    170: "USB",
    200: "XGMI",
    216: "XGBE",
    255: "MP0 (PSP)",
    271: "DMU (Display)",
    272: "DIO",
    274: "DAZ",
}


# ============================================================================
# Parsed data structures
# ============================================================================

@dataclass
class IPBlock:
    """A single IP block from the discovery table."""
    hw_id: int
    hw_name: str
    instance_number: int
    major: int
    minor: int
    revision: int
    sub_revision: int
    variant: int
    harvest: int
    num_base_address: int
    base_addresses: list[int] = field(default_factory=list)
    die_id: int = 0

    @property
    def version_str(self) -> str:
        """Version string like 'v12.0.1'."""
        return f"v{self.major}.{self.minor}.{self.revision}"

    def __repr__(self) -> str:
        addrs = ", ".join(f"0x{a:08X}" for a in self.base_addresses[:3])
        if len(self.base_addresses) > 3:
            addrs += f", ... ({len(self.base_addresses)} total)"
        return (
            f"IPBlock({self.hw_name} {self.version_str} "
            f"inst={self.instance_number} bases=[{addrs}])"
        )


@dataclass
class GCInfo:
    """Graphics/Compute configuration from the GC info table."""
    version_major: int
    version_minor: int
    num_se: int = 0
    num_wgp0_per_sa: int = 0
    num_wgp1_per_sa: int = 0
    num_rb_per_se: int = 0
    num_gl2c: int = 0
    wave_size: int = 0
    max_waves_per_simd: int = 0
    lds_size: int = 0
    num_sa_per_se: int = 0
    num_sc_per_se: int = 0
    num_packer_per_sc: int = 0
    num_gl2a: int = 0
    num_tcp_per_sa: int = 0
    num_tcps: int = 0


@dataclass
class HarvestEntry:
    """A harvested (disabled) IP instance."""
    hw_id: int
    instance_number: int


@dataclass
class IPDiscoveryResult:
    """Complete parsed IP discovery data."""
    binary_version_major: int
    binary_version_minor: int
    discovery_version: int
    num_dies: int
    ip_blocks: list[IPBlock] = field(default_factory=list)
    gc_info: GCInfo | None = None
    harvested: list[HarvestEntry] = field(default_factory=list)

    def find_by_hw_id(self, hw_id: int) -> list[IPBlock]:
        """Find all IP blocks with a given hardware ID."""
        return [b for b in self.ip_blocks if b.hw_id == hw_id]

    def find_first(self, hw_id: int, instance: int = 0) -> IPBlock | None:
        """Find the first IP block with given HW ID and instance."""
        for b in self.ip_blocks:
            if b.hw_id == hw_id and b.instance_number == instance:
                return b
        return None

    def print_summary(self) -> None:
        """Print a human-readable summary of discovered IP blocks."""
        print(f"IP Discovery v{self.binary_version_major}."
              f"{self.binary_version_minor}")
        print(f"  Discovery table version: {self.discovery_version}")
        print(f"  Dies: {self.num_dies}")
        print(f"  IP blocks: {len(self.ip_blocks)}")
        print()
        for block in self.ip_blocks:
            print(f"  [{block.hw_id:3d}] {block.hw_name:25s} "
                  f"{block.version_str:8s} "
                  f"inst={block.instance_number} "
                  f"bases={len(block.base_addresses)}")
        if self.gc_info:
            gc = self.gc_info
            print(f"\nGC Info v{gc.version_major}.{gc.version_minor}:")
            print(f"  SEs={gc.num_se} SA/SE={gc.num_sa_per_se} "
                  f"WGP0/SA={gc.num_wgp0_per_sa} WGP1/SA={gc.num_wgp1_per_sa}")
            print(f"  Wave size={gc.wave_size} LDS={gc.lds_size}")
        if self.harvested:
            print(f"\nHarvested IPs: {len(self.harvested)}")
            for h in self.harvested:
                name = HW_ID_NAMES.get(h.hw_id, f"HW_{h.hw_id}")
                print(f"  {name} instance {h.instance_number}")


# ============================================================================
# Binary parsing
# ============================================================================

def _parse_table_info(data: bytes, offset: int) -> tuple[int, int, int]:
    """Parse a table_info struct. Returns (offset, checksum, size)."""
    tbl_offset, checksum, size, _padding = struct.unpack_from("<HHHH", data, offset)
    return tbl_offset, checksum, size


def _parse_ip_block_v3(data: bytes, offset: int) -> tuple[IPBlock, int]:
    """Parse an ip_v3 struct. Returns (IPBlock, bytes_consumed)."""
    hw_id, instance, num_bases, major, minor, revision, flags = \
        struct.unpack_from("<HBBBBBB", data, offset)

    sub_revision = flags & 0x0F
    variant = (flags >> 4) & 0x0F

    bases = []
    base_offset = offset + 8
    for i in range(num_bases):
        addr = struct.unpack_from("<I", data, base_offset + i * 4)[0]
        bases.append(addr)

    total_size = 8 + num_bases * 4
    hw_name = HW_ID_NAMES.get(hw_id, f"HW_{hw_id}")

    return IPBlock(
        hw_id=hw_id,
        hw_name=hw_name,
        instance_number=instance,
        major=major,
        minor=minor,
        revision=revision,
        sub_revision=sub_revision,
        variant=variant,
        harvest=0,
        num_base_address=num_bases,
        base_addresses=bases,
    ), total_size


def _parse_ip_block_v4(data: bytes, offset: int, is_64bit: bool) -> tuple[IPBlock, int]:
    """Parse an ip_v4 struct. Returns (IPBlock, bytes_consumed)."""
    hw_id, instance, num_bases, major, minor, revision, flags = \
        struct.unpack_from("<HBBBBBB", data, offset)

    sub_revision = flags & 0x0F
    variant = (flags >> 4) & 0x0F

    bases = []
    base_offset = offset + 8
    if is_64bit:
        for i in range(num_bases):
            addr = struct.unpack_from("<Q", data, base_offset + i * 8)[0]
            bases.append(addr)
        total_size = 8 + num_bases * 8
    else:
        for i in range(num_bases):
            addr = struct.unpack_from("<I", data, base_offset + i * 4)[0]
            bases.append(addr)
        total_size = 8 + num_bases * 4

    hw_name = HW_ID_NAMES.get(hw_id, f"HW_{hw_id}")

    return IPBlock(
        hw_id=hw_id,
        hw_name=hw_name,
        instance_number=instance,
        major=major,
        minor=minor,
        revision=revision,
        sub_revision=sub_revision,
        variant=variant,
        harvest=0,
        num_base_address=num_bases,
        base_addresses=bases,
    ), total_size


def _parse_gc_info(data: bytes, offset: int, size: int) -> GCInfo | None:
    """Parse the GC info table."""
    if size < 12:
        return None

    table_id, ver_major, ver_minor, tbl_size = \
        struct.unpack_from("<IHHI", data, offset)

    if table_id != GC_TABLE_ID:
        return None

    gc = GCInfo(version_major=ver_major, version_minor=ver_minor)

    # Header is 12 bytes, followed by uint32 fields
    fields_offset = offset + 12
    fields_available = (min(size, tbl_size) - 12) // 4

    field_names = [
        "num_se", "num_wgp0_per_sa", "num_wgp1_per_sa", "num_rb_per_se",
        "num_gl2c", None, None, None, None, None,  # gprs, gs_thds, gs_depth, gsprim, param_cache
        None, "wave_size", "max_waves_per_simd", None, "lds_size",  # dbl_offchip, max_scratch
        "num_sc_per_se", "num_sa_per_se", "num_packer_per_sc", "num_gl2a",
    ]

    for i, name in enumerate(field_names):
        if i >= fields_available:
            break
        if name is not None:
            val = struct.unpack_from("<I", data, fields_offset + i * 4)[0]
            setattr(gc, name, val)

    # v1.1+ has additional fields
    if ver_major == 1 and ver_minor >= 1 and fields_available > 19:
        gc.num_tcp_per_sa = struct.unpack_from("<I", data, fields_offset + 19 * 4)[0]
        if fields_available > 21:
            gc.num_tcps = struct.unpack_from("<I", data, fields_offset + 21 * 4)[0]

    return gc


def _parse_harvest_info(data: bytes, offset: int, size: int) -> list[HarvestEntry]:
    """Parse the harvest info table."""
    harvested: list[HarvestEntry] = []
    if size < 8:
        return harvested

    sig, version = struct.unpack_from("<II", data, offset)
    if sig != HARVEST_TABLE_SIGNATURE:
        return harvested

    entry_offset = offset + 8
    max_entries = (size - 8) // 4

    for i in range(min(max_entries, 32)):
        hw_id, instance, _reserved = struct.unpack_from(
            "<HBB", data, entry_offset + i * 4)
        if hw_id == 0:
            break
        harvested.append(HarvestEntry(hw_id=hw_id, instance_number=instance))

    return harvested


def parse_ip_discovery(data: bytes) -> IPDiscoveryResult:
    """Parse a complete IP discovery binary blob.

    Args:
        data: Raw bytes of the IP discovery binary (typically read from
              VRAM at VRAM_SIZE - 64KB, starting after PSP_HEADER_SIZE).

    Returns:
        IPDiscoveryResult with all parsed IP blocks and metadata.

    Raises:
        ValueError: If the binary signature is invalid.
    """
    # Skip PSP header (256 bytes) if present
    offset = 0
    if len(data) > PSP_HEADER_SIZE:
        sig_at_psp = struct.unpack_from("<I", data, PSP_HEADER_SIZE)[0]
        if sig_at_psp == BINARY_SIGNATURE:
            offset = PSP_HEADER_SIZE

    # Parse binary_header
    sig = struct.unpack_from("<I", data, offset)[0]
    if sig != BINARY_SIGNATURE:
        raise ValueError(
            f"Invalid IP discovery binary signature: 0x{sig:08X} "
            f"(expected 0x{BINARY_SIGNATURE:08X})"
        )

    ver_major, ver_minor, _checksum, _bin_size = \
        struct.unpack_from("<HHHH", data, offset + 4)

    # Parse table_list[TOTAL_TABLES]
    table_offsets = []
    for i in range(TOTAL_TABLES):
        tbl_off, tbl_checksum, tbl_size = _parse_table_info(
            data, offset + 12 + i * 8)
        table_offsets.append((tbl_off, tbl_checksum, tbl_size))

    result = IPDiscoveryResult(
        binary_version_major=ver_major,
        binary_version_minor=ver_minor,
        discovery_version=0,
        num_dies=0,
    )

    # Parse IP discovery table
    ip_tbl_offset, _, ip_tbl_size = table_offsets[TABLE_IP_DISCOVERY]
    if ip_tbl_offset > 0 and ip_tbl_size > 0:
        abs_offset = offset + ip_tbl_offset
        ip_sig, ip_version, _, _, num_dies = \
            struct.unpack_from("<IHHIH", data, abs_offset)

        result.discovery_version = ip_version
        result.num_dies = num_dies

        # Check for 64-bit base addresses (version 4+)
        is_64bit = False
        if ip_version >= 4:
            # The flag byte is after the die_info array
            flag_offset = abs_offset + 14 + num_dies * 4
            if flag_offset < len(data):
                flags_byte = data[flag_offset]
                is_64bit = bool(flags_byte & 0x01)

        # Parse each die
        for die_idx in range(min(num_dies, 16)):
            die_id, die_offset = struct.unpack_from(
                "<HH", data, abs_offset + 14 + die_idx * 4)

            # die_offset is relative to the binary header start, not the
            # IP discovery table (matches kernel amdgpu_discovery.c behavior)
            die_abs = offset + die_offset
            if die_abs + 4 > len(data):
                break

            d_die_id, d_num_ips = struct.unpack_from("<HH", data, die_abs)

            # Parse IPs in this die
            ip_offset = die_abs + 4
            for _ in range(d_num_ips):
                if ip_offset + 8 > len(data):
                    break

                if ip_version >= 4:
                    block, consumed = _parse_ip_block_v4(
                        data, ip_offset, is_64bit)
                else:
                    block, consumed = _parse_ip_block_v3(data, ip_offset)

                block.die_id = die_id
                result.ip_blocks.append(block)
                ip_offset += consumed

    # Parse GC info table
    gc_tbl_offset, _, gc_tbl_size = table_offsets[TABLE_GC]
    if gc_tbl_offset > 0 and gc_tbl_size > 0:
        gc_abs = offset + gc_tbl_offset
        result.gc_info = _parse_gc_info(data, gc_abs, gc_tbl_size)

    # Parse harvest info table
    harv_tbl_offset, _, harv_tbl_size = table_offsets[TABLE_HARVEST_INFO]
    if harv_tbl_offset > 0 and harv_tbl_size > 0:
        harv_abs = offset + harv_tbl_offset
        result.harvested = _parse_harvest_info(data, harv_abs, harv_tbl_size)

    return result


def read_discovery_table_via_mmio(
    read_fn: callable,
    vram_size: int,
    read_size: int = 65536,
) -> bytes:
    """Read the IP discovery table from GPU memory via register reads.

    This reads the discovery table using 32-bit MMIO reads, which is
    slow but works before BAR2 mapping is available.

    Args:
        read_fn: Function that takes (address: int) -> int for 32-bit reads.
                 Typically WindowsDevice.read_reg_indirect.
        vram_size: Total VRAM size in bytes.
        read_size: Number of bytes to read (default 64KB).

    Returns:
        Raw bytes of the discovery table.
    """
    # Discovery table is at VRAM_SIZE - 64KB
    base_addr = vram_size - read_size

    words = []
    for off in range(0, read_size, 4):
        val = read_fn(base_addr + off)
        words.append(struct.pack("<I", val))

    return b"".join(words)
