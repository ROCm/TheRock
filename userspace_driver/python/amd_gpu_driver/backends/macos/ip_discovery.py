"""macOS-side IP discovery via amdgpu's MM_INDEX/MM_DATA indirect VRAM aperture.

The IP discovery binary on modern AMD GPUs sits at `vram_size - 64KB`, far beyond
the 256 MB BAR0 window macOS gives us over Thunderbolt. This module reads it the
same way `drivers/gpu/drm/amd/amdgpu/amdgpu_device.c::amdgpu_device_mm_access()`
does: window VRAM 4 bytes at a time through 3 fixed BAR5 registers.

Struct layout mirrors `drivers/gpu/drm/amd/include/discovery.h` (pragma pack(1)).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field


# BAR5 absolute DWORD offsets — chip-independent. From amdgpu_discovery.c L138-144.
MM_INDEX    = 0x0
MM_INDEX_HI = 0x6
MM_DATA     = 0x1
RCC_CONFIG_MEMSIZE    = 0xDE3
MP0_SMN_C2PMSG_33     = 0x16061
IP_DISCOVERY_VERSION  = 0x16A00

BINARY_SIGNATURE         = 0x28211407
DISCOVERY_TABLE_SIGNATURE = 0x53445049  # "IPDS"
DISCOVERY_TMR_OFFSET_BYTES = 0x10000    # 64 KB


# HW_ID mapping as used in the IP discovery binary.
# Names match the Linux amdgpu enums where recognizable.
HW_NAMES: dict[int, str] = {
    1: "MP1", 2: "MP2", 3: "THM", 4: "SMUIO", 5: "FUSE", 6: "CLKA",
    9: "NBIF", 10: "PWR", 11: "GC",
    12: "VCN", 13: "AUD", 14: "ACP", 15: "DCI", 16: "DMU", 17: "DCO",
    18: "DIO", 19: "VCE", 20: "UVD1", 21: "VPE",
    22: "DCE", 26: "XDMA", 27: "SDMA1", 29: "ISP", 30: "DBGU_IO",
    32: "DF", 33: "CLKB", 34: "MMHUB", 35: "MP0_SUB",
    40: "OSSSYS", 41: "NBIF41", 42: "SDMA0", 43: "LSDMA",
    45: "HDP", 46: "DBGU_NBIO", 50: "DCN", 51: "PHY", 52: "UMC",
    78: "IOHUB", 80: "ATHUB", 108: "NBIO",
    255: "MP0",
}


@dataclass
class IPBlock:
    """One IP block from the discovery table."""
    hw_id: int
    name: str
    instance: int
    major: int
    minor: int
    revision: int
    sub_revision: int
    variant: int
    bases: list[int] = field(default_factory=list)


@dataclass
class IPDiscoveryResult:
    """Parsed discovery binary."""
    binary_version_major: int
    binary_version_minor: int
    discovery_version: int
    num_dies: int
    ip_blocks: list[IPBlock] = field(default_factory=list)
    raw_bytes: bytes = b""

    def find(self, name_or_id, instance: int = 0) -> IPBlock | None:
        """Lookup an IP block by (hw_id or name) + instance number."""
        for blk in self.ip_blocks:
            if blk.instance != instance:
                continue
            if isinstance(name_or_id, int) and blk.hw_id == name_or_id:
                return blk
            if isinstance(name_or_id, str) and blk.name == name_or_id:
                return blk
        return None


def wait_psp_ready(client, timeout_ms: int = 2000) -> bool:
    """Poll MP0_SMN_C2PMSG_33 for PSP IFWI ready (bit 31).

    amdgpu_discovery.c L280-285: waits up to 2 seconds after device power-up.
    """
    import time
    deadline = time.time() + timeout_ms / 1000
    while time.time() < deadline:
        v = client.mmio_read32(5, MP0_SMN_C2PMSG_33 * 4)
        if v & 0x80000000:
            return True
        time.sleep(0.001)
    return False


def vram_read(client, pos: int, size: int) -> bytes:
    """Read `size` bytes from VRAM starting at byte offset `pos`, via MM_INDEX/MM_DATA.

    Matches `amdgpu_device_mm_access()` — works without Resizable BAR, reads
    all of VRAM through a 3-register window in BAR5.
    """
    out = bytearray()
    last_hi = -1
    for off in range(0, size, 4):
        p = pos + off
        hi = p >> 31
        client.mmio_write32(5, MM_INDEX * 4, (p & 0xFFFFFFFF) | 0x80000000)
        if hi != last_hi:
            client.mmio_write32(5, MM_INDEX_HI * 4, hi)
            last_hi = hi
        out += struct.pack("<I", client.mmio_read32(5, MM_DATA * 4))
    return bytes(out)


def read_discovery(client) -> IPDiscoveryResult:
    """Read + parse the IP discovery table from VRAM.

    Steps (mirrors amdgpu_discovery.c::amdgpu_discovery_init):
      1. Poll MP0_SMN_C2PMSG_33 for bit 31 (PSP IFWI ready).
      2. Read RCC_CONFIG_MEMSIZE (in MB) from absolute BAR5 DWORD 0xDE3.
      3. Compute discovery offset = (memsize << 20) - 64KB.
      4. Read 64 KB via MM_INDEX/MM_DATA.
      5. Parse binary_header + ip_discovery_header + die_info[] + ip_v3/v4[].
    """
    if not wait_psp_ready(client):
        raise RuntimeError("PSP IFWI not ready (MP0_SMN_C2PMSG_33 bit 31 not set)")

    memsize_mb = client.mmio_read32(5, RCC_CONFIG_MEMSIZE * 4)
    if memsize_mb in (0, 0xFFFFFFFF):
        raise RuntimeError(f"invalid VRAM size: {memsize_mb}")
    discovery_offset = (memsize_mb << 20) - DISCOVERY_TMR_OFFSET_BYTES

    data = vram_read(client, discovery_offset, 65536)
    return parse_discovery(data)


def parse_discovery(data: bytes) -> IPDiscoveryResult:
    """Parse a 64 KB discovery binary blob."""
    sig, vmaj, vmin, cksum, bsize = struct.unpack_from("<IHHHH", data, 0)
    if sig != BINARY_SIGNATURE:
        raise ValueError(f"bad binary_signature 0x{sig:x}")

    # table_list[6] follows at offset 12
    # Each table_info: uint16 offset + uint16 checksum + uint16 size + uint16 padding = 8 bytes
    ipd_table_off, _, _, _ = struct.unpack_from("<HHHH", data, 12 + 0 * 8)

    # ip_discovery_header: uint32 sig + uint16 ver + uint16 size + uint32 id +
    #                     uint16 num_dies = 14 bytes before die_info[] array
    ipd_sig, ipd_ver, ipd_size, ipd_id, num_dies = struct.unpack_from(
        "<IHHIH", data, ipd_table_off)
    if ipd_sig != DISCOVERY_TABLE_SIGNATURE:
        raise ValueError(f"bad IPDS signature 0x{ipd_sig:x}")

    # die_info array (up to 16 entries × 4 bytes each).
    result = IPDiscoveryResult(
        binary_version_major=vmaj,
        binary_version_minor=vmin,
        discovery_version=ipd_ver,
        num_dies=num_dies,
        raw_bytes=data[:bsize] if bsize > 0 else data,
    )

    for d in range(num_dies):
        _die_id, die_offset = struct.unpack_from("<HH", data, ipd_table_off + 14 + d * 4)
        # die_header: uint16 die_id + uint16 num_ips = 4 bytes
        _dh_id, num_ips = struct.unpack_from("<HH", data, die_offset)

        cur = die_offset + 4
        for _ in range(num_ips):
            # ip_v3 / ip_v4 layout (packed, 8 bytes before base_address[]):
            #   uint16 hw_id, uint8 instance, uint8 num_base, uint8 major,
            #   uint8 minor, uint8 revision, uint8 (sub_rev:4 | variant:4)
            hw_id, inst, n_base, maj, mn, rev, sub_var = struct.unpack_from(
                "<HBBBBBB", data, cur)
            sub_rev = sub_var & 0x0F
            variant = (sub_var >> 4) & 0x0F

            # base_address[n_base] — uint32 for ver 3, uint32 or uint64 for ver 4.
            # (ip_v4 64-bit base support not yet hit in the wild for gfx1201;
            # add when needed.)
            bases = list(struct.unpack_from(f"<{n_base}I", data, cur + 8))

            result.ip_blocks.append(IPBlock(
                hw_id=hw_id,
                name=HW_NAMES.get(hw_id, f"HW{hw_id}"),
                instance=inst,
                major=maj, minor=mn, revision=rev,
                sub_revision=sub_rev, variant=variant,
                bases=bases,
            ))

            cur += 8 + n_base * 4

    return result
