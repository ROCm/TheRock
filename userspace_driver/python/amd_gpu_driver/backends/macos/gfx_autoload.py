"""gfx12 backdoor autoload — VRAM autoload buffer + IMU boot.

Replaces the PSP LOAD_IP_FW path for RLC on gfx12 (SOC22). Follows
the sequence from `gfx_v12_0.c::gfx_v12_0_rlc_backdoor_autoload_enable`:

  1. Parse `gc_12_0_1_toc.bin` to get the SOC24_FIRMWARE_ID_* →
     (offset, size) autoload layout.
  2. Allocate a region at the low end of VRAM that covers the max TOC
     extent (≥ 23 MB).
  3. memcpy each firmware payload (RLC_G, RLC_IRAM, RLC_DRAM,
     RLX6_UCODE, SDMA, PFP/ME/MEC, MES, and the TOC itself) into the
     buffer at its TOC-specified offset.
  4. Program `GFX_IMU_RLC_BOOTLOADER_ADDR_{HI,LO}` + `...SIZE` with
     the VRAM address of RLC_G_UCODE inside the autoload buffer.
  5. Stream IMU IRAM / DRAM into the IMU's SRAM via
     `GFX_IMU_I_RAM_ADDR / DATA` (and the D_ pair).
  6. Configure IMU access control.
  7. Unhalt the IMU core.
  8. Wait for `GFX_IMU_GFX_RESET_CTRL & 0x1F == 0x1F` (IMU ready).

Once IMU is alive it reads the RLC_G bytes from VRAM using the
bootloader addr we wrote, wakes RLC, and RLC then loads the rest of
the GFX firmware from the autoload buffer.
"""
from __future__ import annotations

import ctypes
import logging
import os
import struct
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ---- GC block register DWORD offsets (gc_12_0_0_offset.h) ----
# All use GC BASE_IDX=1 -> 0xA000 on gfx1201.
GC_BASE_IDX1_DW = 0xA000
_MMIO_BAR = 5

# IMU MMIO streaming
regGFX_IMU_I_RAM_ADDR          = 0x5f90
regGFX_IMU_I_RAM_DATA          = 0x5f91
regGFX_IMU_D_RAM_ADDR          = 0x40fc
regGFX_IMU_D_RAM_DATA          = 0x40fd
# IMU control
regGFX_IMU_CORE_CTRL           = 0x40b6
regGFX_IMU_GFX_RESET_CTRL      = 0x40bc
regGFX_IMU_C2PMSG_ACCESS_CTRL0 = 0x4040
regGFX_IMU_C2PMSG_ACCESS_CTRL1 = 0x4041
# RLC bootloader address (pointed at RLC_G_UCODE inside the autoload buffer)
regGFX_IMU_RLC_BOOTLOADER_ADDR_HI = 0x5f81
regGFX_IMU_RLC_BOOTLOADER_ADDR_LO = 0x5f82
regGFX_IMU_RLC_BOOTLOADER_SIZE    = 0x5f83

# SOC24 firmware IDs we care about (subset — full set in amdgpu_rlc.h).
SOC24_FIRMWARE_ID_INVALID        = 0
SOC24_FIRMWARE_ID_RLC_G_UCODE    = 1
SOC24_FIRMWARE_ID_RLC_TOC        = 2
SOC24_FIRMWARE_ID_RLCG_SCRATCH   = 3
SOC24_FIRMWARE_ID_RLC_SRM_ARAM   = 4
SOC24_FIRMWARE_ID_RLX6_UCODE     = 7
SOC24_FIRMWARE_ID_RLX6_DRAM_BOOT = 9
SOC24_FIRMWARE_ID_SDMA_UCODE_TH0 = 11
SOC24_FIRMWARE_ID_RS64_MES_P0    = 16
SOC24_FIRMWARE_ID_RS64_MES_P1    = 17
SOC24_FIRMWARE_ID_RS64_PFP       = 18
SOC24_FIRMWARE_ID_RS64_ME        = 19
SOC24_FIRMWARE_ID_RS64_MEC       = 20
SOC24_FIRMWARE_ID_RS64_MES_P0_STACK = 21
SOC24_FIRMWARE_ID_RS64_MES_P1_STACK = 22
SOC24_FIRMWARE_ID_RS64_PFP_P0_STACK = 23
SOC24_FIRMWARE_ID_RS64_PFP_P1_STACK = 24
SOC24_FIRMWARE_ID_RS64_ME_P0_STACK  = 25
SOC24_FIRMWARE_ID_RS64_ME_P1_STACK  = 26
SOC24_FIRMWARE_ID_RS64_MEC_P0_STACK = 27
SOC24_FIRMWARE_ID_RS64_MEC_P1_STACK = 28
SOC24_FIRMWARE_ID_RS64_MEC_P2_STACK = 29
SOC24_FIRMWARE_ID_RS64_MEC_P3_STACK = 30
SOC24_FIRMWARE_ID_MAX               = 43

# gfx_v12_0.c:
#   RLC_TOC_OFFSET_DWUNIT = 8     (offset field is in 8-DWORD units = 32 bytes)
#   RLC_SIZE_MULTIPLE     = 1024  (size_x16 flag → size is in 1024-DWORD = 4 KB units)
#   RLC_TOC_FORMAT_API    = 165   (patched into last TOC DWORD; see copy_toc_ucode)
RLC_TOC_OFFSET_DWUNIT = 8
RLC_SIZE_MULTIPLE     = 1024
RLC_TOC_FORMAT_API    = 165
RLC_TOC_UMF_SIZE      = 23 * 1024 * 1024  # minimum buffer size per gfx_v12_0.c


@dataclass
class TocEntry:
    id: int
    offset: int    # bytes from start of autoload buffer
    size: int      # bytes


def parse_rlc_toc(blob: bytes) -> list[TocEntry]:
    """Parse the common-header-wrapped TOC into a list of entries."""
    # common_firmware_header: u32 size, u32 hdr_sz, u16x4 versions, u32x4 ucode_info
    _size, _hdr_sz, _maj, _min, _ipmaj, _ipmin, _uver, _usz, uoff, _crc = \
        struct.unpack_from("<IIHHHHIIII", blob, 0)
    out: list[TocEntry] = []
    pos = uoff
    while pos + 8 <= len(blob):
        dw0, dw1 = struct.unpack_from("<II", blob, pos)
        # RLC_TABLE_OF_CONTENT_V2 bit layout (amdgpu_rlc.h):
        #   DW0: [0..24] offset (25b), [25..31] id (7b)
        #   DW1: [0..2] reserved, [3..4] memory_destination,
        #        [5..8] vfflr_image_code, [9..11] reserved,
        #        [12]   size_x16, [13] reserved, [14..31] size (18b)
        off25 = dw0 & ((1 << 25) - 1)
        fwid  = (dw0 >> 25) & 0x7F
        size_x16 = (dw1 >> 12) & 1
        sz18     = (dw1 >> 14) & ((1 << 18) - 1)
        if fwid == SOC24_FIRMWARE_ID_INVALID:
            break
        off_bytes = off25 * RLC_TOC_OFFSET_DWUNIT * 4
        sz_bytes  = (sz18 * RLC_SIZE_MULTIPLE * 4) if size_x16 else (sz18 * 4)
        out.append(TocEntry(id=fwid, offset=off_bytes, size=sz_bytes))
        pos += 8
    return out


@dataclass
class AutoloadLayout:
    toc_entries: list[TocEntry]
    buffer_size: int   # bytes needed at the low end of VRAM
    rlc_g_offset: int  # offset of RLC_G_UCODE within the buffer
    rlc_g_size: int


def plan_autoload(toc_blob: bytes) -> AutoloadLayout:
    entries = parse_rlc_toc(toc_blob)
    by_id = {e.id: e for e in entries}
    rlc_g = by_id[SOC24_FIRMWARE_ID_RLC_G_UCODE]
    max_end = max(e.offset + e.size for e in entries)
    buffer_size = max(max_end, RLC_TOC_UMF_SIZE)
    return AutoloadLayout(
        toc_entries=entries,
        buffer_size=buffer_size,
        rlc_g_offset=rlc_g.offset,
        rlc_g_size=rlc_g.size,
    )


# --- Sub-firmware extraction from individual .bin files ----------------

def _common_ucode_off(blob: bytes) -> int:
    return struct.unpack_from("<I", blob, 24)[0]  # ucode_array_offset_bytes field


def extract_rlc_subfw(rlc_blob: bytes) -> dict[int, bytes]:
    """From the RLC firmware blob, return {SOC24_FIRMWARE_ID: bytes}.

    Matches `gfx_v12_0_rlc_backdoor_autoload_copy_gfx_ucode` (RLC path).
    """
    def _u32(off: int) -> int:
        return struct.unpack_from("<I", rlc_blob, off)[0]

    uoff = _common_ucode_off(rlc_blob)
    usz  = struct.unpack_from("<I", rlc_blob, 20)[0]  # ucode_size_bytes
    hver_min = struct.unpack_from("<H", rlc_blob, 10)[0]  # header_version_minor

    out: dict[int, bytes] = {
        SOC24_FIRMWARE_ID_RLC_G_UCODE: rlc_blob[uoff:uoff + usz],
    }

    # v2_1: save_restore_list_cntl/gpm/srm (offsets absolute from start of file,
    # confirmed earlier against the real gfx1201 rlc.bin).
    if hver_min >= 1:
        srlg_sz, srlg_off = _u32(132), _u32(136)
        srls_sz, srls_off = _u32(148), _u32(152)
        if srlg_sz and srlg_off:
            out[SOC24_FIRMWARE_ID_RLCG_SCRATCH] = rlc_blob[srlg_off:srlg_off + srlg_sz]
        if srls_sz and srls_off:
            out[SOC24_FIRMWARE_ID_RLC_SRM_ARAM] = rlc_blob[srls_off:srls_off + srls_sz]

    # v2_2: rlc_iram / rlc_dram → RLX6_UCODE / RLX6_DRAM_BOOT
    if hver_min >= 2:
        iram_sz, iram_off = _u32(156), _u32(160)
        dram_sz, dram_off = _u32(164), _u32(168)
        if iram_sz and iram_off:
            out[SOC24_FIRMWARE_ID_RLX6_UCODE]    = rlc_blob[iram_off:iram_off + iram_sz]
        if dram_sz and dram_off:
            out[SOC24_FIRMWARE_ID_RLX6_DRAM_BOOT] = rlc_blob[dram_off:dram_off + dram_sz]

    return out


def extract_sdma(sdma_blob: bytes) -> bytes:
    """SDMA v3_0 (gfx12): common header + ucode_feature_version +
    ucode_offset_bytes + ucode_size_bytes.

    Those gfx12-specific fields live at +32, +36, +40 respectively.
    """
    u_off = struct.unpack_from("<I", sdma_blob, 36)[0]
    u_sz  = struct.unpack_from("<I", sdma_blob, 40)[0]
    return sdma_blob[u_off:u_off + u_sz]


def extract_rs64_gfx(cpv2_blob: bytes) -> tuple[bytes, bytes]:
    """gfx_firmware_header_v2_0: returns (ucode_bytes, data_bytes)."""
    # struct gfx_firmware_header_v2_0:
    #   common_firmware_header (32)
    #   +32 ucode_feature_version
    #   +36 ucode_size_bytes
    #   +40 ucode_offset_bytes
    #   +44 data_size_bytes
    #   +48 data_offset_bytes
    #   +52 ucode_start_addr_lo
    #   +56 ucode_start_addr_hi
    u_sz, u_off, d_sz, d_off = struct.unpack_from("<IIII", cpv2_blob, 36)
    return cpv2_blob[u_off:u_off + u_sz], cpv2_blob[d_off:d_off + d_sz]


def extract_mes(mes_blob: bytes) -> tuple[bytes, bytes]:
    """mes_firmware_header_v1_0: returns (ucode_bytes, data_bytes)."""
    # +32 mes_ucode_version, +36 mes_ucode_size, +40 mes_ucode_offset,
    # +44 mes_ucode_data_version, +48 mes_ucode_data_size, +52 mes_ucode_data_off
    _uver, u_sz, u_off, _dver, d_sz, d_off = struct.unpack_from("<IIIIII", mes_blob, 32)
    return mes_blob[u_off:u_off + u_sz], mes_blob[d_off:d_off + d_sz]


# --- Buffer assembly + upload -----------------------------------------

def _vram_base_cpu(client) -> tuple[int, int]:
    """Map BAR0 (VRAM, 256 MB on our card) into this process."""
    return client.map_bar(0)


def _memcpy_to_vram(vram_cpu: int, off: int, data: bytes) -> None:
    (ctypes.c_ubyte * len(data)).from_address(vram_cpu + off)[:] = data


def build_autoload_buffer(client, firmware_dir: str,
                          layout: AutoloadLayout, toc_blob: bytes) -> None:
    """Copy every firmware payload into low VRAM at TOC-specified offsets.

    Assumes BAR0 currently windows VRAM starting at offset 0, which is
    the vBIOS-POST default.
    """
    vram_cpu, vram_bar_size = _vram_base_cpu(client)
    logger.info("Autoload buffer plan: size=0x%x  VRAM BAR=%d MB",
                layout.buffer_size, vram_bar_size >> 20)
    if layout.buffer_size > vram_bar_size:
        raise RuntimeError(
            f"Autoload size 0x{layout.buffer_size:x} exceeds BAR0 window "
            f"0x{vram_bar_size:x}; implement VRAM BAR re-windowing first"
        )
    # Zero the full buffer region so untouched slots are benign.
    # ctypes.memset on the VRAM BAR mapping triggers SIGBUS on Apple
    # Silicon once the region crosses ~1 MB — but a slice-copy of the
    # same size works. Use a slice-copy of zeros.
    (ctypes.c_ubyte * layout.buffer_size).from_address(vram_cpu)[:] = \
        b"\x00" * layout.buffer_size

    by_id = {e.id: e for e in layout.toc_entries}

    # Load individual firmware files and extract sub-payloads.
    def _read(name): return open(os.path.join(firmware_dir, name), "rb").read()
    def _read_opt(name):
        p = os.path.join(firmware_dir, name)
        return open(p, "rb").read() if os.path.exists(p) else None
    rlc_blob = _read("gc_12_0_1_rlc.bin")
    # gfx1201 wants SDMA 7.0.1 (per IP discovery). 7.0.0 is the wrong
    # ASIC and SMU / IMU won't accept it.
    sdma_blob = _read_opt("sdma_7_0_1.bin")
    pfp_blob = _read("gc_12_0_1_pfp.bin")
    me_blob  = _read("gc_12_0_1_me.bin")
    mec_blob = _read("gc_12_0_1_mec.bin")
    mes_blob  = _read("gc_12_0_1_mes.bin")
    mes1_blob = _read_opt("gc_12_0_1_mes1.bin")

    def _place(fwid: int, data: bytes, label: str) -> None:
        if fwid not in by_id:
            logger.warning("  %-20s: no TOC entry (skipped)", label)
            return
        entry = by_id[fwid]
        copy_size = min(len(data), entry.size)
        if len(data) > entry.size:
            logger.warning("  %-20s: truncating %d bytes to TOC slot 0x%x",
                           label, len(data), entry.size)
        _memcpy_to_vram(vram_cpu, entry.offset, data[:copy_size])
        logger.info("  %-20s: off=0x%08x size=0x%x -> %d bytes",
                    label, entry.offset, entry.size, copy_size)

    # RLC sub-firmwares
    for fwid, data in extract_rlc_subfw(rlc_blob).items():
        _place(fwid, data, f"RLC id={fwid}")

    # SDMA (optional — not strictly needed to unblock SMU PWR_GFX, and
    # our firmware dir may not have the matching sdma binary).
    if sdma_blob is not None:
        _place(SOC24_FIRMWARE_ID_SDMA_UCODE_TH0, extract_sdma(sdma_blob), "SDMA_TH0")

    # RS64 CP (PFP, ME, MEC) — Linux copies instruction + data, and
    # duplicates the data into P0_STACK/P1_STACK (and P2/P3 for MEC).
    pfp_u, pfp_d = extract_rs64_gfx(pfp_blob)
    me_u,  me_d  = extract_rs64_gfx(me_blob)
    mec_u, mec_d = extract_rs64_gfx(mec_blob)
    _place(SOC24_FIRMWARE_ID_RS64_PFP, pfp_u, "RS64_PFP")
    _place(SOC24_FIRMWARE_ID_RS64_PFP_P0_STACK, pfp_d, "PFP_P0_STACK")
    _place(SOC24_FIRMWARE_ID_RS64_PFP_P1_STACK, pfp_d, "PFP_P1_STACK")
    _place(SOC24_FIRMWARE_ID_RS64_ME,  me_u,  "RS64_ME")
    _place(SOC24_FIRMWARE_ID_RS64_ME_P0_STACK,  me_d,  "ME_P0_STACK")
    _place(SOC24_FIRMWARE_ID_RS64_ME_P1_STACK,  me_d,  "ME_P1_STACK")
    _place(SOC24_FIRMWARE_ID_RS64_MEC, mec_u, "RS64_MEC")
    _place(SOC24_FIRMWARE_ID_RS64_MEC_P0_STACK, mec_d, "MEC_P0_STACK")
    _place(SOC24_FIRMWARE_ID_RS64_MEC_P1_STACK, mec_d, "MEC_P1_STACK")
    _place(SOC24_FIRMWARE_ID_RS64_MEC_P2_STACK, mec_d, "MEC_P2_STACK")
    _place(SOC24_FIRMWARE_ID_RS64_MEC_P3_STACK, mec_d, "MEC_P3_STACK")

    # MES pipes 0 and 1 — if mes1.bin isn't present, reuse mes for P1.
    mes_u,  mes_d  = extract_mes(mes_blob)
    mes1_u, mes1_d = extract_mes(mes1_blob) if mes1_blob else (mes_u, mes_d)
    _place(SOC24_FIRMWARE_ID_RS64_MES_P0,       mes_u,  "RS64_MES_P0")
    _place(SOC24_FIRMWARE_ID_RS64_MES_P0_STACK, mes_d,  "MES_P0_STACK")
    _place(SOC24_FIRMWARE_ID_RS64_MES_P1,       mes1_u, "RS64_MES_P1")
    _place(SOC24_FIRMWARE_ID_RS64_MES_P1_STACK, mes1_d, "MES_P1_STACK")

    # Finally, the TOC itself. gfx_v12_0_rlc_backdoor_autoload_copy_toc_ucode
    # patches the last DWORD with RLC_TOC_FORMAT_API << 24 | 0x1 before copying.
    toc_slot = by_id.get(SOC24_FIRMWARE_ID_RLC_TOC)
    if toc_slot is not None:
        toc_copy = bytearray(toc_blob)
        # Overwrite the last 4 bytes of the region PSP gave us (size of the
        # TOC slot in VRAM — not the .bin file size). The driver patches
        # (RLC_TOC_FORMAT_API << 24) | 0x1 at that position.
        if toc_slot.size >= 4:
            patched = (RLC_TOC_FORMAT_API << 24) | 0x1
            copy_size = min(len(toc_copy), toc_slot.size)
            # Pad to toc_slot.size with zeros, then patch tail.
            if copy_size < toc_slot.size:
                toc_copy += b"\x00" * (toc_slot.size - copy_size)
            struct.pack_into("<I", toc_copy, toc_slot.size - 4, patched)
            _memcpy_to_vram(vram_cpu, toc_slot.offset, bytes(toc_copy[:toc_slot.size]))
            logger.info("  %-20s: off=0x%08x size=0x%x (patched last DW = 0x%08x)",
                        "RLC_TOC", toc_slot.offset, toc_slot.size, patched)


# --- IMU boot ---------------------------------------------------------

def _gc_wr(client, dw_off: int, value: int) -> None:
    client.mmio_write32(_MMIO_BAR, (GC_BASE_IDX1_DW + dw_off) * 4, value & 0xFFFFFFFF)


def _gc_rd(client, dw_off: int) -> int:
    return client.mmio_read32(_MMIO_BAR, (GC_BASE_IDX1_DW + dw_off) * 4)


def _stream_imu(client, iram: bytes, dram: bytes) -> None:
    """Stream IMU IRAM+DRAM into IMU's internal SRAM (via MMIO)."""
    if len(iram) % 4 or len(dram) % 4:
        raise ValueError("IMU ucode not DWORD-aligned")
    _gc_wr(client, regGFX_IMU_I_RAM_ADDR, 0)
    for i in range(0, len(iram), 4):
        word = struct.unpack_from("<I", iram, i)[0]
        _gc_wr(client, regGFX_IMU_I_RAM_DATA, word)
    # Linux writes fw_version at the end to lock it down.
    _gc_wr(client, regGFX_IMU_I_RAM_ADDR, 0)
    _gc_wr(client, regGFX_IMU_D_RAM_ADDR, 0)
    for i in range(0, len(dram), 4):
        word = struct.unpack_from("<I", dram, i)[0]
        _gc_wr(client, regGFX_IMU_D_RAM_DATA, word)
    _gc_wr(client, regGFX_IMU_D_RAM_ADDR, 0)


def run_imu_boot(client, firmware_dir: str,
                 layout: AutoloadLayout,
                 imu_fw_name: str = "gc_12_0_1_imu.bin") -> None:
    """Steps 4-8 from the module docstring."""
    import time
    # (4) point IMU at the RLC_G_UCODE region inside the autoload buffer
    # (at VRAM offset 0 + rlc_g_offset, since we parked the buffer at 0).
    gpu_addr = layout.rlc_g_offset
    _gc_wr(client, regGFX_IMU_RLC_BOOTLOADER_ADDR_HI, (gpu_addr >> 32) & 0xFFFFFFFF)
    _gc_wr(client, regGFX_IMU_RLC_BOOTLOADER_ADDR_LO, gpu_addr & 0xFFFFFFFF)
    _gc_wr(client, regGFX_IMU_RLC_BOOTLOADER_SIZE,    layout.rlc_g_size)
    logger.info("RLC bootloader: addr=0x%x size=0x%x", gpu_addr, layout.rlc_g_size)

    # (5) stream IMU firmware via MMIO
    imu_blob = open(os.path.join(firmware_dir, imu_fw_name), "rb").read()
    # IMU header parsing — iram starts at ucode_array_offset_bytes,
    # dram follows iram (see amdgpu_ucode.c — the header's iram/dram
    # offset fields are ignored).
    uoff = _common_ucode_off(imu_blob)
    iram_sz = struct.unpack_from("<I", imu_blob, 32)[0]
    dram_sz = struct.unpack_from("<I", imu_blob, 40)[0]
    iram = imu_blob[uoff:uoff + iram_sz]
    dram = imu_blob[uoff + iram_sz:uoff + iram_sz + dram_sz]
    logger.info("Streaming IMU: iram=%d bytes, dram=%d bytes", len(iram), len(dram))
    _stream_imu(client, iram, dram)

    # (6) IMU access control
    _gc_wr(client, regGFX_IMU_C2PMSG_ACCESS_CTRL0, 0xFFFFFF)
    _gc_wr(client, regGFX_IMU_C2PMSG_ACCESS_CTRL1, 0xFFFF)

    # (7) Clear halt bit 0 of IMU_CORE_CTRL to start the IMU.
    core_ctrl = _gc_rd(client, regGFX_IMU_CORE_CTRL)
    _gc_wr(client, regGFX_IMU_CORE_CTRL, core_ctrl & 0xFFFFFFFE)
    logger.info("IMU_CORE_CTRL: was 0x%08x, now 0x%08x",
                core_ctrl, core_ctrl & 0xFFFFFFFE)

    # (8) wait for GFX_IMU_GFX_RESET_CTRL & 0x1F == 0x1F
    deadline = time.time() + 2.0
    last = 0
    while time.time() < deadline:
        last = _gc_rd(client, regGFX_IMU_GFX_RESET_CTRL)
        if (last & 0x1F) == 0x1F:
            logger.info("IMU ready: GFX_IMU_GFX_RESET_CTRL = 0x%08x", last)
            return
        time.sleep(0.002)
    raise TimeoutError(
        f"IMU did not reach reset-ready state (GFX_IMU_GFX_RESET_CTRL = 0x{last:08x})"
    )
