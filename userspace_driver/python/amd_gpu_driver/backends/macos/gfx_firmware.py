"""GFX-engine firmware loading via PSP for gfx1201 (gc_12_0_1).

Status (2026-04-20):
  - IMU via PSP LOAD_IP_FW works (types 68/69).
  - RLC_G via PSP LOAD_IP_FW FAILS with status 0xFFFF0000 on gfx12.

gfx12 migrated RLC (and most GFX firmware) to a "backdoor autoload"
mechanism: the driver allocates a VRAM autoload buffer, copies each
firmware blob into it at TOC-specified offsets, and triggers RLC
self-load via MMIO. See `gfx_v12_0.c::gfx_v12_0_rlc_backdoor_autoload_*`
and `SOC24_FIRMWARE_ID_*`. That is the path we need next — this
module's RLC loader is retained for reference / gfx11 parity but is
not usable on gfx1201.

What **does** work here:
  - IMU (gc_12_0_1_imu.bin)  — IRAM + DRAM halves via PSP LOAD_IP_FW.

Each sub-firmware is copied into a DMA-mapped buffer, and a PSP
`LOAD_IP_FW` command is submitted for each. All sub-firmwares share
the same PSP cmd ctx but each gets its own DMA staging buffer so we
don't stomp while PSP is still processing a prior one.
"""
from __future__ import annotations

import ctypes
import logging
import struct
from dataclasses import dataclass

from amd_gpu_driver.backends.macos.psp_cmd import (
    GFX_FW_TYPE_GLOBAL_TAP_DELAYS,
    GFX_FW_TYPE_IMU_D,
    GFX_FW_TYPE_IMU_I,
    GFX_FW_TYPE_RLC_G,
    GFX_FW_TYPE_RLC_IRAM,
    GFX_FW_TYPE_RLC_P,
    GFX_FW_TYPE_RLC_RESTORE_LIST_GPM_MEM,
    GFX_FW_TYPE_RLC_RESTORE_LIST_SRM_CNTL,
    GFX_FW_TYPE_RLC_RESTORE_LIST_SRM_MEM,
    GFX_FW_TYPE_RLC_V,
    GFX_FW_TYPE_SE0_TAP_DELAYS,
    GFX_FW_TYPE_SE1_TAP_DELAYS,
    GFX_FW_TYPE_SE2_TAP_DELAYS,
    GFX_FW_TYPE_SE3_TAP_DELAYS,
    alloc_cmd_ctx,
    submit_ip_fw_load,
)
from amd_gpu_driver.backends.macos.psp_ring import PSPRing

logger = logging.getLogger(__name__)


# ---- Sub-firmware slice to upload ----
@dataclass
class _SubFw:
    name: str
    fw_type: int
    data: bytes


# --- common_firmware_header parse (shared with SMU / PSP loaders) ---

def _parse_common_header(blob: bytes) -> dict:
    (size_bytes, header_size_bytes, hver_maj, hver_min,
     ipver_maj, ipver_min, ucode_version, ucode_size_bytes,
     ucode_array_offset_bytes, _crc32) = struct.unpack_from("<IIHHHHIIII", blob, 0)
    return {
        "size": size_bytes,
        "header_size": header_size_bytes,
        "hver": (hver_maj, hver_min),
        "ipver": (ipver_maj, ipver_min),
        "ucode_version": ucode_version,
        "ucode_size": ucode_size_bytes,
        "ucode_off": ucode_array_offset_bytes,
    }


# --- Header parsers for the two file formats we handle here ---

def parse_imu(blob: bytes) -> list[_SubFw]:
    """imu_firmware_header_v1_0: common header + {iram,dram} size/offset.

    IMU's iram_offset / dram_offset are **relative to
    ucode_array_offset_bytes**, unlike RLC where they're absolute. See
    `amdgpu_ucode.c::amdgpu_ucode_init_ucode_addr` (the IMU_I/IMU_D
    cases compute `fw->data + ucode_array_offset_bytes` and the DRAM
    pointer is then + iram_size). amdgpu actually ignores the
    imu_{iram,dram}_offset_bytes fields in the header entirely.
    """
    hdr = _parse_common_header(blob)
    iram_size, _iram_off, dram_size, _dram_off = struct.unpack_from("<IIII", blob, 32)
    base = hdr["ucode_off"]
    iram_start = base
    dram_start = base + iram_size
    return [
        _SubFw("IMU_I", GFX_FW_TYPE_IMU_I, blob[iram_start:iram_start + iram_size]),
        _SubFw("IMU_D", GFX_FW_TYPE_IMU_D, blob[dram_start:dram_start + dram_size]),
    ]


def parse_rlc(blob: bytes) -> list[_SubFw]:
    """rlc_firmware_header_v2_0..v2_4 — main ucode plus ≤10 sub-firmwares.

    We read each versioned extension conditionally based on hver_min,
    so older/newer firmware variants don't crash this parser.
    """
    hdr = _parse_common_header(blob)
    ucode_off, ucode_size = hdr["ucode_off"], hdr["ucode_size"]
    hver_min = hdr["hver"][1]  # e.g. v2_4 -> 4
    subs: list[_SubFw] = [
        _SubFw("RLC_G", GFX_FW_TYPE_RLC_G, blob[ucode_off:ucode_off + ucode_size]),
    ]

    # Common header = 32 bytes. v2_0 adds 18 u32s (72 bytes) → ends at 104.
    # v2_1 adds 13 u32s (52 bytes) → ends at 156.
    # v2_2 adds 4  u32s (16 bytes) → ends at 172.
    # v2_3 adds 8  u32s (32 bytes) → ends at 204.
    # v2_4 adds 10 u32s (40 bytes) → ends at 244.
    def _u32(off: int) -> int:
        return struct.unpack_from("<I", blob, off)[0]

    # v2_1 sub-firmware descriptors (each is ucode_ver+feature_ver+size+offset).
    if hver_min >= 1:
        # save_restore_list_cntl: 108 ucode_ver, 112 feature_ver, 116 size, 120 offset
        srlc_size, srlc_off = _u32(116), _u32(120)
        # save_restore_list_gpm: 124 ucode_ver, 128 feature_ver, 132 size, 136 offset
        srlg_size, srlg_off = _u32(132), _u32(136)
        # save_restore_list_srm: 140 ucode_ver, 144 feature_ver, 148 size, 152 offset
        srls_size, srls_off = _u32(148), _u32(152)
        for name, t, sz, off in [
            ("RLC_SRLC", GFX_FW_TYPE_RLC_RESTORE_LIST_SRM_CNTL, srlc_size, srlc_off),
            ("RLC_SRLG", GFX_FW_TYPE_RLC_RESTORE_LIST_GPM_MEM,  srlg_size, srlg_off),
            ("RLC_SRLS", GFX_FW_TYPE_RLC_RESTORE_LIST_SRM_MEM,  srls_size, srls_off),
        ]:
            if sz and off:
                subs.append(_SubFw(name, t, blob[off:off + sz]))

    # v2_2: rlc_iram_{size,off}, rlc_dram_{size,off}
    if hver_min >= 2:
        iram_size, iram_off = _u32(156), _u32(160)
        dram_size, dram_off = _u32(164), _u32(168)
        if iram_size and iram_off:
            subs.append(_SubFw("RLC_IRAM", GFX_FW_TYPE_RLC_IRAM, blob[iram_off:iram_off + iram_size]))
        # GFX_FW_TYPE_RLC_DRAM_BOOT = 48 (not exported in psp_cmd yet).
        if dram_size and dram_off:
            subs.append(_SubFw("RLC_DRAM_BOOT", 48, blob[dram_off:dram_off + dram_size]))

    # v2_3: rlcp (ver,feat,size,off), rlcv (ver,feat,size,off)
    if hver_min >= 3:
        rlcp_size, rlcp_off = _u32(180), _u32(184)
        rlcv_size, rlcv_off = _u32(196), _u32(200)
        if rlcp_size and rlcp_off:
            subs.append(_SubFw("RLC_P", GFX_FW_TYPE_RLC_P, blob[rlcp_off:rlcp_off + rlcp_size]))
        if rlcv_size and rlcv_off:
            subs.append(_SubFw("RLC_V", GFX_FW_TYPE_RLC_V, blob[rlcv_off:rlcv_off + rlcv_size]))

    # v2_4: 5x tap-delay sections (size, offset pairs) starting at 204.
    if hver_min >= 4:
        pos = 204
        for name, fw_type in [
            ("GLOBAL_TAP", GFX_FW_TYPE_GLOBAL_TAP_DELAYS),
            ("SE0_TAP",    GFX_FW_TYPE_SE0_TAP_DELAYS),
            ("SE1_TAP",    GFX_FW_TYPE_SE1_TAP_DELAYS),
            ("SE2_TAP",    GFX_FW_TYPE_SE2_TAP_DELAYS),
            ("SE3_TAP",    GFX_FW_TYPE_SE3_TAP_DELAYS),
        ]:
            sz, dataoff = _u32(pos), _u32(pos + 4)
            pos += 8
            if sz and dataoff:
                subs.append(_SubFw(name, fw_type, blob[dataoff:dataoff + sz]))

    return subs


# --- Upload ---

def _load_subfw(client, driver, mp0_base_dw: int, ring: PSPRing, ctx,
                sub: _SubFw) -> None:
    """Stage one sub-firmware into DMA and call PSP LOAD_IP_FW for it."""
    if not sub.data:
        raise ValueError(f"{sub.name}: empty payload")
    # Page-align the staging buffer so the padded tail is zeroed.
    size = (len(sub.data) + 0xFFF) & ~0xFFF
    cpu, bus, handle = driver.alloc_dma(size)
    try:
        ctypes.memset(cpu, 0, size)
        (ctypes.c_ubyte * len(sub.data)).from_address(cpu)[:] = sub.data
        resp = submit_ip_fw_load(client, driver, mp0_base_dw, ring, ctx,
                                 bus, len(sub.data), sub.fw_type, verbose=False)
    finally:
        # PSP has (supposedly) consumed the data by now — the response is
        # back. Free the buffer to keep the DMA allocator from bloating.
        driver.free_dma(handle)
    if resp["status"] != 0:
        raise RuntimeError(
            f"PSP LOAD_IP_FW({sub.name}, type={sub.fw_type}) failed: "
            f"status=0x{resp['status']:08x}"
        )
    logger.info("  loaded %-14s (type=%2d, %d bytes) OK",
                sub.name, sub.fw_type, len(sub.data))


def load_gfx_firmware(client, driver, mp0_base_dw: int, ring: PSPRing,
                      firmware_dir: str,
                      *, imu_fw: str = "gc_12_0_1_imu.bin",
                      rlc_fw: str = "gc_12_0_1_rlc.bin") -> None:
    """Load IMU + all RLC sub-firmwares via PSP."""
    import os
    ctx = alloc_cmd_ctx(driver)

    logger.info("Loading IMU firmware from %s", imu_fw)
    with open(os.path.join(firmware_dir, imu_fw), "rb") as f:
        imu_blob = f.read()
    for sub in parse_imu(imu_blob):
        _load_subfw(client, driver, mp0_base_dw, ring, ctx, sub)

    logger.info("Loading RLC firmware from %s", rlc_fw)
    with open(os.path.join(firmware_dir, rlc_fw), "rb") as f:
        rlc_blob = f.read()
    subs = parse_rlc(rlc_blob)
    logger.info("RLC container has %d sub-firmwares", len(subs))
    for sub in subs:
        _load_subfw(client, driver, mp0_base_dw, ring, ctx, sub)
