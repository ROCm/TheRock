"""PSP-handled gfx12 autoload — the path tinygrad actually uses.

Discovered while investigating why our manual VRAM-memcpy autoload
path ran into an unshifting GC-block write gate: **PSP itself has a
single command (`GFX_CMD_ID_AUTOLOAD_RLC = 0x21`) that tells SOS to
perform the whole backdoor autoload internally**. We feed PSP each
IP firmware via `LOAD_IP_FW`, then send `AUTOLOAD_RLC`, and PSP
handles GC write access, VRAM layout, and IMU start on its own.

Sequence (mirrors tinygrad's AM_PSP.init_hw for gfx11+):

  1. SOS alive + PSP KM ring (assumed already set up).
  2. LOAD_IP_FW(SMU) — smu_bring_up does this.
  3. TMR skip for MP0 14.0.2/3 (boot_time_tmr=True).
  4. LOAD_IP_FW for every GFX-side IP firmware piece.
  5. AUTOLOAD_RLC — PSP starts RLC backdoor autoload.
  6. Poll for RLC bootload_complete.

This module does NOT touch the GC block directly — PSP does that.
"""
from __future__ import annotations

import ctypes
import logging
import os
import struct
from dataclasses import dataclass

from amd_gpu_driver.backends.macos.psp_cmd import (
    GFX_CMD_ID_AUTOLOAD_RLC,
    GFX_FW_TYPE_IMU_D,
    GFX_FW_TYPE_IMU_I,
    GFX_FW_TYPE_RLC_G,
    GFX_FW_TYPE_RLC_IRAM,
    GFX_FW_TYPE_RLC_RESTORE_LIST_GPM_MEM,
    GFX_FW_TYPE_RLC_RESTORE_LIST_SRM_MEM,
    GFX_FW_TYPE_RS64_ME,
    GFX_FW_TYPE_RS64_ME_P0_STACK,
    GFX_FW_TYPE_RS64_ME_P1_STACK,
    GFX_FW_TYPE_RS64_MEC,
    GFX_FW_TYPE_RS64_MEC_P0_STACK,
    GFX_FW_TYPE_RS64_MEC_P1_STACK,
    GFX_FW_TYPE_RS64_MEC_P2_STACK,
    GFX_FW_TYPE_RS64_MEC_P3_STACK,
    GFX_FW_TYPE_RS64_MES,
    GFX_FW_TYPE_RS64_MES_STACK,
    GFX_FW_TYPE_RS64_PFP,
    GFX_FW_TYPE_RS64_PFP_P0_STACK,
    GFX_FW_TYPE_RS64_PFP_P1_STACK,
    GFX_FW_TYPE_SDMA_UCODE_TH0,
    PSP_GFX_CMD_BUF_SIZE,
    PSP_GFX_CMD_BUF_VERSION,
    alloc_cmd_ctx,
    submit,
    submit_ip_fw_load,
)
from amd_gpu_driver.backends.macos.psp_ring import PSPRing

logger = logging.getLogger(__name__)


# --- Firmware parsing helpers (reused from gfx_autoload/gfx_firmware) -

def _common_hdr(blob: bytes) -> tuple[int, int, int]:
    """Returns (ucode_version, ucode_size_bytes, ucode_array_offset_bytes)."""
    uver, usz = struct.unpack_from("<II", blob, 16)
    uoff = struct.unpack_from("<I", blob, 24)[0]
    return uver, usz, uoff


def _parse_rlc_subs(blob: bytes) -> list[tuple[int, bytes, str]]:
    """Return [(fw_type, payload, label), ...] for RLC main ucode + sub-fw.

    Offsets in the RLC header are absolute from start of file — matches
    what amdgpu's psp_load / rlc_backdoor_autoload_copy_ucode use.
    """
    _uver, usz, uoff = _common_hdr(blob)
    hver_min = struct.unpack_from("<H", blob, 10)[0]

    def _u32(off): return struct.unpack_from("<I", blob, off)[0]

    subs: list[tuple[int, bytes, str]] = [
        (GFX_FW_TYPE_RLC_G, blob[uoff:uoff + usz], "RLC_G"),
    ]
    if hver_min >= 1:
        # save_restore_list_cntl (SRLC) — absent in our file but handle it
        srlc_sz, srlc_off = _u32(116), _u32(120)
        srlg_sz, srlg_off = _u32(132), _u32(136)
        srls_sz, srls_off = _u32(148), _u32(152)
        if srlg_sz and srlg_off:
            subs.append((GFX_FW_TYPE_RLC_RESTORE_LIST_GPM_MEM,
                         blob[srlg_off:srlg_off + srlg_sz], "RLC_SRLG"))
        if srls_sz and srls_off:
            subs.append((GFX_FW_TYPE_RLC_RESTORE_LIST_SRM_MEM,
                         blob[srls_off:srls_off + srls_sz], "RLC_SRLS"))
    if hver_min >= 2:
        iram_sz, iram_off = _u32(156), _u32(160)
        dram_sz, dram_off = _u32(164), _u32(168)
        if iram_sz and iram_off:
            subs.append((GFX_FW_TYPE_RLC_IRAM,
                         blob[iram_off:iram_off + iram_sz], "RLC_IRAM"))
        if dram_sz and dram_off:
            # RLC_DRAM_BOOT = 48; keep here for PSP even though the constant
            # isn't in our exports — PSP takes a raw integer fw_type.
            subs.append((48, blob[dram_off:dram_off + dram_sz], "RLC_DRAM_BOOT"))
    return subs


def _extract_imu(blob: bytes) -> tuple[bytes, bytes]:
    """IMU: iram + dram payloads.

    Header's iram_offset / dram_offset fields are ignored by amdgpu;
    iram starts at ucode_array_offset_bytes and dram immediately follows.
    """
    _uver, _usz, uoff = _common_hdr(blob)
    iram_sz = struct.unpack_from("<I", blob, 32)[0]
    dram_sz = struct.unpack_from("<I", blob, 40)[0]
    return (blob[uoff:uoff + iram_sz],
            blob[uoff + iram_sz:uoff + iram_sz + dram_sz])


def _extract_rs64_gfx(blob: bytes) -> tuple[bytes, bytes]:
    """gfx_firmware_header_v2_0 → (ucode, data)."""
    u_sz, u_off, d_sz, d_off = struct.unpack_from("<IIII", blob, 36)
    return blob[u_off:u_off + u_sz], blob[d_off:d_off + d_sz]


def _extract_mes(blob: bytes) -> tuple[bytes, bytes]:
    """mes_firmware_header_v1_0 → (ucode, data)."""
    _uver, u_sz, u_off, _dver, d_sz, d_off = struct.unpack_from("<IIIIII", blob, 32)
    return blob[u_off:u_off + u_sz], blob[d_off:d_off + d_sz]


def _extract_sdma_v3(blob: bytes) -> bytes:
    """sdma_firmware_header_v3_0 — ucode_offset/size at +36/+40."""
    u_off = struct.unpack_from("<I", blob, 36)[0]
    u_sz  = struct.unpack_from("<I", blob, 40)[0]
    return blob[u_off:u_off + u_sz]


# --- Submit helpers ---------------------------------------------------

def _build_autoload_rlc_cmd(ctx) -> None:
    """Build a GFX_CMD_ID_AUTOLOAD_RLC command in ctx.cmd_cpu."""
    ctypes.memset(ctx.cmd_cpu, 0, PSP_GFX_CMD_BUF_SIZE)
    hdr = struct.pack(
        "<IIIIIII",
        PSP_GFX_CMD_BUF_SIZE,
        PSP_GFX_CMD_BUF_VERSION,
        GFX_CMD_ID_AUTOLOAD_RLC,
        0, 0, 0, 0,
    )
    (ctypes.c_ubyte * len(hdr)).from_address(ctx.cmd_cpu)[:] = hdr


def _load_one(client, driver, mp0_base_dw: int, ring: PSPRing, ctx,
              payload: bytes, fw_type: int, label: str,
              *, strict: bool = False) -> int:
    """Stage + PSP LOAD_IP_FW + free. Returns the PSP status.

    If `strict` is True, raises on non-zero. Otherwise just logs the
    status so the caller can continue probing other fw_types.
    """
    if not payload:
        logger.warning("  %-20s: empty payload, skipped", label)
        return 0
    size = (len(payload) + 0xFFF) & ~0xFFF
    cpu, bus, handle = driver.alloc_dma(size)
    try:
        ctypes.memset(cpu, 0, size)
        (ctypes.c_ubyte * len(payload)).from_address(cpu)[:] = payload
        resp = submit_ip_fw_load(client, driver, mp0_base_dw, ring, ctx,
                                 bus, len(payload), fw_type, verbose=False)
    finally:
        driver.free_dma(handle)
    status = resp["status"]
    if status == 0:
        logger.info("  %-20s: type=%-3d size=%-7d OK", label, fw_type, len(payload))
    else:
        logger.warning("  %-20s: type=%-3d size=%-7d STATUS=0x%08x",
                       label, fw_type, len(payload), status)
        if strict:
            raise RuntimeError(
                f"PSP LOAD_IP_FW({label}, type={fw_type}) returned 0x{status:08x}"
            )
    return status


def submit_autoload_rlc(client, driver, mp0_base_dw: int, ring: PSPRing, ctx,
                        *, timeout_ms: int = 30000) -> dict:
    """Send the AUTOLOAD_RLC kick. PSP will internally run the full
    backdoor autoload (program GFX_IMU_RLC_BOOTLOADER_ADDR, stream
    IMU, start IMU, wait for RLC ready). Much slower than other
    commands — tinygrad and Linux let this one take multiple seconds.
    """
    _build_autoload_rlc_cmd(ctx)
    return submit(client, driver, mp0_base_dw, ring, ctx,
                  timeout_ms=timeout_ms, verbose=True)


# --- Top-level ---------------------------------------------------------

def psp_load_gfx_and_autoload(client, driver, mp0_base_dw: int,
                              ring: PSPRing, firmware_dir: str,
                              *, ctx=None) -> None:
    """Load all non-SMU GFX firmware via PSP LOAD_IP_FW, then kick
    AUTOLOAD_RLC. Assumes SOS + ring + SMU FW are already in place.
    """
    ctx = ctx or alloc_cmd_ctx(driver)

    def _read(name):
        return open(os.path.join(firmware_dir, name), "rb").read()

    # --- Parse everything up front, before issuing any ring commands ---
    imu_blob  = _read("gc_12_0_1_imu.bin")
    rlc_blob  = _read("gc_12_0_1_rlc.bin")
    pfp_blob  = _read("gc_12_0_1_pfp.bin")
    me_blob   = _read("gc_12_0_1_me.bin")
    mec_blob  = _read("gc_12_0_1_mec.bin")
    mes_blob  = _read("gc_12_0_1_mes.bin")
    sdma_blob = _read("sdma_7_0_1.bin")

    imu_iram, imu_dram = _extract_imu(imu_blob)
    rlc_subs = _parse_rlc_subs(rlc_blob)
    pfp_u, pfp_d = _extract_rs64_gfx(pfp_blob)
    me_u,  me_d  = _extract_rs64_gfx(me_blob)
    mec_u, mec_d = _extract_rs64_gfx(mec_blob)
    mes_u, mes_d = _extract_mes(mes_blob)
    sdma_ucode   = _extract_sdma_v3(sdma_blob)

    # Probe-style list: continue on error so we map every fw_type that
    # is or isn't accepted by PSP in the current state.
    logger.info("PSP LOAD_IP_FW probe:")
    items = []
    items.append((sdma_ucode, GFX_FW_TYPE_SDMA_UCODE_TH0, "SDMA_TH0"))
    for fw_type, payload, label in rlc_subs:
        items.append((payload, fw_type, label))
    items.append((pfp_u, GFX_FW_TYPE_RS64_PFP,          "RS64_PFP"))
    items.append((pfp_d, GFX_FW_TYPE_RS64_PFP_P0_STACK, "RS64_PFP_P0"))
    items.append((pfp_d, GFX_FW_TYPE_RS64_PFP_P1_STACK, "RS64_PFP_P1"))
    items.append((me_u,  GFX_FW_TYPE_RS64_ME,           "RS64_ME"))
    items.append((me_d,  GFX_FW_TYPE_RS64_ME_P0_STACK,  "RS64_ME_P0"))
    items.append((me_d,  GFX_FW_TYPE_RS64_ME_P1_STACK,  "RS64_ME_P1"))
    items.append((mec_u, GFX_FW_TYPE_RS64_MEC,          "RS64_MEC"))
    items.append((mec_d, GFX_FW_TYPE_RS64_MEC_P0_STACK, "RS64_MEC_P0"))
    items.append((mec_d, GFX_FW_TYPE_RS64_MEC_P1_STACK, "RS64_MEC_P1"))
    items.append((mec_d, GFX_FW_TYPE_RS64_MEC_P2_STACK, "RS64_MEC_P2"))
    items.append((mec_d, GFX_FW_TYPE_RS64_MEC_P3_STACK, "RS64_MEC_P3"))
    items.append((mes_u, GFX_FW_TYPE_RS64_MES,          "RS64_MES"))
    items.append((mes_d, GFX_FW_TYPE_RS64_MES_STACK,    "RS64_MES_STACK"))
    items.append((imu_iram, GFX_FW_TYPE_IMU_I, "IMU_I"))
    items.append((imu_dram, GFX_FW_TYPE_IMU_D, "IMU_D"))
    results = []
    for payload, fw_type, label in items:
        status = _load_one(client, driver, mp0_base_dw, ring, ctx,
                           payload, fw_type, label, strict=False)
        results.append((label, fw_type, status))

    oks    = [(l, t) for l, t, s in results if s == 0]
    fails  = [(l, t, s) for l, t, s in results if s != 0]
    logger.info("Summary: %d OK, %d failed", len(oks), len(fails))
    for label, fw_type, status in fails:
        logger.info("  FAILED type=%-3d %-20s status=0x%08x", fw_type, label, status)

    if fails:
        raise RuntimeError(
            f"{len(fails)} firmware type(s) rejected by PSP — cannot proceed to AUTOLOAD_RLC"
        )

    logger.info("All firmware loaded via PSP; sending AUTOLOAD_RLC...")
    resp = submit_autoload_rlc(client, driver, mp0_base_dw, ring, ctx)
    if resp["status"] != 0:
        raise RuntimeError(
            f"GFX_CMD_ID_AUTOLOAD_RLC returned status 0x{resp['status']:08x}"
        )
    logger.info("AUTOLOAD_RLC OK — PSP is now driving backdoor autoload.")
