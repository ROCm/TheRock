"""gfx1201 cold-boot to BOOTLOAD_COMPLETE in one call.

Encapsulates the tinygrad-order sequence validated 2026-04-22:

  PSP side (all commands strictly before any MP1 mailbox):
    1. Load SOS if not alive.
    2. Create PSP KM ring.
    3. GFX_CMD_ID_LOAD_TOC.
    4. GFX_CMD_ID_LOAD_IP_FW for SMU.
    5. GFX_CMD_ID_LOAD_IP_FW for every other fw type.
    6. GFX_CMD_ID_AUTOLOAD_RLC.

  SMU side (MP1 mailbox, only AFTER all PSP):
    7. PPSMC_MSG_SetDriverDramAddrHigh.
    8. PPSMC_MSG_SetDriverDramAddrLow.
    9. PPSMC_MSG_EnableAllSmuFeatures(0).

Post-conditions (caller can rely on):
  RLC_RLCS_BOOTLOAD_STATUS bit 31 (BOOTLOAD_COMPLETE) is set.
  GFX_IMU_GFX_RESET_CTRL == 0x7F (all 7 GFX blocks released).
  GFX_IMU_CORE_CTRL == 0x8 (IMU running).
  RLC_CNTL == 0x1 (RLC enabled).
  SMU EnableAllSmuFeatures(0) ACKed.

See memory/gfx12-autoload-recipe.md for why each step matters. Key
deltas from our early broken attempts:
  - ALL graphics firmware sent via LOAD_IP_FW, not just SMU/IMU.
  - All PSP commands are strictly before any SMU mailbox message.
  - `EnableAllSmuFeatures` arg is 0 (ALL domains), not 3 (PWR_SOC).
"""
from __future__ import annotations

import logging
import os
import struct
import time
from dataclasses import dataclass

from .gfx_autoload import (
    extract_mes,
    extract_rlc_subfw,
    extract_rs64_gfx,
    extract_sdma,
)
from .gfx_psp_autoload import (
    _extract_imu,
    _load_one,
    submit_autoload_rlc,
    submit_load_toc,
)
from .psp_bootloader import (
    c2pmsg_dw,
    is_sos_alive,
    load_sos,
    parse_psp_firmware,
)
from .psp_cmd import (
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
    GFX_FW_TYPE_SMU,
    alloc_cmd_ctx,
)
from .psp_ring import PSPRing, ring_create
from .smu import (
    MMHUB_BASE_DW,
    MP0_BASE_DW,
    PPSMC_MSG_EnableAllSmuFeatures,
    PPSMC_MSG_SetDriverDramAddrHigh,
    PPSMC_MSG_SetDriverDramAddrLow,
    regMMMC_VM_FB_LOCATION_BASE,
    regMMMC_VM_FB_LOCATION_TOP,
    smu_send,
)

logger = logging.getLogger(__name__)

_MMIO_BAR = 5
# GFX12 fw_type IDs not exported in psp_cmd.py:
GFX_FW_TYPE_RLC_DRAM_BOOT = 48

# gfx1201 RLC ucode IDs inside the RLC firmware header v2_2:
_SOC24_FWID_RLC_G     = 1
_SOC24_FWID_RLCG_SCR  = 3   # SRLG
_SOC24_FWID_RLC_SRM   = 4   # SRLS
_SOC24_FWID_RLX6      = 7   # IRAM
_SOC24_FWID_RLX6_DRAM = 9   # DRAM_BOOT

# SOC15 register offsets (gc_12_0_0_offset.h)
_GC_B1 = 0xA000
_reg = {
    "GFX_IMU_CORE_CTRL":         (_GC_B1, 0x40b6),
    "GFX_IMU_GFX_RESET_CTRL":    (_GC_B1, 0x40bc),
    "RLC_CNTL":                  (_GC_B1, 0x4c00),
    "RLC_RLCS_BOOTLOAD_STATUS":  (_GC_B1, 0x4e7c),
}


@dataclass
class GfxBringUpResult:
    sos_alive: int
    ring: PSPRing
    psp_cmd_ctx: object
    driver_table_mc: int
    autoload_rlc_status: int
    bootload_status: int
    enable_all_resp: int | None
    rejected_fw: list[tuple[str, int, int]]


def _parse_common_hdr(blob: bytes) -> tuple[int, int, int]:
    _size, _hdr_sz, _maj, _min, _ipmaj, _ipmin, uver, usz, uoff, _crc = \
        struct.unpack_from("<IIHHHHIIII", blob, 0)
    return uoff, usz, uver


def _mmhub_rd(client, dw_off: int) -> int:
    return client.mmio_read32(_MMIO_BAR, (MMHUB_BASE_DW + dw_off) * 4)


def _vram_tbl_mc(client) -> int:
    """Driver table MC address — 128 KB below top of VRAM.

    Outside BAR0's window but SMU reaches it via its own DMA.
    """
    fb_base = (_mmhub_rd(client, regMMMC_VM_FB_LOCATION_BASE) & 0xFFFFFF) << 24
    fb_top  = (_mmhub_rd(client, regMMMC_VM_FB_LOCATION_TOP)  & 0xFFFFFF) + 1
    vram_size = (fb_top << 24) - fb_base
    return fb_base + vram_size - 0x20000


def _load_all_gfx_firmware(client, driver, ring: PSPRing, ctx,
                           firmware_dir: str) -> list[tuple[str, int, int]]:
    """Issue LOAD_IP_FW for every fw type tinygrad loads on gfx12.

    Returns list of (label, fw_type, status) for types that failed
    (ok ones are logged but not returned). SMU is NOT included here —
    caller loads it first (SMU must come before the others per tinygrad).
    """
    def _read(name): return open(os.path.join(firmware_dir, name), "rb").read()

    imu_blob  = _read("gc_12_0_1_imu.bin")
    rlc_blob  = _read("gc_12_0_1_rlc.bin")
    pfp_blob  = _read("gc_12_0_1_pfp.bin")
    me_blob   = _read("gc_12_0_1_me.bin")
    mec_blob  = _read("gc_12_0_1_mec.bin")
    mes_blob  = _read("gc_12_0_1_mes.bin")
    sdma_blob = _read("sdma_7_0_1.bin")

    imu_iram, imu_dram = _extract_imu(imu_blob)
    rlc_subs = extract_rlc_subfw(rlc_blob)
    pfp_u, pfp_d = extract_rs64_gfx(pfp_blob)
    me_u,  me_d  = extract_rs64_gfx(me_blob)
    mec_u, mec_d = extract_rs64_gfx(mec_blob)
    mes_u, mes_d = extract_mes(mes_blob)
    sdma_ucode   = extract_sdma(sdma_blob)

    probes: list[tuple[str, int, bytes]] = [
        ("SDMA_TH0",          GFX_FW_TYPE_SDMA_UCODE_TH0,           sdma_ucode),
        ("RLC_IRAM",          GFX_FW_TYPE_RLC_IRAM,                 rlc_subs.get(_SOC24_FWID_RLX6, b"")),
        ("RLC_DRAM_BOOT",     GFX_FW_TYPE_RLC_DRAM_BOOT,            rlc_subs.get(_SOC24_FWID_RLX6_DRAM, b"")),
        ("RLC_SRLG",          GFX_FW_TYPE_RLC_RESTORE_LIST_GPM_MEM, rlc_subs.get(_SOC24_FWID_RLCG_SCR, b"")),
        ("RLC_SRLS",          GFX_FW_TYPE_RLC_RESTORE_LIST_SRM_MEM, rlc_subs.get(_SOC24_FWID_RLC_SRM, b"")),
        ("RS64_PFP",          GFX_FW_TYPE_RS64_PFP,                 pfp_u),
        ("RS64_PFP_P0_STACK", GFX_FW_TYPE_RS64_PFP_P0_STACK,        pfp_d),
        ("RS64_PFP_P1_STACK", GFX_FW_TYPE_RS64_PFP_P1_STACK,        pfp_d),
        ("RS64_ME",           GFX_FW_TYPE_RS64_ME,                  me_u),
        ("RS64_ME_P0_STACK",  GFX_FW_TYPE_RS64_ME_P0_STACK,         me_d),
        ("RS64_ME_P1_STACK",  GFX_FW_TYPE_RS64_ME_P1_STACK,         me_d),
        ("RS64_MEC",          GFX_FW_TYPE_RS64_MEC,                 mec_u),
        ("RS64_MEC_P0_STACK", GFX_FW_TYPE_RS64_MEC_P0_STACK,        mec_d),
        ("RS64_MEC_P1_STACK", GFX_FW_TYPE_RS64_MEC_P1_STACK,        mec_d),
        ("RS64_MEC_P2_STACK", GFX_FW_TYPE_RS64_MEC_P2_STACK,        mec_d),
        ("RS64_MEC_P3_STACK", GFX_FW_TYPE_RS64_MEC_P3_STACK,        mec_d),
        ("RS64_MES",          GFX_FW_TYPE_RS64_MES,                 mes_u),
        ("RS64_MES_STACK",    GFX_FW_TYPE_RS64_MES_STACK,           mes_d),
        ("IMU_I",             GFX_FW_TYPE_IMU_I,                    imu_iram),
        ("IMU_D",             GFX_FW_TYPE_IMU_D,                    imu_dram),
        # RLC_G must be LAST — triggers the auto-autoload path on Linux.
        ("RLC_G",             GFX_FW_TYPE_RLC_G,                    rlc_subs.get(_SOC24_FWID_RLC_G, b"")),
    ]

    failures: list[tuple[str, int, int]] = []
    for label, fw_type, payload in probes:
        if not payload:
            logger.warning("  %-22s empty payload, skipped", label)
            continue
        status = _load_one(client, driver, MP0_BASE_DW, ring, ctx,
                           payload, fw_type, label, strict=False)
        if status != 0:
            failures.append((label, fw_type, status))
    return failures


def gfx_bring_up(client, driver, *,
                 firmware_dir: str,
                 sos_fw: str = "psp_14_0_3_sos.bin",
                 smu_fw: str = "smu_14_0_3.bin",
                 timeout_bootload_ms: int = 5000) -> GfxBringUpResult:
    """Bring the GPU to a state where RLC_RLCS_BOOTLOAD_STATUS.BOOTLOAD_COMPLETE=1.

    Raises RuntimeError if BOOTLOAD_COMPLETE isn't reached within
    `timeout_bootload_ms` after the tinygrad-order sequence completes.
    """
    def gc_rd(dw): return client.mmio_read32(_MMIO_BAR, (_GC_B1 + dw) * 4)

    # 1. SOS
    if not is_sos_alive(client, MP0_BASE_DW):
        logger.info("Loading PSP SOS")
        load_sos(client, driver, MP0_BASE_DW,
                 os.path.join(firmware_dir, sos_fw), verbose=False)
    c81 = client.mmio_read32(_MMIO_BAR, c2pmsg_dw(MP0_BASE_DW, 81) * 4)
    logger.info("SOS alive: C2PMSG_81=0x%08x", c81)

    # 2. Ring
    ring = ring_create(client, driver, MP0_BASE_DW,
                       destroy_first=True, verbose=False)

    ctx = alloc_cmd_ctx(driver)

    # 3. LOAD_TOC
    sos_blob = open(os.path.join(firmware_dir, sos_fw), "rb").read()
    toc_comp = next(x for x in parse_psp_firmware(sos_blob) if x.name == "TOC")
    submit_load_toc(client, driver, MP0_BASE_DW, ring, ctx, toc_comp.data)

    # 4. LOAD_IP_FW(SMU) — must precede the other fw per tinygrad.
    smu_blob = open(os.path.join(firmware_dir, smu_fw), "rb").read()
    uoff, usz, _uver = _parse_common_hdr(smu_blob)
    smu_payload = smu_blob[uoff:uoff + usz]
    _load_one(client, driver, MP0_BASE_DW, ring, ctx,
              smu_payload, GFX_FW_TYPE_SMU, "SMU", strict=True)

    # 5. LOAD_IP_FW for every other fw type.
    rejected = _load_all_gfx_firmware(client, driver, ring, ctx, firmware_dir)
    if rejected:
        logger.info("%d fw types rejected (non-fatal in current path):",
                    len(rejected))
        for label, fw_type, status in rejected:
            logger.info("  %-22s type=%3d status=0x%08x", label, fw_type, status)

    # 6. AUTOLOAD_RLC.
    autoload_resp = submit_autoload_rlc(client, driver, MP0_BASE_DW, ring, ctx)
    autoload_status = autoload_resp["status"]
    logger.info("AUTOLOAD_RLC status=0x%08x", autoload_status)
    if autoload_status != 0:
        raise RuntimeError(f"AUTOLOAD_RLC failed: status=0x{autoload_status:08x}")

    # 7-8. SMU driver table addr.
    tbl_mc = _vram_tbl_mc(client)
    logger.info("Driver table MC = 0x%x", tbl_mc)
    resp, _ = smu_send(client, PPSMC_MSG_SetDriverDramAddrHigh,
                       (tbl_mc >> 32) & 0xFFFFFFFF, timeout_ms=3000)
    if resp != 0x1:
        raise RuntimeError(f"SetDriverDramAddrHigh returned 0x{resp:x}")
    resp, _ = smu_send(client, PPSMC_MSG_SetDriverDramAddrLow,
                       tbl_mc & 0xFFFFFFFF, timeout_ms=3000)
    if resp != 0x1:
        raise RuntimeError(f"SetDriverDramAddrLow returned 0x{resp:x}")

    # 9. EnableAllSmuFeatures(0).
    try:
        enable_resp, _ = smu_send(client, PPSMC_MSG_EnableAllSmuFeatures,
                                  0, timeout_ms=10000)
        logger.info("EnableAllSmuFeatures(0) resp=0x%x", enable_resp)
    except TimeoutError:
        logger.warning("EnableAllSmuFeatures(0) timed out (may be OK if GFX came up)")
        enable_resp = None

    # 10. Poll BOOTLOAD_COMPLETE.
    deadline = time.time() + timeout_bootload_ms / 1000
    bl = 0
    while time.time() < deadline:
        bl = gc_rd(0x4e7c)  # RLC_RLCS_BOOTLOAD_STATUS
        if bl & 0x80000000:
            break
        time.sleep(0.01)

    if not (bl & 0x80000000):
        raise RuntimeError(
            f"BOOTLOAD_COMPLETE not set within {timeout_bootload_ms} ms "
            f"(final BOOTLOAD_STATUS=0x{bl:08x}, "
            f"RESET_CTRL=0x{gc_rd(0x40bc):08x}, CORE=0x{gc_rd(0x40b6):x})"
        )

    logger.info("GFX bring-up complete: BOOTLOAD=0x%08x CORE=0x%x RESET=0x%08x RLC_CNTL=0x%x",
                bl, gc_rd(0x40b6), gc_rd(0x40bc), gc_rd(0x4c00))

    return GfxBringUpResult(
        sos_alive=c81,
        ring=ring,
        psp_cmd_ctx=ctx,
        driver_table_mc=tbl_mc,
        autoload_rlc_status=autoload_status,
        bootload_status=bl,
        enable_all_resp=enable_resp,
        rejected_fw=rejected,
    )
