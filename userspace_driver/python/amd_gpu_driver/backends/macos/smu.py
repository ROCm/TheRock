"""SMU v14.0 mailbox + bring-up for gfx1201 (MP1 14.0.3, Navi 48).

Provides:
  - PPSMC_MSG_* constants (subset we use — cross-checked against
    Linux `smu_v14_0_2_ppsmc.h`).
  - FEATURE_PWR_DOMAIN_e enum values.
  - `smu_send(client, msg_id, arg)` — MP1 C2PMSG_66/82/90 mailbox txn.
  - `smu_bring_up(client, driver, *, firmware_path, ...)` — cold-boot
    orchestrator that wires SOS → PSP KM ring → LOAD_IP_FW(SMU) →
    SetDriverDramAddr → EnableAllSmuFeatures(PWR_SOC). Leaves the SMU
    with SOC DPM running, ready for DisallowGfxOff.

Gotchas that cost us real hardware time (see commit history + memory):
  1. PSP `LOAD_IP_FW` wants **raw ucode bytes**, not the whole .bin
     container. We parse `common_firmware_header` and pass only the
     `[ucode_off, ucode_off + ucode_size]` slice. Passing the whole
     file returns PSP status 0x11.
  2. `PPSMC_MSG_SetDriverDramAddrHigh/Low` are **0x0E / 0x0F** on
     smu_v14_0_2 — **not** 0x04 / 0x05 (those are
     SetAllowedFeaturesMaskLow/High and get rejected 0xFD here).
  3. `EnableAllSmuFeatures` arg is a `FEATURE_PWR_DOMAIN_e` selector
     (0=ALL, 3=SOC, 4=GFX) — **not** a feature bitmask. Arg 0 (ALL)
     and arg 4 (GFX) require GMC/MMHUB + GFXHUB init; they hang the
     SMU on a bare GPU. Arg 3 (SOC) works with no GMC setup.
  4. The driver table is placed in VRAM (inside FB aperture) so SMU
     can DMA-read it without any MMHUB GART/AGP setup.
"""
from __future__ import annotations

import ctypes
import logging
import os
import struct
import time
from dataclasses import dataclass

from amd_gpu_driver.backends.macos.psp_bootloader import c2pmsg_dw, is_sos_alive, load_sos
from amd_gpu_driver.backends.macos.psp_cmd import (
    GFX_FW_TYPE_SMU,
    alloc_cmd_ctx,
    submit_ip_fw_load,
)
from amd_gpu_driver.backends.macos.psp_ring import PSPRing, ring_create

logger = logging.getLogger(__name__)

# ---- gfx1201 IP bases (DWORD units, from IP discovery) ----
MP0_BASE_DW   = 0x16000
MP1_BASE_DW   = 0x16200
MMHUB_BASE_DW = 0x1A000

# ---- MP1 SMU mailbox DWORD slots within the MP1 base ----
# Mirrors the PSP convention — slot N lives at base + 0x40 + N.
C2PMSG_BASE_DW = 0x40

# ---- PPSMC message IDs (smu_v14_0_2_ppsmc.h) ----
PPSMC_MSG_TestMessage             = 0x01
PPSMC_MSG_GetSmuVersion           = 0x02
PPSMC_MSG_GetDriverIfVersion      = 0x03
# 0x04 / 0x05 are SetAllowedFeaturesMaskLow/High — NOT SetDriverDramAddr.
PPSMC_MSG_EnableAllSmuFeatures    = 0x06
PPSMC_MSG_EnableSmuFeaturesLow    = 0x08
PPSMC_MSG_EnableSmuFeaturesHigh   = 0x09
PPSMC_MSG_GetRunningSmuFeaturesLo = 0x0C
PPSMC_MSG_GetRunningSmuFeaturesHi = 0x0D
PPSMC_MSG_SetDriverDramAddrHigh   = 0x0E
PPSMC_MSG_SetDriverDramAddrLow    = 0x0F
PPSMC_MSG_AllowGfxOff             = 0x28
PPSMC_MSG_DisallowGfxOff          = 0x29

# ---- FEATURE_PWR_DOMAIN_e (smu14_driver_if_v14_0.h) ----
# Selector, not a bitmask. Passed as the arg to EnableAllSmuFeatures.
FEATURE_PWR_ALL   = 0  # hangs on bare GPU — needs MMHUB/GFXHUB
FEATURE_PWR_S5    = 1
FEATURE_PWR_BACO  = 2
FEATURE_PWR_SOC   = 3  # works without GMC — SOC DPM comes up
FEATURE_PWR_GFX   = 4  # hangs without GFXHUB init

# ---- MMHUB MC_VM register DWORD offsets (mmhub_4_1_0_offset.h) ----
regMMMC_VM_FB_LOCATION_BASE = 0x0554
regMMMC_VM_FB_LOCATION_TOP  = 0x0555

# ---- PPSMC response codes ----
PPSMC_Result_OK                = 0x01
PPSMC_Result_Failed            = 0xFF
PPSMC_Result_UnknownCmd        = 0xFE
PPSMC_Result_CmdRejectedPrereq = 0xFD
PPSMC_Result_CmdRejectedBusy   = 0xFC


# ---- MMIO BAR index (matches MacOSDevice._MMIO_BAR). ----
_MMIO_BAR = 5


@dataclass
class SmuBringUpResult:
    sos_alive: int               # C2PMSG_81 sign-of-life
    ring: PSPRing                # Live PSP KM ring; caller keeps the handle
    smu_version: int             # Value returned by GetSmuVersion
    driver_table_mc: int         # MC address we gave the SMU
    running_features_low: int
    running_features_high: int


def _mmhub_rd(client, dw_off):
    return client.mmio_read32(_MMIO_BAR, (MMHUB_BASE_DW + dw_off) * 4)


def smu_send(client, msg_id: int, arg: int = 0, *, timeout_ms: int = 2000):
    """Send one message via the MP1 mailbox.

    Returns (response, arg_out) on ack; raises `TimeoutError` if C2PMSG_90
    stays zero past the deadline. Callers that expect the SMU may hang
    (e.g. probing a new EnableAllSmuFeatures arg) should catch the
    TimeoutError and treat it as a hard failure — a timed-out SMU stays
    hung until the card is reset.
    """
    c66 = (MP1_BASE_DW + C2PMSG_BASE_DW + 66) * 4
    c82 = (MP1_BASE_DW + C2PMSG_BASE_DW + 82) * 4
    c90 = (MP1_BASE_DW + C2PMSG_BASE_DW + 90) * 4
    client.mmio_write32(_MMIO_BAR, c90, 0)
    client.mmio_write32(_MMIO_BAR, c82, arg & 0xFFFFFFFF)
    client.mmio_write32(_MMIO_BAR, c66, msg_id)
    deadline = time.time() + timeout_ms / 1000
    while time.time() < deadline:
        v = client.mmio_read32(_MMIO_BAR, c90)
        if v != 0:
            return (v, client.mmio_read32(_MMIO_BAR, c82))
        time.sleep(0.002)
    raise TimeoutError(
        f"SMU msg 0x{msg_id:02x} (arg=0x{arg:x}) did not ack in {timeout_ms} ms"
    )


def _parse_ucode(blob: bytes) -> tuple[int, int, int]:
    """Parse common_firmware_header. Returns (ucode_version, ucode_size, ucode_off)."""
    (_size_bytes, _header_size_bytes, _hver_maj, _hver_min,
     _ipver_maj, _ipver_min, ucode_version, ucode_size_bytes,
     ucode_array_offset_bytes, _crc32) = struct.unpack_from("<IIHHHHIIII", blob, 0)
    return ucode_version, ucode_size_bytes, ucode_array_offset_bytes


def _vram_tbl_mc(client) -> int:
    """Pick an MC address inside the FB aperture for the SMU driver table.

    We park it 128 KB below the top of VRAM so we don't collide with the
    IP discovery region (which lives at VRAM_SIZE - 64 KB).
    """
    fb_base_val = _mmhub_rd(client, regMMMC_VM_FB_LOCATION_BASE) & 0xFFFFFF
    fb_top_val  = _mmhub_rd(client, regMMMC_VM_FB_LOCATION_TOP) & 0xFFFFFF
    fb_start_mc = fb_base_val << 24
    fb_end_mc   = (fb_top_val + 1) << 24
    vram_size   = fb_end_mc - fb_start_mc
    return fb_start_mc + vram_size - 0x20000


def _smu_load_firmware(client, driver, ring: PSPRing,
                       smu_fw_path: str) -> tuple[int, int]:
    """Load SMU ucode via PSP LOAD_IP_FW. Returns (bus_addr, ucode_version)."""
    with open(smu_fw_path, "rb") as f:
        blob = f.read()
    ucode_ver, ucode_size, ucode_off = _parse_ucode(blob)
    # Round the staging buffer up to a page so `ctypes.memset` zeroes the
    # tail past the ucode payload — PSP's authenticator is strict about
    # unexpected bytes, and we've seen status 0x11 returns when the tail
    # was left uninitialised.
    staging_size = (ucode_size + 0xFFF) & ~0xFFF
    cpu, bus, _handle = driver.alloc_dma(staging_size)
    ctypes.memset(cpu, 0, staging_size)
    (ctypes.c_ubyte * ucode_size).from_address(cpu)[:] = \
        blob[ucode_off:ucode_off + ucode_size]
    ctx = alloc_cmd_ctx(driver)
    # Pre-submit snapshot helps us disambiguate "PSP didn't process the
    # frame" (fence stays 0, wptr advances) from "PSP isn't receiving
    # kicks" (wptr register doesn't even accept writes).
    c67_pre = client.mmio_read32(_MMIO_BAR, c2pmsg_dw(MP0_BASE_DW, 67) * 4)
    c64_pre = client.mmio_read32(_MMIO_BAR, c2pmsg_dw(MP0_BASE_DW, 64) * 4)
    logger.info("PSP submit pre-state: C2PMSG_64=0x%08x C2PMSG_67=0x%08x",
                c64_pre, c67_pre)
    resp = submit_ip_fw_load(client, driver, MP0_BASE_DW, ring, ctx,
                             bus, ucode_size, GFX_FW_TYPE_SMU, verbose=True)
    if resp["status"] != 0:
        raise RuntimeError(
            f"PSP LOAD_IP_FW(SMU) failed: status=0x{resp['status']:08x}. "
            "Status 0x11 usually means the ucode container header was "
            "passed instead of the raw ucode payload."
        )
    return bus, ucode_ver


def smu_bring_up(client, driver, *,
                 firmware_dir: str,
                 sos_fw: str = "psp_14_0_3_sos.bin",
                 smu_fw: str = "smu_14_0_3.bin",
                 enable_domain: int = FEATURE_PWR_SOC,
                 ring: PSPRing | None = None) -> SmuBringUpResult:
    """Cold-boot the SMU to the point where SOC DPM is running.

    Post-conditions on success:
      - SOS alive (C2PMSG_81 nonzero).
      - PSP KM ring created (returned in result so callers can reuse it).
      - SMU v14 firmware loaded and its mailbox responsive.
      - Driver table MC address registered with the SMU (in VRAM).
      - `EnableAllSmuFeatures(enable_domain)` sent successfully; SOC
        features are reported in the running-features bitmask.

    Use `enable_domain=FEATURE_PWR_GFX` AFTER GMC init (Phase 7) — it
    hangs the SMU on a bare GPU.

    `ring` may be supplied by callers who already created the PSP KM
    ring earlier in the bring-up; otherwise this function will.
    """
    sos_path = os.path.join(firmware_dir, sos_fw)
    smu_path = os.path.join(firmware_dir, smu_fw)

    # 1. Load SOS if the card hasn't been POST'd into a running SOS yet.
    if not is_sos_alive(client, MP0_BASE_DW):
        logger.info("Loading PSP SOS from %s", sos_path)
        load_sos(client, driver, MP0_BASE_DW, sos_path, verbose=False)
    c81 = client.mmio_read32(_MMIO_BAR, c2pmsg_dw(MP0_BASE_DW, 81) * 4)
    logger.info("SOS alive: C2PMSG_81=0x%08x", c81)

    # 2. PSP KM ring (create if caller didn't).
    if ring is None:
        logger.info("Creating PSP KM ring")
        ring = ring_create(client, driver, MP0_BASE_DW,
                           destroy_first=True, verbose=False)

    # 3. Load SMU firmware (ucode payload only — see comment in _smu_load_firmware).
    logger.info("Loading SMU firmware from %s", smu_path)
    _smu_bus, ucode_ver = _smu_load_firmware(client, driver, ring, smu_path)

    # 4. Mailbox sanity — GetSmuVersion should echo the ucode version we loaded.
    resp, smu_ver = smu_send(client, PPSMC_MSG_GetSmuVersion, 0)
    if resp != PPSMC_Result_OK:
        raise RuntimeError(f"SMU GetSmuVersion returned 0x{resp:x}")
    if smu_ver != ucode_ver:
        logger.warning("SMU reports version 0x%x; ucode header 0x%x",
                       smu_ver, ucode_ver)

    # 5. Driver table MC address (VRAM, inside FB aperture).
    tbl_mc = _vram_tbl_mc(client)
    logger.info("Driver table MC = 0x%x", tbl_mc)

    # 6. Register driver table with SMU.
    resp, _ = smu_send(client, PPSMC_MSG_SetDriverDramAddrHigh,
                       (tbl_mc >> 32) & 0xFFFFFFFF)
    if resp != PPSMC_Result_OK:
        raise RuntimeError(f"SetDriverDramAddrHigh returned 0x{resp:x}")
    resp, _ = smu_send(client, PPSMC_MSG_SetDriverDramAddrLow,
                       tbl_mc & 0xFFFFFFFF)
    if resp != PPSMC_Result_OK:
        raise RuntimeError(f"SetDriverDramAddrLow returned 0x{resp:x}")

    # 7. Enable the requested power domain. Keep the timeout short — a
    # hang here is terminal until the GPU is reset, so we'd rather fail
    # fast than wait 10 s.
    resp, arg_out = smu_send(client, PPSMC_MSG_EnableAllSmuFeatures,
                             enable_domain, timeout_ms=3000)
    if resp != PPSMC_Result_OK:
        raise RuntimeError(
            f"EnableAllSmuFeatures({enable_domain}) returned 0x{resp:x}"
        )
    logger.info("EnableAllSmuFeatures(%d) OK, arg_out=0x%x", enable_domain, arg_out)

    # 8. Query running features for the result.
    _, run_lo = smu_send(client, PPSMC_MSG_GetRunningSmuFeaturesLo, 0)
    _, run_hi = smu_send(client, PPSMC_MSG_GetRunningSmuFeaturesHi, 0)
    logger.info("RunningFeatures: low=0x%x high=0x%x", run_lo, run_hi)

    return SmuBringUpResult(
        sos_alive=c81,
        ring=ring,
        smu_version=smu_ver,
        driver_table_mc=tbl_mc,
        running_features_low=run_lo,
        running_features_high=run_hi,
    )
