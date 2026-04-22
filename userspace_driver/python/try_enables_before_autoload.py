"""Reorder: per-bit feature enables BEFORE AUTOLOAD_RLC.

Prior run (try_scpm_enable_all.py): per-bit EnableSmuFeaturesLow
succeeded on 25/26 bits, EnableSmuFeaturesHigh(feat_high) accepted
all at once — but this was AFTER AUTOLOAD_RLC, and enabling DPM
regressed GFX state (IMU_GFX_RESET_CTRL 0x7F -> 0x30,
BOOTLOAD_STATUS 0x3F -> 0x00). SMU gated GFX as soon as DPM_GFXCLK
came online because PSP had brought RLC up with DPM-off assumptions.

This run flips the order: stand up the SMU driver tables, warm SOC,
enable all pptable features, THEN ask PSP to run AUTOLOAD_RLC. If
DPM is up when PSP brings GFX online, RLC should be able to reach
BOOTLOAD_COMPLETE without being gated back into reset.

Sequence:
  1. smu_bring_up(enable_domain=None).
  2. LOAD_TOC + LOAD_IP_FW(IMU_I/D) — PSP firmware prep (no AUTOLOAD).
  3. SetToolsDramAddr + SetSystemVirtualDramAddr.
  4. UseDefaultPPTable + TransferTableSmu2Dram(COMBO_PPTABLE).
  5. RunDcBtc.
  6. OverridePcieParameters × 3 levels.
  7. EnableAll(PWR_SOC=3) warm-up.
  8. Per-bit EnableSmuFeaturesLow + EnableSmuFeaturesHigh(feat_high).
  9. DisallowGfxOff.
  10. Fill autoload buffer + AUTOLOAD_RLC.
  11. Poll BOOTLOAD_COMPLETE.
"""
from __future__ import annotations

import ctypes
import logging
import os
import sys
import time

from amd_gpu_driver.backends.macos.gfx_autoload import (
    build_autoload_buffer,
    plan_autoload,
)
from amd_gpu_driver.backends.macos.gfx_psp_autoload import (
    _extract_imu,
    _load_one,
    submit_autoload_rlc,
    submit_load_toc,
)
from amd_gpu_driver.backends.macos.iokit_client import IOKitClient
from amd_gpu_driver.backends.macos.psp_bootloader import parse_psp_firmware
from amd_gpu_driver.backends.macos.psp_cmd import (
    GFX_FW_TYPE_IMU_D,
    GFX_FW_TYPE_IMU_I,
    alloc_cmd_ctx,
)
from amd_gpu_driver.backends.macos.smu import (
    MP0_BASE_DW,
    PPSMC_MSG_DisallowGfxOff,
    PPSMC_MSG_EnableAllSmuFeatures,
    PPSMC_MSG_EnableSmuFeaturesHigh,
    PPSMC_MSG_EnableSmuFeaturesLow,
    PPSMC_MSG_GetRunningSmuFeaturesHi,
    PPSMC_MSG_GetRunningSmuFeaturesLo,
    PPSMC_MSG_SetSystemVirtualDramAddrHigh,
    PPSMC_MSG_SetSystemVirtualDramAddrLow,
    PPSMC_MSG_SetToolsDramAddrHigh,
    PPSMC_MSG_SetToolsDramAddrLow,
    PPSMC_MSG_TransferTableSmu2Dram,
    PPSMC_MSG_UseDefaultPPTable,
    smu_bring_up,
    smu_send,
)

FIRMWARE_DIR = os.path.expanduser("~/firmware/linux-firmware/amdgpu")
PPSMC_MSG_RunDcBtc               = 0x36
PPSMC_MSG_OverridePcieParameters = 0x20
TABLE_COMBO_PPTABLE = 1
TOOL_VRAM_OFF = 0x1810000
POOL_VRAM_OFF = 0x1900000


class _DriverShim:
    def __init__(self, client): self.client = client
    def alloc_dma(self, size):
        dma = self.client.alloc_dma(size)
        bus = dma.segments[0][0] if dma.segments else 0
        return (dma.cpu_addr, bus, dma.buffer_id)
    def free_dma(self, h): self.client.free_dma(h)


def _try(c, msg, arg, name, *, timeout=3000):
    try:
        r, a = smu_send(c, msg, arg, timeout_ms=timeout)
        print(f"  {name:52s} resp=0x{r:x} arg_out=0x{a:x}")
        return r, a
    except TimeoutError:
        print(f"  {name:52s} TIMEOUT")
        return None, None


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    c = IOKitClient(); c.open()
    info = c.get_info()
    print(f"device=0x{info.device_id:04x} rev=0x{info.revision_id:02x}")
    if c.mmio_read32(5, (0x16000 + 0x40 + 81) * 4) != 0:
        print("SOS already alive — replug first.")
        sys.exit(0)
    drv = _DriverShim(c)

    print("\n== 1: smu_bring_up ==")
    result = smu_bring_up(c, drv, firmware_dir=FIRMWARE_DIR, enable_domain=None)
    fb_base = (c.mmio_read32(5, (0x1A000 + 0x0554) * 4) & 0xFFFFFF) << 24
    tbl_mc  = result.driver_table_mc
    tool_mc = fb_base + TOOL_VRAM_OFF
    pool_mc = fb_base + POOL_VRAM_OFF

    ctx = alloc_cmd_ctx(drv)

    print("\n== 2: LOAD_TOC + LOAD_IP_FW(IMU_I/D) (PSP prep, no AUTOLOAD yet) ==")
    sos_blob = open(os.path.join(FIRMWARE_DIR, "psp_14_0_3_sos.bin"), "rb").read()
    toc_comp = next(x for x in parse_psp_firmware(sos_blob) if x.name == "TOC")
    submit_load_toc(c, drv, MP0_BASE_DW, result.ring, ctx, toc_comp.data)
    imu_blob = open(os.path.join(FIRMWARE_DIR, "gc_12_0_1_imu.bin"), "rb").read()
    iram, dram = _extract_imu(imu_blob)
    _load_one(c, drv, MP0_BASE_DW, result.ring, ctx, iram, GFX_FW_TYPE_IMU_I, "IMU_I", strict=True)
    _load_one(c, drv, MP0_BASE_DW, result.ring, ctx, dram, GFX_FW_TYPE_IMU_D, "IMU_D", strict=True)

    print("\n== 3: SetToolsDramAddr + SetSystemVirtualDramAddr ==")
    _try(c, PPSMC_MSG_SetToolsDramAddrHigh, (tool_mc >> 32) & 0xFFFFFFFF, "SetToolsDramAddrHigh")
    _try(c, PPSMC_MSG_SetToolsDramAddrLow, tool_mc & 0xFFFFFFFF, "SetToolsDramAddrLow")
    _try(c, PPSMC_MSG_SetSystemVirtualDramAddrHigh, (pool_mc >> 32) & 0xFFFFFFFF, "SetSystemVirtualDramAddrHigh")
    _try(c, PPSMC_MSG_SetSystemVirtualDramAddrLow, pool_mc & 0xFFFFFFFF, "SetSystemVirtualDramAddrLow")

    print("\n== 4: UseDefaultPPTable + TransferTableSmu2Dram(COMBO_PPTABLE) ==")
    _try(c, PPSMC_MSG_UseDefaultPPTable, 0, "UseDefaultPPTable")
    _try(c, PPSMC_MSG_TransferTableSmu2Dram, TABLE_COMBO_PPTABLE,
         "TransferTableSmu2Dram(COMBO_PPTABLE=1)", timeout=5000)

    print("\n== 5: RunDcBtc ==")
    _try(c, PPSMC_MSG_RunDcBtc, 0, "RunDcBtc", timeout=5000)

    print("\n== 6: OverridePcieParameters × 3 levels (gen=2 Gen3, width=3 x4) ==")
    for level in range(3):
        arg = (level << 16) | (2 << 8) | 3
        _try(c, PPSMC_MSG_OverridePcieParameters, arg,
             f"OverridePcieParameters(level={level}, gen=2, width=3)", timeout=5000)

    bar0_cpu, _ = c.map_bar(0)
    base = bar0_cpu + (tbl_mc - fb_base)
    dw1 = (ctypes.c_uint32 * 1).from_address(base + 4)[0]
    pp_off = (dw1 >> 16) & 0xFFFF
    feat_low  = (ctypes.c_uint32 * 1).from_address(base + pp_off + 4)[0]
    feat_high = (ctypes.c_uint32 * 1).from_address(base + pp_off + 8)[0]
    print(f"\n  Pptable FeaturesToRun: low=0x{feat_low:08x} high=0x{feat_high:08x}")

    print("\n== 7: EnableAll(PWR_SOC=3) warm-up ==")
    _try(c, PPSMC_MSG_EnableAllSmuFeatures, 3, "EnableAll(PWR_SOC=3)", timeout=5000)

    print("\n== 8a: per-bit EnableSmuFeaturesLow (BEFORE AUTOLOAD_RLC) ==")
    running_lo = 0
    for bit in range(32):
        bit_mask = 1 << bit
        if not (feat_low & bit_mask):
            continue
        if running_lo & bit_mask:
            continue
        try:
            r, arg_out = smu_send(c, PPSMC_MSG_EnableSmuFeaturesLow, bit_mask, timeout_ms=2000)
            tag = "enabled" if (arg_out & bit_mask) else "refused"
            print(f"  bit {bit:2d} (0x{bit_mask:08x}): resp=0x{r:x} arg_out=0x{arg_out:08x}  {tag}")
            if arg_out & bit_mask:
                running_lo |= bit_mask
        except TimeoutError:
            print(f"  bit {bit:2d} (0x{bit_mask:08x}): HUNG")
            break
    print(f"\n  Low-mask enabled: 0x{running_lo:08x}")

    print("\n== 8b: EnableSmuFeaturesHigh(feat_high) ==")
    _try(c, PPSMC_MSG_EnableSmuFeaturesHigh, feat_high,
         f"EnableSmuFeaturesHigh(0x{feat_high:x})", timeout=5000)

    print("\n== 9: DisallowGfxOff ==")
    _try(c, PPSMC_MSG_DisallowGfxOff, 0, "DisallowGfxOff", timeout=3000)

    # Snapshot GFX state right before AUTOLOAD_RLC — if DPM gated
    # things already, the autoload will fail. If state is still the
    # default power-gated look, the autoload has the clean slate we
    # wanted.
    GC_B1 = 0xA000
    def gc_rd(o): return c.mmio_read32(5, (GC_B1 + o) * 4)
    print("\n  pre-autoload GFX snapshot:")
    print(f"    IMU_CORE_CTRL = 0x{gc_rd(0x40b6):x}")
    print(f"    IMU_GFX_RESET_CTRL = 0x{gc_rd(0x40bc):08x}")
    print(f"    RLC_RLCS_BOOTLOAD_STATUS = 0x{gc_rd(0x4e7c):08x}")
    print(f"    RLC_CNTL = 0x{gc_rd(0x4b00):x}")

    print("\n== 10: fill autoload buffer + AUTOLOAD_RLC ==")
    with open(os.path.join(FIRMWARE_DIR, "gc_12_0_1_toc.bin"), "rb") as f:
        toc_blob = f.read()
    layout = plan_autoload(toc_blob)
    build_autoload_buffer(c, FIRMWARE_DIR, layout, toc_blob)
    resp = submit_autoload_rlc(c, drv, MP0_BASE_DW, result.ring, ctx)
    print(f"  AUTOLOAD_RLC status = 0x{resp['status']:08x}")

    print("\n== 11: poll BOOTLOAD_COMPLETE (10s) ==")
    deadline = time.time() + 10
    last = None
    while time.time() < deadline:
        bl = gc_rd(0x4e7c); rst = gc_rd(0x40bc); core = gc_rd(0x40b6)
        if (bl, rst, core) != last:
            t = time.time() - deadline + 10
            print(f"  t={t:5.2f}s CORE=0x{core:x} RESET=0x{rst:08x} BOOTLOAD=0x{bl:08x}")
            last = (bl, rst, core)
        if bl & 0x80000000:
            print("  BOOTLOAD_COMPLETE ✓")
            break
        time.sleep(0.05)

    print("\n== Final feature state ==")
    _try(c, PPSMC_MSG_GetRunningSmuFeaturesLo, 0, "GetRunningFeaturesLo", timeout=1500)
    _try(c, PPSMC_MSG_GetRunningSmuFeaturesHi, 0, "GetRunningFeaturesHi", timeout=1500)


if __name__ == "__main__":
    main()
