"""SCPM-aware bring-up path — skip DramToSmu pptable + SetAllowedMask.

Finding from the previous run: on gfx1201, both
`TransferTableDram2Smu(PPTABLE)` and `SetAllowedFeaturesMask` are
rejected (0xFF and 0xFD respectively). That matches Linux's
`smu_smc_hw_setup` which skips both when `adev->scpm_enabled`:

    if (!adev->scpm_enabled) {
        smu_write_pptable(smu);            // TransferTableDram2Smu
    }
    ...
    if (!adev->scpm_enabled) {
        smu_feature_set_allowed_mask(smu); // SetAllowedMask
    }

On SCPM cards PSP is responsible for the pptable and allowed-mask;
the driver just configures, runs BTC, updates PCIe params, and
kicks EnableAll. Our card is SCPM — confirmed by the rejections.

SCPM bring-up order (matching Linux):
  1. smu_bring_up (enable_domain=None).
  2. Full PSP autoload (LOAD_TOC + LOAD_IP_FW(IMU) + VRAM + AUTOLOAD_RLC).
  3. SetToolsDramAddr + SetSystemVirtualDramAddr.
  4. UseDefaultPPTable + TransferTableSmu2Dram(COMBO_PPTABLE=1).
  5. RunDcBtc.
  6. OverridePcieParameters (PPSMC_MSG 0x20) — new step for us.
  7. EnableAll(PWR_ALL=0).
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
    FEATURE_PWR_ALL,
    MP0_BASE_DW,
    PPSMC_MSG_EnableAllSmuFeatures,
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

    print("\n== 2: full PSP autoload ==")
    sos_blob = open(os.path.join(FIRMWARE_DIR, "psp_14_0_3_sos.bin"), "rb").read()
    toc_comp = next(x for x in parse_psp_firmware(sos_blob) if x.name == "TOC")
    submit_load_toc(c, drv, MP0_BASE_DW, result.ring, ctx, toc_comp.data)
    imu_blob = open(os.path.join(FIRMWARE_DIR, "gc_12_0_1_imu.bin"), "rb").read()
    iram, dram = _extract_imu(imu_blob)
    _load_one(c, drv, MP0_BASE_DW, result.ring, ctx, iram, GFX_FW_TYPE_IMU_I, "IMU_I", strict=True)
    _load_one(c, drv, MP0_BASE_DW, result.ring, ctx, dram, GFX_FW_TYPE_IMU_D, "IMU_D", strict=True)
    with open(os.path.join(FIRMWARE_DIR, "gc_12_0_1_toc.bin"), "rb") as f:
        toc_blob = f.read()
    layout = plan_autoload(toc_blob)
    build_autoload_buffer(c, FIRMWARE_DIR, layout, toc_blob)
    print(f"  AUTOLOAD_RLC status = 0x{submit_autoload_rlc(c, drv, MP0_BASE_DW, result.ring, ctx)['status']:08x}")

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

    # OverridePcieParameters param layout (from Linux
    # smu_v14_0_2_update_pcie_parameters):
    #   smu_pcie_arg = (link_level << 16) | (pcie_gen << 8) | pcie_width
    #
    # PcieGenSpeed enum:  0=Gen1  1=Gen2  2=Gen3  3=Gen4  4=Gen5
    # PcieLaneCount enum: 1=x1  2=x2  3=x4  4=x8  5=x12  6=x16
    # NUM_LINK_LEVELS = 3 — Linux sends one message per level.
    # Thunderbolt eGPU is effectively PCIe 3.0 x4 → gen=2, width=3.
    print("\n== 6: OverridePcieParameters × 3 levels (gen=2 Gen3, width=3 x4) ==")
    for level in range(3):
        arg = (level << 16) | (2 << 8) | 3
        _try(c, PPSMC_MSG_OverridePcieParameters, arg,
             f"OverridePcieParameters(level={level}, gen=2, width=3)", timeout=5000)

    print("\n== 7: EnableAll(PWR_ALL=0) ==")
    _try(c, PPSMC_MSG_EnableAllSmuFeatures, FEATURE_PWR_ALL,
         "EnableAll(PWR_ALL=0)", timeout=10000)

    print("\n== 8: poll BOOTLOAD_COMPLETE (5s) ==")
    GC_B1 = 0xA000
    def gc_rd(o): return c.mmio_read32(5, (GC_B1 + o) * 4)
    deadline = time.time() + 5
    last = None
    while time.time() < deadline:
        bl = gc_rd(0x4e7c); rst = gc_rd(0x40bc); core = gc_rd(0x40b6)
        if (bl, rst, core) != last:
            t = time.time() - deadline + 5
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
