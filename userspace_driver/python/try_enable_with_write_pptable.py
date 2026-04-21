"""Full Linux-parity SMU prep: write pptable TO SMU, run BTC, then EnableAll.

The prior runs showed `SetAllowedFeaturesMask` returning 0xFD
(CmdRejectedPrereq) even with the per-ASIC FeaturesToRun mask from
pptable. Reading amdgpu's `smu_smc_hw_setup` more carefully, the
step we were missing is `smu_write_pptable` — the DRIVER TO SMU
transfer that actually seeds SMU's internal pptable state:

  SmuToDram (TransferTableSmu2Dram, COMBO_PPTABLE)   <- we do this
  driver-side: store/check/parse pptable             <- we skip (not needed)
  DramToSmu (TransferTableDram2Smu, PPTABLE)         <- MISSING
  smu_run_btc -> PPSMC_MSG_RunDcBtc (0x36)           <- MISSING
  set_allowed_mask                                   <- the thing that was failing
  system_features_control (EnableAll)

`smu_cmn_write_pptable` copies the PPTable_t struct (from inside
the combo pptable at offset `pmfw_pptable_start_offset = 0x540`,
size `pmfw_pptable_size = 0x1174`) into the driver_table MC region,
then sends TransferTableDram2Smu with table_id=0 (TABLE_PPTABLE).
"""
from __future__ import annotations

import ctypes
import logging
import os
import struct
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
    PPSMC_MSG_SetAllowedFeaturesMaskHigh,
    PPSMC_MSG_SetAllowedFeaturesMaskLow,
    PPSMC_MSG_SetSystemVirtualDramAddrHigh,
    PPSMC_MSG_SetSystemVirtualDramAddrLow,
    PPSMC_MSG_SetToolsDramAddrHigh,
    PPSMC_MSG_SetToolsDramAddrLow,
    PPSMC_MSG_TransferTableDram2Smu,
    PPSMC_MSG_TransferTableSmu2Dram,
    PPSMC_MSG_UseDefaultPPTable,
    smu_bring_up,
    smu_send,
)

FIRMWARE_DIR = os.path.expanduser("~/firmware/linux-firmware/amdgpu")
TABLE_PPTABLE       = 0
TABLE_COMBO_PPTABLE = 1
PPSMC_MSG_RunDcBtc  = 0x36
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
    tbl_vram_off = tbl_mc - fb_base

    ctx = alloc_cmd_ctx(drv)

    print("\n== 2: full autoload ==")
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

    # Parse the combo pptable header to find the PPTable_t slice.
    bar0_cpu, _ = c.map_bar(0)
    base = bar0_cpu + tbl_vram_off
    hdr_dws = [(ctypes.c_uint32 * 1).from_address(base + i * 4)[0] for i in range(2)]
    # struct_size:16, fmt_rev:8, content_rev:8  |  tab_rev:8, src:8, pp_off:16
    pp_off  = (hdr_dws[1] >> 16) & 0xFFFF
    pp_size_dw = (ctypes.c_uint32 * 1).from_address(base + 8)[0]
    pp_size = pp_size_dw & 0xFFFF
    sku_off = (pp_size_dw >> 16) & 0xFFFF
    feat_low  = (ctypes.c_uint32 * 1).from_address(base + pp_off + 4)[0]
    feat_high = (ctypes.c_uint32 * 1).from_address(base + pp_off + 8)[0]
    print(f"  pmfw_pptable_start=0x{pp_off:x} size=0x{pp_size:x} sku_off=0x{sku_off:x}")
    print(f"  FeaturesToRun = 0x{feat_low:08x} / 0x{feat_high:08x}")

    # Copy the PPTable_t slice from [pp_off .. pp_off + pp_size] to
    # driver_table[0]. TransferTableDram2Smu will read from there.
    # Use DWORD reads + writes — c_ubyte slice reads SIGBUS past ~1 MB
    # on Apple Silicon's VRAM BAR mapping.
    print("\n== 5: copy PPTable_t (DWORD-by-DWORD) to driver_table[0] ==")
    n_full_dw = pp_size // 4
    for i in range(n_full_dw):
        dw = (ctypes.c_uint32 * 1).from_address(base + pp_off + i * 4)[0]
        (ctypes.c_uint32 * 1).from_address(base + i * 4)[0] = dw
    print(f"  wrote {pp_size} bytes ({n_full_dw} DWORDs) of PPTable_t to driver_table")

    print("\n== 6: TransferTableDram2Smu(PPTABLE=0) ==")
    _try(c, PPSMC_MSG_TransferTableDram2Smu, TABLE_PPTABLE,
         "TransferTableDram2Smu(PPTABLE=0)", timeout=5000)

    print("\n== 7: RunDcBtc (0x36) ==")
    _try(c, PPSMC_MSG_RunDcBtc, 0, "RunDcBtc", timeout=5000)

    print("\n== 8: SetAllowedFeaturesMask ==")
    _try(c, PPSMC_MSG_SetAllowedFeaturesMaskHigh, feat_high,
         f"SetAllowedFeaturesMaskHigh(0x{feat_high:x})")
    _try(c, PPSMC_MSG_SetAllowedFeaturesMaskLow, feat_low,
         f"SetAllowedFeaturesMaskLow (0x{feat_low:x})")

    print("\n== 9: EnableAll(PWR_ALL=0) ==")
    _try(c, PPSMC_MSG_EnableAllSmuFeatures, FEATURE_PWR_ALL, "EnableAll(PWR_ALL=0)", timeout=10000)

    print("\n== 10: poll BOOTLOAD_COMPLETE (5s) ==")
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
