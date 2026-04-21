"""Full SMU prep before SetAllowedMask + EnableAll:
  - SetDriverDramAddr (done by smu_bring_up).
  - SetToolsDramAddr (PMSTATUSLOG, 100 KB in VRAM).
  - SetSystemVirtualDramAddr (memory pool, 1 MB in VRAM).
  - UseDefaultPPTable + TransferTableSmu2Dram(COMBO_PPTABLE=1).
  - SetAllowedFeaturesMask with FeaturesToRun[0..1] from pptable.
  - EnableAll(PWR_ALL=0).

Maybe what SMU wants before SetAllowedMask isn't about the mask
bits themselves — it's one of the other table-location messages
Linux's smu_smc_hw_setup sends first.
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
    PPSMC_MSG_TransferTableSmu2Dram,
    PPSMC_MSG_UseDefaultPPTable,
    smu_bring_up,
    smu_send,
)

FIRMWARE_DIR = os.path.expanduser("~/firmware/linux-firmware/amdgpu")
TABLE_COMBO_PPTABLE = 1
# VRAM offsets (all inside BAR0's 256 MB window, past the 23 MB autoload buffer):
TOOL_VRAM_OFF = 0x1810000    # PMSTATUSLOG (SMU14_TOOL_SIZE = 0x19000)
POOL_VRAM_OFF = 0x1900000    # memory pool (1 MB)


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
    print(f"  driver_table MC = 0x{tbl_mc:x}")
    print(f"  tools (PMSTATUSLOG) MC = 0x{tool_mc:x}")
    print(f"  memory pool MC          = 0x{pool_mc:x}")

    ctx = alloc_cmd_ctx(drv)

    print("\n== 2: LOAD_TOC + LOAD_IP_FW(IMU) + VRAM autoload + AUTOLOAD_RLC ==")
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
    resp = submit_autoload_rlc(c, drv, MP0_BASE_DW, result.ring, ctx)
    print(f"  AUTOLOAD_RLC status = 0x{resp['status']:08x}")

    print("\n== 3: SetToolsDramAddr ==")
    _try(c, PPSMC_MSG_SetToolsDramAddrHigh, (tool_mc >> 32) & 0xFFFFFFFF,
         f"SetToolsDramAddrHigh(0x{(tool_mc >> 32) & 0xFFFFFFFF:x})")
    _try(c, PPSMC_MSG_SetToolsDramAddrLow, tool_mc & 0xFFFFFFFF,
         f"SetToolsDramAddrLow (0x{tool_mc & 0xFFFFFFFF:x})")

    print("\n== 4: SetSystemVirtualDramAddr (memory pool) ==")
    _try(c, PPSMC_MSG_SetSystemVirtualDramAddrHigh, (pool_mc >> 32) & 0xFFFFFFFF,
         f"SetSystemVirtualDramAddrHigh(0x{(pool_mc >> 32) & 0xFFFFFFFF:x})")
    _try(c, PPSMC_MSG_SetSystemVirtualDramAddrLow, pool_mc & 0xFFFFFFFF,
         f"SetSystemVirtualDramAddrLow (0x{pool_mc & 0xFFFFFFFF:x})")

    print("\n== 5: UseDefaultPPTable + TransferTableSmu2Dram(COMBO_PPTABLE) ==")
    _try(c, PPSMC_MSG_UseDefaultPPTable, 0, "UseDefaultPPTable")
    _try(c, PPSMC_MSG_TransferTableSmu2Dram, TABLE_COMBO_PPTABLE,
         "TransferTableSmu2Dram(COMBO_PPTABLE=1)", timeout=5000)

    # Parse pptable + FeaturesToRun.
    bar0_cpu, _ = c.map_bar(0)
    base = bar0_cpu + (tbl_mc - fb_base)
    hdr0 = (ctypes.c_uint32 * 1).from_address(base + 0)[0]      # struct_size + fmt_rev + content_rev
    hdr1 = (ctypes.c_uint32 * 1).from_address(base + 4)[0]      # tab_rev + src + pp_off
    pp_off = (hdr1 >> 16) & 0xFFFF                               # pmfw_pptable_start_offset
    feat_low  = (ctypes.c_uint32 * 1).from_address(base + pp_off + 4)[0]
    feat_high = (ctypes.c_uint32 * 1).from_address(base + pp_off + 8)[0]
    print(f"\n  pmfw_pptable_start=0x{pp_off:x}")
    print(f"  FeaturesToRun[0] = 0x{feat_low:08x}")
    print(f"  FeaturesToRun[1] = 0x{feat_high:08x}")

    print("\n== 6: SetAllowedFeaturesMask with parsed values ==")
    _try(c, PPSMC_MSG_SetAllowedFeaturesMaskHigh, feat_high,
         f"SetAllowedFeaturesMaskHigh(0x{feat_high:x})")
    _try(c, PPSMC_MSG_SetAllowedFeaturesMaskLow, feat_low,
         f"SetAllowedFeaturesMaskLow (0x{feat_low:x})")

    print("\n== 7: EnableAll(PWR_ALL=0) ==")
    _try(c, PPSMC_MSG_EnableAllSmuFeatures, FEATURE_PWR_ALL, "EnableAll(PWR_ALL=0)", timeout=10000)

    print("\n== 8: poll BOOTLOAD_COMPLETE (5s) ==")
    GC_B1 = 0xA000
    def gc_rd(o): return c.mmio_read32(5, (GC_B1 + o) * 4)
    deadline = time.time() + 5
    last = None
    while time.time() < deadline:
        bl = gc_rd(0x4e7c); rst = gc_rd(0x40bc); core = gc_rd(0x40b6)
        snap = (bl, rst, core)
        if snap != last:
            t = time.time() - deadline + 5
            print(f"  t={t:5.2f}s CORE=0x{core:x} RESET=0x{rst:08x} BOOTLOAD=0x{bl:08x}")
            last = snap
        if bl & 0x80000000:
            print("  BOOTLOAD_COMPLETE ✓")
            break
        time.sleep(0.05)

    print("\n== Final ==")
    _try(c, PPSMC_MSG_GetRunningSmuFeaturesLo, 0, "GetRunningFeaturesLo", timeout=1500)
    _try(c, PPSMC_MSG_GetRunningSmuFeaturesHi, 0, "GetRunningFeaturesHi", timeout=1500)


if __name__ == "__main__":
    main()
