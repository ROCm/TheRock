"""Phase 9 entry point: cold-boot to BOOTLOAD_COMPLETE via library.

Uses the `gfx_bring_up()` function from backends/macos/gfx_bringup.py
(validates the library is correctly wired up). After BOOTLOAD_COMPLETE
fires, dumps a broader set of registers that we'll need to touch next:

  - GFXHUB MC_VM_* / VM_L2_* (for GMC init)
  - MEC control regs (for compute ring setup)
  - Memory apertures (FB_LOCATION, SYSTEM_APERTURE)

This script doesn't write anything — pure diagnostic after bring-up.
"""
from __future__ import annotations

import logging
import os
import sys

from amd_gpu_driver.backends.macos.gfx_bringup import gfx_bring_up
from amd_gpu_driver.backends.macos.iokit_client import IOKitClient

FIRMWARE_DIR = os.path.expanduser("~/firmware/linux-firmware/amdgpu")


class _DriverShim:
    def __init__(self, client): self.client = client
    def alloc_dma(self, size):
        dma = self.client.alloc_dma(size)
        bus = dma.segments[0][0] if dma.segments else 0
        return (dma.cpu_addr, bus, dma.buffer_id)
    def free_dma(self, h): self.client.free_dma(h)


# Register tables — all BASE_IDX annotated.
# GC BASE_IDX=0 -> 0x1260, BASE_IDX=1 -> 0xA000
# MMHUB BASE_IDX=0 -> 0x1A000

GC_B0 = 0x1260
GC_B1 = 0xA000
MM_B0 = 0x1A000

GFX_REGS = [
    ("GRBM_STATUS",                        GC_B0, 0x0da4),
    ("CP_STAT",                            GC_B0, 0x0f40),
    ("GFX_IMU_CORE_CTRL",                  GC_B1, 0x40b6),
    ("GFX_IMU_GFX_RESET_CTRL",             GC_B1, 0x40bc),
    ("RLC_CNTL",                           GC_B1, 0x4c00),
    ("RLC_STAT",                           GC_B1, 0x4c04),
    ("RLC_RLCS_BOOTLOAD_STATUS",           GC_B1, 0x4e7c),
    ("RLC_GPM_STAT",                       GC_B1, 0x4e6c),
]

# GFXHUB MC/VM regs (BASE_IDX=0 per gc_12_0_0_offset.h)
GFXHUB_REGS = [
    ("GCMC_VM_FB_LOCATION_BASE",           GC_B0, 0x1614),
    ("GCMC_VM_FB_LOCATION_TOP",            GC_B0, 0x1615),
    ("GCMC_VM_FB_OFFSET",                  GC_B0, 0x1616),
    ("GCMC_VM_AGP_BASE",                   GC_B0, 0x1618),
    ("GCMC_VM_SYSTEM_APERTURE_LOW_ADDR",   GC_B0, 0x1619),
    ("GCMC_VM_SYSTEM_APERTURE_HIGH_ADDR",  GC_B0, 0x161a),
    ("GCMC_VM_MX_L1_TLB_CNTL",             GC_B0, 0x161b),
    ("GCVM_L2_CNTL",                       GC_B0, 0x15c4),
    ("GCVM_CONTEXT0_CNTL",                 GC_B0, 0x1624),
]

# MMHUB equivalents (BASE_IDX=0)
MMHUB_REGS = [
    ("MMMC_VM_FB_LOCATION_BASE",           MM_B0, 0x0554),
    ("MMMC_VM_FB_LOCATION_TOP",            MM_B0, 0x0555),
    ("MMMC_VM_AGP_BASE",                   MM_B0, 0x0558),
    ("MMMC_VM_SYSTEM_APERTURE_LOW_ADDR",   MM_B0, 0x0559),
    ("MMMC_VM_SYSTEM_APERTURE_HIGH_ADDR",  MM_B0, 0x055a),
    ("MMMC_VM_MX_L1_TLB_CNTL",             MM_B0, 0x055b),
]

# MEC control
MEC_REGS = [
    ("CP_MEC_RS64_CNTL",                   GC_B1, 0x2a6b),
    ("CP_MEC_RS64_PRGRM_CNTR_START",       GC_B1, 0x2a68),
    ("CP_MEC_RS64_PRGRM_CNTR_START_HI",    GC_B1, 0x2a6a),
    ("CP_MEC_DOORBELL_RANGE_LOWER",        GC_B1, 0x2b29),
    ("CP_MEC_DOORBELL_RANGE_UPPER",        GC_B1, 0x2b2a),
]


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    c = IOKitClient(); c.open()
    info = c.get_info()
    print(f"device=0x{info.device_id:04x} rev=0x{info.revision_id:02x}")
    drv = _DriverShim(c)

    result = gfx_bring_up(c, drv, firmware_dir=FIRMWARE_DIR)

    print(f"\n== gfx_bring_up() complete ==")
    print(f"  BOOTLOAD_STATUS  = 0x{result.bootload_status:08x}")
    print(f"  autoload status  = 0x{result.autoload_rlc_status:08x}")
    print(f"  driver_table_mc  = 0x{result.driver_table_mc:x}")
    print(f"  enable_all resp  = {result.enable_all_resp}")
    if result.rejected_fw:
        print(f"  rejected {len(result.rejected_fw)} fw types:")
        for label, fw_type, status in result.rejected_fw:
            print(f"    {label} (type={fw_type}) status=0x{status:08x}")

    def rd(base, off): return c.mmio_read32(5, (base + off) * 4)

    def dump(title, regs):
        print(f"\n=== {title} ===")
        for name, base, off in regs:
            v = rd(base, off)
            print(f"  {name:40s} = 0x{v:08x}")

    dump("GFX / RLC / IMU", GFX_REGS)
    dump("GFXHUB (GC MC/VM)", GFXHUB_REGS)
    dump("MMHUB (MM MC/VM)", MMHUB_REGS)
    dump("MEC", MEC_REGS)


if __name__ == "__main__":
    main()
