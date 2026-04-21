"""Try EnableAll with the real allowed-features mask from SMU's pptable.

Previous run asked SMU to write its built-in combo pptable and we
parsed out FeaturesToRun[0..3]:
  [0] = 0x3AFFFCFB   (SetAllowedFeaturesMaskLow)
  [1] = 0x0488F19E   (SetAllowedFeaturesMaskHigh)
  [2] = 0x0003FFFF   (> 64 bits — ignored by SetAllowedMask)

Those specific bits (not 0xFFFFFFFF) are what Linux would end up
sending after parsing pptable + init_allowed_features →
smu_feature_list_to_arr32. SetAllowedMask with 0xFFFFFFFF got
rejected 0xFD (CmdRejectedPrereq); the real mask should be
accepted.

Sequence:
  1. smu_bring_up(enable_domain=None).
  2. Full PSP autoload (LOAD_TOC + LOAD_IP_FW(IMU) + VRAM + AUTOLOAD_RLC).
  3. UseDefaultPPTable.
  4. TransferTableSmu2Dram(COMBO_PPTABLE=1) — re-reads the pptable into
     our driver table; also re-derive the mask so we don't rely on
     the hard-coded values above if SMU changes them.
  5. SetAllowedFeaturesMaskLow/High with the parsed mask.
  6. EnableAll(PWR_ALL=0).
  7. Poll RLC_RLCS_BOOTLOAD_STATUS for BOOTLOAD_COMPLETE.
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
    PPSMC_MSG_TransferTableSmu2Dram,
    PPSMC_MSG_UseDefaultPPTable,
    smu_bring_up,
    smu_send,
)

FIRMWARE_DIR = os.path.expanduser("~/firmware/linux-firmware/amdgpu")
TABLE_COMBO_PPTABLE = 1


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
        print(f"  {name:48s} resp=0x{r:x} arg_out=0x{a:x}")
        return r, a
    except TimeoutError:
        print(f"  {name:48s} TIMEOUT")
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
    tbl_vram_off = result.driver_table_mc - fb_base
    print(f"  driver_table VRAM offset = 0x{tbl_vram_off:x}")

    ctx = alloc_cmd_ctx(drv)

    # Full autoload.
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
    autoload_resp = submit_autoload_rlc(c, drv, MP0_BASE_DW, result.ring, ctx)
    print(f"  AUTOLOAD_RLC status = 0x{autoload_resp['status']:08x}")

    # Use default pptable + transfer it to VRAM.
    print("\n== 3: UseDefaultPPTable + TransferTableSmu2Dram(COMBO_PPTABLE) ==")
    _try(c, PPSMC_MSG_UseDefaultPPTable, 0, "UseDefaultPPTable")
    _try(c, PPSMC_MSG_TransferTableSmu2Dram, TABLE_COMBO_PPTABLE,
         "TransferTableSmu2Dram(COMBO_PPTABLE=1)", timeout=5000)

    # Read pptable header + FeaturesToRun from VRAM via BAR0.
    print("\n== 4: parse pptable header for pmfw_pptable offset ==")
    bar0_cpu, _ = c.map_bar(0)
    base = bar0_cpu + tbl_vram_off
    hdr = b"".join(struct.pack("<I", (ctypes.c_uint32 * 1).from_address(base + i * 4)[0])
                   for i in range(8))
    (struct_size, fmt_rev, _content_rev, _tab_rev, _src,
     pp_off, _pp_size, _sku_off, _sku_size,
     _brd_off, _brd_size, _csku_off, _csku_size) = struct.unpack_from("<HBBBBHHHHHHHH", hdr, 0)
    print(f"  struct_size={struct_size} fmt_rev={fmt_rev} pmfw_pptable_start=0x{pp_off:x}")

    # FeaturesToRun at pp_off + 4 (skipping the 4-byte Version + Spare header).
    feat_low  = (ctypes.c_uint32 * 1).from_address(base + pp_off + 4)[0]
    feat_high = (ctypes.c_uint32 * 1).from_address(base + pp_off + 8)[0]
    feat_hi2  = (ctypes.c_uint32 * 1).from_address(base + pp_off + 12)[0]
    print(f"  FeaturesToRun[0] = 0x{feat_low:08x}  (SetAllowedMaskLow)")
    print(f"  FeaturesToRun[1] = 0x{feat_high:08x}  (SetAllowedMaskHigh)")
    print(f"  FeaturesToRun[2] = 0x{feat_hi2:08x}   (beyond 64 bits)")

    # Send the real mask.
    print("\n== 5: SetAllowedFeaturesMask with parsed values ==")
    _try(c, PPSMC_MSG_SetAllowedFeaturesMaskHigh, feat_high,
         f"SetAllowedFeaturesMaskHigh(0x{feat_high:x})")
    _try(c, PPSMC_MSG_SetAllowedFeaturesMaskLow, feat_low,
         f"SetAllowedFeaturesMaskLow (0x{feat_low:x})")

    # Enable all.
    print("\n== 6: EnableAll(PWR_ALL=0) ==")
    resp, _ = _try(c, PPSMC_MSG_EnableAllSmuFeatures, FEATURE_PWR_ALL,
                   "EnableAll(PWR_ALL=0)", timeout=10000)
    if resp is None:
        print("\n  SMU hung on EnableAll. Current card state:")

    # Poll BOOTLOAD_COMPLETE.
    print("\n== 7: poll BOOTLOAD_COMPLETE (5s) ==")
    GC_B1 = 0xA000
    def gc_rd(o): return c.mmio_read32(5, (GC_B1 + o) * 4)
    deadline = time.time() + 5
    last = None
    while time.time() < deadline:
        bl = gc_rd(0x4e7c); rst = gc_rd(0x40bc); core = gc_rd(0x40b6)
        if (bl, rst, core) != last:
            print(f"  t={time.time()-deadline+5:5.2f}s CORE=0x{core:x} RESET=0x{rst:08x} BOOTLOAD=0x{bl:08x}")
            last = (bl, rst, core)
        if bl & 0x80000000:
            print("  BOOTLOAD_COMPLETE ✓")
            break
        time.sleep(0.05)

    print("\n== Final ==")
    _try(c, PPSMC_MSG_GetRunningSmuFeaturesLo, 0, "GetRunningFeaturesLo", timeout=1500)
    _try(c, PPSMC_MSG_GetRunningSmuFeaturesHi, 0, "GetRunningFeaturesHi", timeout=1500)


if __name__ == "__main__":
    main()
