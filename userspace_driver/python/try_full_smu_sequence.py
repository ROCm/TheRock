"""Fresh Linux-order flow with SetAllowedMask + UseDefaultPPTable.

Prior run confirmed autoload gets us to IMU+RLC up with 6 of 7 bootload
bits set, but EnableAllSmuFeatures(PWR_ALL=0) hangs.

Linux's `smu_smc_hw_setup` calls many SMU pre-Enable steps we were
skipping. The two most likely needed for our case:
  - PPSMC_MSG_SetAllowedFeaturesMaskLow/High (0x04 / 0x05) — sets
    which features SMU is allowed to enable. Without this the
    allowed mask is 0 and EnableAll has nothing to enable.
  - PPSMC_MSG_UseDefaultPPTable (0x14) — tell SMU to use its
    built-in default power-play table (bypasses the vBIOS pptable
    extraction that Linux does via ATOM BIOS).

Sequence:
  1. smu_bring_up (enable_domain=None — no EnableAll).
  2. UseDefaultPPTable.
  3. SetAllowedMaskLow/High = 0xFFFFFFFF (enable everything).
  4. LOAD_TOC + LOAD_IP_FW(IMU) + VRAM autoload + AUTOLOAD_RLC.
  5. Poll BOOTLOAD_COMPLETE.
  6. EnableAllSmuFeatures(PWR_SOC=3) first (known-working);
     if OK, try (PWR_GFX=4) then (PWR_ALL=0).
"""
from __future__ import annotations

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
    FEATURE_PWR_GFX,
    FEATURE_PWR_SOC,
    MP0_BASE_DW,
    PPSMC_MSG_DisallowGfxOff,
    PPSMC_MSG_EnableAllSmuFeatures,
    PPSMC_MSG_GetRunningSmuFeaturesHi,
    PPSMC_MSG_GetRunningSmuFeaturesLo,
    PPSMC_MSG_SetAllowedFeaturesMaskHigh,
    PPSMC_MSG_SetAllowedFeaturesMaskLow,
    PPSMC_MSG_UseDefaultPPTable,
    PPSMC_Result_OK,
    smu_bring_up,
    smu_send,
)

FIRMWARE_DIR = os.path.expanduser("~/firmware/linux-firmware/amdgpu")


class _DriverShim:
    def __init__(self, client): self.client = client
    def alloc_dma(self, size):
        dma = self.client.alloc_dma(size)
        bus = dma.segments[0][0] if dma.segments else 0
        return (dma.cpu_addr, bus, dma.buffer_id)
    def free_dma(self, h): self.client.free_dma(h)


def _try_smu(c, msg, arg, name, *, timeout=3000):
    try:
        r, a = smu_send(c, msg, arg, timeout_ms=timeout)
        print(f"  {name:40s} resp=0x{r:x} arg_out=0x{a:x}")
        return r
    except TimeoutError:
        print(f"  {name:40s} TIMEOUT")
        return None


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    c = IOKitClient(); c.open()
    info = c.get_info()
    print(f"device=0x{info.device_id:04x} rev=0x{info.revision_id:02x}")
    if c.mmio_read32(5, (0x16000 + 0x40 + 81) * 4) != 0:
        print("SOS already alive — replug first.")
        sys.exit(0)
    drv = _DriverShim(c)

    # 1. SOS + ring + SMU FW + SetDriverDramAddr, no EnableAll.
    print("\n== 1: smu_bring_up (no EnableAll) ==")
    r = smu_bring_up(c, drv, firmware_dir=FIRMWARE_DIR, enable_domain=None)

    # 2. UseDefaultPPTable — avoid pptable extraction from vBIOS.
    print("\n== 2: PPSMC_MSG_UseDefaultPPTable ==")
    _try_smu(c, PPSMC_MSG_UseDefaultPPTable, 0, "UseDefaultPPTable")

    # 3. SetAllowedFeaturesMaskLow/High = 0xFFFFFFFF — enable everything.
    print("\n== 3: SetAllowedFeaturesMaskLow/High = 0xFFFFFFFF ==")
    _try_smu(c, PPSMC_MSG_SetAllowedFeaturesMaskHigh, 0xFFFFFFFF,
             "SetAllowedFeaturesMaskHigh(0xFFFFFFFF)")
    _try_smu(c, PPSMC_MSG_SetAllowedFeaturesMaskLow, 0xFFFFFFFF,
             "SetAllowedFeaturesMaskLow (0xFFFFFFFF)")

    ctx = alloc_cmd_ctx(drv)

    # 4a. LOAD_TOC
    print("\n== 4a: LOAD_TOC ==")
    sos_blob = open(os.path.join(FIRMWARE_DIR, "psp_14_0_3_sos.bin"), "rb").read()
    toc_comp = next(c2 for c2 in parse_psp_firmware(sos_blob) if c2.name == "TOC")
    submit_load_toc(c, drv, MP0_BASE_DW, r.ring, ctx, toc_comp.data)

    # 4b. LOAD_IP_FW(IMU)
    print("\n== 4b: LOAD_IP_FW(IMU_I, IMU_D) ==")
    imu_blob = open(os.path.join(FIRMWARE_DIR, "gc_12_0_1_imu.bin"), "rb").read()
    iram, dram = _extract_imu(imu_blob)
    _load_one(c, drv, MP0_BASE_DW, r.ring, ctx, iram, GFX_FW_TYPE_IMU_I, "IMU_I", strict=True)
    _load_one(c, drv, MP0_BASE_DW, r.ring, ctx, dram, GFX_FW_TYPE_IMU_D, "IMU_D", strict=True)

    # 4c. VRAM autoload buffer
    print("\n== 4c: VRAM autoload buffer ==")
    with open(os.path.join(FIRMWARE_DIR, "gc_12_0_1_toc.bin"), "rb") as f:
        toc_blob = f.read()
    layout = plan_autoload(toc_blob)
    build_autoload_buffer(c, FIRMWARE_DIR, layout, toc_blob)

    # 4d. AUTOLOAD_RLC
    print("\n== 4d: GFX_CMD_ID_AUTOLOAD_RLC ==")
    resp = submit_autoload_rlc(c, drv, MP0_BASE_DW, r.ring, ctx)
    print(f"  status = 0x{resp['status']:08x}")

    # 5. Poll BOOTLOAD_COMPLETE (5s — it may or may not fire before EnableAll)
    print("\n== 5: poll BOOTLOAD_COMPLETE (5s) ==")
    GC_B0 = 0x1260; GC_B1 = 0xA000
    def gc_rd(b, o): return c.mmio_read32(5, (b+o)*4)
    deadline = time.time() + 5
    last = None
    while time.time() < deadline:
        bl = gc_rd(GC_B1, 0x4e7c)
        rst = gc_rd(GC_B1, 0x40bc)
        cp = gc_rd(GC_B0, 0x0f40)
        if (bl, rst, cp) != last:
            print(f"  t={time.time()-deadline+5:5.2f}s BOOTLOAD=0x{bl:08x} RESET=0x{rst:08x} CP=0x{cp:08x}")
            last = (bl, rst, cp)
        if bl & 0x80000000:
            print("  BOOTLOAD_COMPLETE ✓")
            break
        time.sleep(0.05)

    # 6. Try EnableAll in escalating domains
    print("\n== 6: EnableAllSmuFeatures attempts ==")
    _try_smu(c, PPSMC_MSG_EnableAllSmuFeatures, FEATURE_PWR_SOC, "EnableAll(PWR_SOC=3)", timeout=5000)
    _try_smu(c, PPSMC_MSG_EnableAllSmuFeatures, FEATURE_PWR_GFX, "EnableAll(PWR_GFX=4)", timeout=5000)
    _try_smu(c, PPSMC_MSG_EnableAllSmuFeatures, FEATURE_PWR_ALL, "EnableAll(PWR_ALL=0)", timeout=5000)

    # Final feature snapshot.
    print("\n== Final ==")
    _try_smu(c, PPSMC_MSG_GetRunningSmuFeaturesLo, 0, "GetRunningFeaturesLo", timeout=1500)
    _try_smu(c, PPSMC_MSG_GetRunningSmuFeaturesHi, 0, "GetRunningFeaturesHi", timeout=1500)
    # Re-read BOOTLOAD_STATUS
    print(f"  BOOTLOAD_STATUS = 0x{gc_rd(GC_B1, 0x4e7c):08x}")
    print(f"  RLC_CNTL        = 0x{gc_rd(GC_B1, 0x4c00):08x}")


if __name__ == "__main__":
    main()
