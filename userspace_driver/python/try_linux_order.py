"""Linux-order bring-up: PSP → firmware in VRAM → AUTOLOAD_RLC → SMU EnableAll.

Reordering the flow to match what Linux amdgpu does on gfx12:
  1. PSP SOS alive + KM ring.
  2. PSP LOAD_IP_FW(SMU).
  3. SMU SetDriverDramAddr  (enable_domain=None — skip EnableAll).
  4. PSP LOAD_TOC.
  5. PSP LOAD_IP_FW(IMU_I, IMU_D).
  6. VRAM autoload buffer.
  7. PSP AUTOLOAD_RLC.
  8. SMU DisallowGfxOff.
  9. SMU EnableAllSmuFeatures(PWR_ALL=0) — single call, enables
     everything in one shot (Linux's smu_system_features_control).
 10. Poll RLC_RLCS_BOOTLOAD_STATUS for BOOTLOAD_COMPLETE.
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
    FEATURE_PWR_GFX,
    MP0_BASE_DW,
    PPSMC_MSG_DisallowGfxOff,
    PPSMC_MSG_EnableAllSmuFeatures,
    PPSMC_MSG_GetRunningSmuFeaturesHi,
    PPSMC_MSG_GetRunningSmuFeaturesLo,
    PPSMC_MSG_GetSmuVersion,
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


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    c = IOKitClient(); c.open()
    info = c.get_info()
    print(f"device=0x{info.device_id:04x} rev=0x{info.revision_id:02x}")

    if c.mmio_read32(5, (0x16000 + 0x40 + 81) * 4) != 0:
        print("SOS already alive — replug first.")
        sys.exit(0)

    drv = _DriverShim(c)

    # --- Step 1-3: SOS + SMU FW + SetDriverDramAddr (no EnableAll) ---
    print("\n== Step 1-3: PSP SOS + ring + SMU FW + SetDriverDramAddr ==")
    result = smu_bring_up(c, drv, firmware_dir=FIRMWARE_DIR,
                          enable_domain=None)

    ctx = alloc_cmd_ctx(drv)

    # --- Step 4: LOAD_TOC ---
    print("\n== Step 4: GFX_CMD_ID_LOAD_TOC ==")
    sos_blob = open(os.path.join(FIRMWARE_DIR, "psp_14_0_3_sos.bin"), "rb").read()
    toc_comp = next(c for c in parse_psp_firmware(sos_blob) if c.name == "TOC")
    submit_load_toc(c, drv, MP0_BASE_DW, result.ring, ctx, toc_comp.data)

    # --- Step 5: LOAD_IP_FW(IMU) ---
    print("\n== Step 5: LOAD_IP_FW(IMU_I, IMU_D) ==")
    imu_blob = open(os.path.join(FIRMWARE_DIR, "gc_12_0_1_imu.bin"), "rb").read()
    iram, dram = _extract_imu(imu_blob)
    _load_one(c, drv, MP0_BASE_DW, result.ring, ctx, iram, GFX_FW_TYPE_IMU_I, "IMU_I", strict=True)
    _load_one(c, drv, MP0_BASE_DW, result.ring, ctx, dram, GFX_FW_TYPE_IMU_D, "IMU_D", strict=True)

    # --- Step 6: VRAM autoload buffer ---
    print("\n== Step 6: build VRAM autoload buffer ==")
    with open(os.path.join(FIRMWARE_DIR, "gc_12_0_1_toc.bin"), "rb") as f:
        toc_blob = f.read()
    layout = plan_autoload(toc_blob)
    build_autoload_buffer(c, FIRMWARE_DIR, layout, toc_blob)

    # --- Step 7: AUTOLOAD_RLC ---
    print("\n== Step 7: GFX_CMD_ID_AUTOLOAD_RLC ==")
    resp = submit_autoload_rlc(c, drv, MP0_BASE_DW, result.ring, ctx)
    print(f"  status = 0x{resp['status']:08x}")

    # --- Step 8: DisallowGfxOff ---
    print("\n== Step 8: DisallowGfxOff ==")
    r, _ = smu_send(c, PPSMC_MSG_DisallowGfxOff, 0, timeout_ms=1500)
    print(f"  DisallowGfxOff -> resp=0x{r:x}")

    # --- Step 9: EnableAllSmuFeatures(PWR_ALL=0) ---
    print("\n== Step 9: EnableAllSmuFeatures(PWR_ALL=0) ==")
    target = int(os.environ.get("SMU_FEATURE_PWR", "0"))
    names = {0: "ALL", 3: "SOC", 4: "GFX"}
    try:
        resp, arg_out = smu_send(c, PPSMC_MSG_EnableAllSmuFeatures,
                                 target, timeout_ms=15000)
    except TimeoutError as e:
        print(f"  TIMEOUT: {e}")
        sys.exit(2)
    print(f"  EnableAll({names.get(target, target)}) -> resp=0x{resp:x} arg_out=0x{arg_out:x}")

    # --- Step 10: poll BOOTLOAD_COMPLETE ---
    print("\n== Step 10: poll BOOTLOAD_COMPLETE ==")
    GC_B0 = 0x1260
    GC_B1 = 0xA000
    def gc_rd(base, off): return c.mmio_read32(5, (base + off) * 4)
    deadline = time.time() + 10.0
    last = None
    while time.time() < deadline:
        core     = gc_rd(GC_B1, 0x40b6)
        reset    = gc_rd(GC_B1, 0x40bc)
        bootload = gc_rd(GC_B1, 0x4e7c)
        cp_stat  = gc_rd(GC_B0, 0x0f40)
        grbm     = gc_rd(GC_B0, 0x0da4)
        snap = (core, reset, bootload, cp_stat, grbm)
        if snap != last:
            t = time.time() - deadline + 10
            print(f"  t={t:5.2f}s CORE=0x{core:08x} RESET=0x{reset:08x} "
                  f"BOOTLOAD=0x{bootload:08x} CP_STAT=0x{cp_stat:08x} GRBM=0x{grbm:08x}")
            last = snap
        if (bootload & 0x80000000) and cp_stat == 0:
            print("  BOOTLOAD_COMPLETE ✓")
            break
        time.sleep(0.05)

    # Final status
    _, lo = smu_send(c, PPSMC_MSG_GetRunningSmuFeaturesLo, 0, timeout_ms=2000)
    _, hi = smu_send(c, PPSMC_MSG_GetRunningSmuFeaturesHi, 0, timeout_ms=2000)
    print(f"\nFinal RunningFeatures: low=0x{lo:08x} high=0x{hi:08x}")


if __name__ == "__main__":
    main()
