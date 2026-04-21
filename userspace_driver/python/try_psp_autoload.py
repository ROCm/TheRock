"""PSP-driven gfx12 autoload smoke test.

Alternative to try_autoload_gfx.py — this path lets PSP handle the
entire backdoor autoload (VRAM layout + IMU stream + IMU start + RLC
wake). We just feed the firmware via LOAD_IP_FW and kick
GFX_CMD_ID_AUTOLOAD_RLC.

Sequence:
  1. smu_bring_up(PWR_SOC)
  2. psp_load_gfx_and_autoload → loads SDMA/RLC/RS64 CP/MES/IMU via
     PSP LOAD_IP_FW, then sends AUTOLOAD_RLC
  3. EnableAllSmuFeatures(PWR_GFX) / PWR_ALL
"""
from __future__ import annotations

import logging
import os
import sys

from amd_gpu_driver.backends.macos.gfx_psp_autoload import psp_load_gfx_and_autoload
from amd_gpu_driver.backends.macos.iokit_client import IOKitClient
from amd_gpu_driver.backends.macos.smu import (
    FEATURE_PWR_ALL,
    FEATURE_PWR_GFX,
    FEATURE_PWR_SOC,
    MP0_BASE_DW,
    PPSMC_MSG_DisallowGfxOff,
    PPSMC_MSG_EnableAllSmuFeatures,
    PPSMC_MSG_GetRunningSmuFeaturesHi,
    PPSMC_MSG_GetRunningSmuFeaturesLo,
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
        print("SOS already alive — replug first for a clean test.")
        sys.exit(0)

    drv = _DriverShim(c)

    print("\n== Step 1: smu_bring_up(FEATURE_PWR_SOC) ==")
    result = smu_bring_up(c, drv, firmware_dir=FIRMWARE_DIR,
                          enable_domain=FEATURE_PWR_SOC)

    # Keep GFX power rail on before the autoload kick.
    print("\n== Step 1.5: DisallowGfxOff ==")
    r, _ = smu_send(c, PPSMC_MSG_DisallowGfxOff, 0, timeout_ms=1500)
    print(f"  DisallowGfxOff -> resp=0x{r:x}")

    print("\n== Step 2: PSP LOAD_IP_FW chain + AUTOLOAD_RLC ==")
    try:
        psp_load_gfx_and_autoload(c, drv, MP0_BASE_DW, result.ring,
                                  FIRMWARE_DIR)
    except (TimeoutError, RuntimeError) as e:
        print(f"  FAILED: {e}")
        sys.exit(1)

    # --- Step 3: observe RLC + SMU state after autoload ---
    print("\n== Step 3: post-autoload state ==")
    GC_B1 = 0xA000
    def gc_rd(off):
        return c.mmio_read32(5, (GC_B1 + off) * 4)
    print(f"  GFX_IMU_CORE_CTRL       = 0x{gc_rd(0x40b6):08x}")
    print(f"  GFX_IMU_GFX_RESET_CTRL  = 0x{gc_rd(0x40bc):08x}")
    print(f"  RLC_CNTL                = 0x{gc_rd(0x4c00):08x}")
    print(f"  RLC_GPM_THREAD_ENABLE   = 0x{gc_rd(0x4c45):08x}")
    # RLC_RLCS_BOOTLOAD_STATUS is what tinygrad polls; offset unknown
    # to me offhand but we can at least see if GRBM/GC feels alive.
    print(f"  GRBM_STATUS (0xda4 idx0) = 0x{c.mmio_read32(5, (0x1260 + 0x0da4) * 4):08x}")

    # --- Step 4: try PWR_GFX ---
    target = int(os.environ.get("SMU_FEATURE_PWR", str(FEATURE_PWR_GFX)))
    names = {0: "ALL", 3: "SOC", 4: "GFX"}
    print(f"\n== Step 4: EnableAllSmuFeatures(FEATURE_PWR_{names.get(target, target)}) ==")
    try:
        resp, arg_out = smu_send(c, PPSMC_MSG_EnableAllSmuFeatures,
                                 target, timeout_ms=5000)
    except TimeoutError as e:
        print(f"  SMU TIMEOUT: {e}")
        sys.exit(2)
    if resp != PPSMC_Result_OK:
        print(f"  FAILED: resp=0x{resp:x} arg_out=0x{arg_out:x}")
        sys.exit(3)
    print(f"  SUCCESS: resp=0x{resp:x} arg_out=0x{arg_out:x}")

    _, lo = smu_send(c, PPSMC_MSG_GetRunningSmuFeaturesLo, 0)
    _, hi = smu_send(c, PPSMC_MSG_GetRunningSmuFeaturesHi, 0)
    print(f"  RunningFeaturesLow  = 0x{lo:08x}")
    print(f"  RunningFeaturesHigh = 0x{hi:08x}")


if __name__ == "__main__":
    main()
