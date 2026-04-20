"""Phase 7+8 smoke test: SMU(PWR_SOC) → MMHUB init → RLC/IMU load → SMU(PWR_GFX).

Hypothesis: SMU's FEATURE_PWR_GFX waits for the GFX engine to respond
during DPM enable, and the GFX engine needs RLC (which needs IMU to
program RLC's boot registers). With IMU + RLC loaded via PSP, this
should finally let PWR_GFX succeed.

Needs a fresh card (C2PMSG_81 == 0). Bails early otherwise.
"""
from __future__ import annotations

import logging
import os
import sys

from amd_gpu_driver.backends.macos.gfx_firmware import load_gfx_firmware
from amd_gpu_driver.backends.macos.gmc import init_mmhub
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


def _dump_features(c, label):
    _, lo = smu_send(c, PPSMC_MSG_GetRunningSmuFeaturesLo, 0)
    _, hi = smu_send(c, PPSMC_MSG_GetRunningSmuFeaturesHi, 0)
    print(f"  [{label}] RunningFeaturesLow=0x{lo:08x} High=0x{hi:08x}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    c = IOKitClient()
    c.open()
    info = c.get_info()
    print(f"device=0x{info.device_id:04x} rev=0x{info.revision_id:02x}")

    if c.mmio_read32(5, (0x16000 + 0x40 + 81) * 4) != 0:
        print("SOS already alive — replug first for a clean test.")
        sys.exit(0)

    drv = _DriverShim(c)

    # --- Step 1: SMU up to PWR_SOC. ---
    print("\n== Step 1: smu_bring_up(FEATURE_PWR_SOC) ==")
    result = smu_bring_up(c, drv, firmware_dir=FIRMWARE_DIR,
                          enable_domain=FEATURE_PWR_SOC)
    _dump_features(c, "after PWR_SOC")
    ring = result.ring

    # --- Step 2: MMHUB init. ---
    print("\n== Step 2: init_mmhub ==")
    init_mmhub(c)

    # --- Step 3: Load IMU + RLC via PSP. ---
    print("\n== Step 3: load IMU + RLC via PSP LOAD_IP_FW ==")
    try:
        load_gfx_firmware(c, drv, MP0_BASE_DW, ring, FIRMWARE_DIR)
    except (TimeoutError, RuntimeError, ValueError) as e:
        print(f"  GFX firmware load FAILED: {e}")
        sys.exit(1)

    # Sanity: SMU still responsive after all that?
    try:
        resp, _ = smu_send(c, PPSMC_MSG_DisallowGfxOff, 0, timeout_ms=1500)
        print(f"  SMU sanity DisallowGfxOff -> resp=0x{resp:x}")
    except TimeoutError as e:
        print(f"  SMU sanity TIMEOUT post-firmware-load: {e}")
        sys.exit(2)

    # --- Step 4: Try PWR_GFX. ---
    target = int(os.environ.get("SMU_FEATURE_PWR", str(FEATURE_PWR_GFX)))
    names = {0: "ALL", 3: "SOC", 4: "GFX"}
    label = names.get(target, str(target))
    print(f"\n== Step 4: EnableAllSmuFeatures(FEATURE_PWR_{label}) — 3 s timeout ==")
    try:
        resp, arg_out = smu_send(c, PPSMC_MSG_EnableAllSmuFeatures,
                                 target, timeout_ms=3000)
    except TimeoutError as e:
        print(f"  TIMEOUT: {e}")
        sys.exit(3)
    if resp != PPSMC_Result_OK:
        print(f"  FAILED: resp=0x{resp:x} arg_out=0x{arg_out:x}")
        sys.exit(4)
    print(f"  SUCCESS: resp=0x{resp:x} arg_out=0x{arg_out:x}")
    _dump_features(c, f"after PWR_{label}")


if __name__ == "__main__":
    main()
