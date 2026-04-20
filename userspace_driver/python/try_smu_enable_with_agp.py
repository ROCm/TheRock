"""Smoke-test for `amd_gpu_driver.backends.macos.smu.smu_bring_up`.

Run after a fresh DEXT / power-cycle of the eGPU:

    SMU_FEATURE_PWR=3 python3 try_smu_enable_with_agp.py

Expected on a working setup:
  - SOS loads (or is already alive).
  - SMU firmware loads.
  - EnableAllSmuFeatures(FEATURE_PWR_SOC=3) returns OK in a few ms.
  - RunningFeaturesLow/High report SOC DPM bits active.
  - DisallowGfxOff / AllowGfxOff both ACK.

See `memory/smu-feature-enable-attempt.md` for the theory.
"""
from __future__ import annotations

import logging
import os
import sys

from amd_gpu_driver.backends.macos.iokit_client import IOKitClient
from amd_gpu_driver.backends.macos.smu import (
    FEATURE_PWR_SOC,
    PPSMC_MSG_AllowGfxOff,
    PPSMC_MSG_DisallowGfxOff,
    PPSMC_Result_OK,
    smu_bring_up,
    smu_send,
)

FIRMWARE_DIR = os.path.expanduser("~/firmware/linux-firmware/amdgpu")


class _DriverShim:
    """Minimal adapter so `smu_bring_up` can run against a raw IOKitClient."""
    def __init__(self, client): self.client = client
    def alloc_dma(self, size):
        dma = self.client.alloc_dma(size)
        bus = dma.segments[0][0] if dma.segments else 0
        return (dma.cpu_addr, bus, dma.buffer_id)
    def free_dma(self, h): self.client.free_dma(h)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    c = IOKitClient()
    c.open()
    info = c.get_info()
    print(f"device=0x{info.device_id:04x} rev=0x{info.revision_id:02x} "
          f"vram={info.vram_size // (1024 * 1024)}MB")

    # The recipe is not idempotent: re-running after SMU features are
    # already enabled leaves PSP refusing to re-load SMU via
    # LOAD_IP_FW. Detect that case and bail out with guidance instead
    # of timing out on the PSP fence and hanging the SMU.
    MP1 = 0x16200
    smu_mailbox_alive = c.mmio_read32(5, (MP1 + 0x40 + 90) * 4) != 0
    if smu_mailbox_alive:
        print("SMU mailbox already has a nonzero C2PMSG_90 — SMU is likely "
              "already bootstrapped from a previous run. Unplug/replug "
              "(or sudo-kill the DEXT) before rerunning this smoke test.")
        sys.exit(0)

    drv = _DriverShim(c)
    domain = int(os.environ.get("SMU_FEATURE_PWR", str(FEATURE_PWR_SOC)))

    try:
        result = smu_bring_up(c, drv,
                              firmware_dir=FIRMWARE_DIR,
                              enable_domain=domain)
    except TimeoutError as e:
        print(f"SMU bring-up timed out: {e}")
        print("SMU is now hung; unplug/replug the eGPU to recover.")
        sys.exit(2)
    except RuntimeError as e:
        print(f"SMU bring-up failed: {e}")
        sys.exit(1)

    print()
    print(f"SMU version           = 0x{result.smu_version:08x}")
    print(f"driver_table MC       = 0x{result.driver_table_mc:x}")
    print(f"RunningFeaturesLow    = 0x{result.running_features_low:08x}")
    print(f"RunningFeaturesHigh   = 0x{result.running_features_high:08x}")

    # Exercise the GfxOff mailbox — this is the phase 8.5c goal.
    try:
        resp, _ = smu_send(c, PPSMC_MSG_DisallowGfxOff, 0)
        print(f"DisallowGfxOff        -> resp=0x{resp:x} (expected 0x{PPSMC_Result_OK:x})")
        resp, _ = smu_send(c, PPSMC_MSG_AllowGfxOff, 0)
        print(f"AllowGfxOff           -> resp=0x{resp:x}")
    except TimeoutError as e:
        print(f"GfxOff mailbox probe timed out: {e}")
        sys.exit(3)


if __name__ == "__main__":
    main()
