"""Phase 7 bring-up smoke test: SMU(PWR_SOC) → MMHUB init → SMU(PWR_GFX).

Starts from a freshly reset card:
  1. Run smu_bring_up(enable_domain=FEATURE_PWR_SOC)  — known to work.
  2. init_mmhub()                                      — Phase 7 experiment.
  3. Try EnableAllSmuFeatures(FEATURE_PWR_GFX=4)       — previously hung.

If step 3 returns OK, the MMHUB state is sufficient for SMU's GFX
power domain. If it hangs, the MMHUB init is missing something (likely
the VM_CONTEXT0 page table that tinygrad builds).

Bailing out early if the SMU is already up to avoid the non-idempotent
PSP re-load path.
"""
from __future__ import annotations

import logging
import os
import sys

from amd_gpu_driver.backends.macos.gmc import init_mmhub
from amd_gpu_driver.backends.macos.iokit_client import IOKitClient
from amd_gpu_driver.backends.macos.smu import (
    FEATURE_PWR_ALL,
    FEATURE_PWR_GFX,
    FEATURE_PWR_SOC,
    PPSMC_MSG_EnableAllSmuFeatures,
    PPSMC_MSG_GetRunningSmuFeaturesHi,
    PPSMC_MSG_GetRunningSmuFeaturesLo,
    PPSMC_Result_OK,
    smu_bring_up,
    smu_send,
)

FIRMWARE_DIR = os.path.expanduser("~/firmware/linux-firmware/amdgpu")
MP1 = 0x16200


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

    # Fresh-card gate: PSP C2PMSG_81 (SOS sign-of-life) is the
    # authoritative "no previous bring-up" signal. MP1 C2PMSG_90 can
    # carry a leftover 0x1 across replugs, so it's not reliable alone.
    if c.mmio_read32(5, (0x16000 + 0x40 + 81) * 4) != 0:
        print("SOS already alive — replug first for a clean test.")
        sys.exit(0)

    drv = _DriverShim(c)

    # --- Step 1: bring SMU up to PWR_SOC (known good). ---
    print("\n== Step 1: smu_bring_up(FEATURE_PWR_SOC) ==")
    try:
        smu_bring_up(c, drv, firmware_dir=FIRMWARE_DIR,
                     enable_domain=FEATURE_PWR_SOC)
    except (TimeoutError, RuntimeError) as e:
        print(f"  SMU bring-up FAILED: {e}")
        sys.exit(1)
    _dump_features(c, "after PWR_SOC")

    # --- Step 2: MMHUB init. ---
    print("\n== Step 2: init_mmhub ==")
    cfg = init_mmhub(c)
    print(f"  cfg = {cfg}")

    # --- Step 3: Try PWR_GFX. ---
    target = int(os.environ.get("SMU_FEATURE_PWR", str(FEATURE_PWR_GFX)))
    names = {0: "ALL", 3: "SOC", 4: "GFX"}
    label = names.get(target, str(target))
    print(f"\n== Step 3: EnableAllSmuFeatures(FEATURE_PWR_{label}) — 3 s timeout ==")
    try:
        resp, arg_out = smu_send(c, PPSMC_MSG_EnableAllSmuFeatures,
                                 target, timeout_ms=3000)
    except TimeoutError as e:
        print(f"  TIMEOUT: {e}")
        print("  MMHUB init not sufficient for this feature domain.")
        sys.exit(2)
    if resp != PPSMC_Result_OK:
        print(f"  FAILED: resp=0x{resp:x} arg_out=0x{arg_out:x}")
        sys.exit(3)
    print(f"  SUCCESS: resp=0x{resp:x} arg_out=0x{arg_out:x}")
    _dump_features(c, f"after PWR_{label}")


if __name__ == "__main__":
    main()
