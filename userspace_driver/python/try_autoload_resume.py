"""Resume autoload test after smu_bring_up already completed.

Skips the SMU bring-up step (which left SMU at PWR_SOC from a previous
run) and jumps straight to: parse TOC → fill VRAM → IMU boot →
SMU(PWR_GFX). Use after `try_autoload_gfx.py` failed mid-build but
SMU is still alive.
"""
from __future__ import annotations

import logging
import os
import sys

from amd_gpu_driver.backends.macos.gfx_autoload import (
    build_autoload_buffer,
    plan_autoload,
    run_imu_boot,
)
from amd_gpu_driver.backends.macos.iokit_client import IOKitClient
from amd_gpu_driver.backends.macos.smu import (
    FEATURE_PWR_GFX,
    PPSMC_MSG_EnableAllSmuFeatures,
    PPSMC_MSG_GetRunningSmuFeaturesHi,
    PPSMC_MSG_GetRunningSmuFeaturesLo,
    PPSMC_MSG_GetSmuVersion,
    PPSMC_Result_OK,
    smu_send,
)

FIRMWARE_DIR = os.path.expanduser("~/firmware/linux-firmware/amdgpu")


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    c = IOKitClient(); c.open()

    # Check SMU is alive (PWR_SOC should already be enabled from prior run).
    try:
        resp, ver = smu_send(c, PPSMC_MSG_GetSmuVersion, 0, timeout_ms=500)
        print(f"SMU alive: version=0x{ver:x} (resp=0x{resp:x})")
    except TimeoutError:
        print("SMU not responsive — run try_autoload_gfx.py from a fresh card.")
        sys.exit(1)

    with open(os.path.join(FIRMWARE_DIR, "gc_12_0_1_toc.bin"), "rb") as f:
        toc_blob = f.read()
    layout = plan_autoload(toc_blob)
    print(f"Autoload buffer: 0x{layout.buffer_size:x} "
          f"({layout.buffer_size // (1024*1024)} MB)")

    print("\n== Build autoload buffer ==")
    build_autoload_buffer(c, FIRMWARE_DIR, layout, toc_blob)

    print("\n== IMU boot ==")
    try:
        run_imu_boot(c, FIRMWARE_DIR, layout)
    except TimeoutError as e:
        print(f"  IMU boot TIMEOUT: {e}")
        sys.exit(2)

    print("\n== EnableAllSmuFeatures(PWR_GFX) ==")
    try:
        resp, arg_out = smu_send(c, PPSMC_MSG_EnableAllSmuFeatures,
                                 FEATURE_PWR_GFX, timeout_ms=3000)
    except TimeoutError as e:
        print(f"  SMU TIMEOUT: {e}")
        sys.exit(3)
    if resp != PPSMC_Result_OK:
        print(f"  FAILED: resp=0x{resp:x} arg_out=0x{arg_out:x}")
        sys.exit(4)
    print(f"  SUCCESS: resp=0x{resp:x} arg_out=0x{arg_out:x}")

    _, lo = smu_send(c, PPSMC_MSG_GetRunningSmuFeaturesLo, 0)
    _, hi = smu_send(c, PPSMC_MSG_GetRunningSmuFeaturesHi, 0)
    print(f"  RunningFeaturesLow  = 0x{lo:08x}")
    print(f"  RunningFeaturesHigh = 0x{hi:08x}")


if __name__ == "__main__":
    main()
