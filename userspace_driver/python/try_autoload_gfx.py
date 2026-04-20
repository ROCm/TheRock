"""Phase 7+8 bring-up: gfx12 backdoor autoload → IMU start → SMU(PWR_GFX).

Requires a freshly reset card. Sequence:
  1. smu_bring_up(FEATURE_PWR_SOC)          # proven good
  2. Parse gc_12_0_1_toc.bin.
  3. Copy all GFX firmware payloads into the VRAM autoload buffer
     (BAR0-mapped CPU region) at TOC-specified offsets.
  4. Program GFX_IMU_RLC_BOOTLOADER_ADDR/SIZE.
  5. Stream IMU IRAM/DRAM into IMU SRAM via MMIO.
  6. Unhalt IMU; wait for IMU_GFX_RESET_CTRL ready.
  7. Try EnableAllSmuFeatures(FEATURE_PWR_GFX).
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
    FEATURE_PWR_SOC,
    PPSMC_MSG_EnableAllSmuFeatures,
    PPSMC_MSG_GetRunningSmuFeaturesHi,
    PPSMC_MSG_GetRunningSmuFeaturesLo,
    PPSMC_Result_OK,
    smu_bring_up,
    smu_send,
)

FIRMWARE_DIR = os.path.expanduser("~/firmware/linux-firmware/amdgpu")
TOC_FW = "gc_12_0_1_toc.bin"


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
    smu_bring_up(c, drv, firmware_dir=FIRMWARE_DIR,
                 enable_domain=FEATURE_PWR_SOC)

    print("\n== Step 2: parse gc_12_0_1_toc.bin ==")
    with open(os.path.join(FIRMWARE_DIR, TOC_FW), "rb") as f:
        toc_blob = f.read()
    layout = plan_autoload(toc_blob)
    print(f"  autoload buffer size: 0x{layout.buffer_size:x} "
          f"({layout.buffer_size // (1024*1024)} MB)")
    print(f"  RLC_G at offset 0x{layout.rlc_g_offset:x}  "
          f"size=0x{layout.rlc_g_size:x}")

    print("\n== Step 3: build autoload buffer in VRAM (via BAR0) ==")
    build_autoload_buffer(c, FIRMWARE_DIR, layout, toc_blob)

    print("\n== Step 4-8: IMU boot ==")
    try:
        run_imu_boot(c, FIRMWARE_DIR, layout)
    except TimeoutError as e:
        print(f"  IMU boot TIMEOUT: {e}")
        sys.exit(2)

    print("\n== Step 9: EnableAllSmuFeatures(FEATURE_PWR_GFX) — 3 s ==")
    try:
        resp, arg_out = smu_send(c, PPSMC_MSG_EnableAllSmuFeatures,
                                 FEATURE_PWR_GFX, timeout_ms=3000)
    except TimeoutError as e:
        print(f"  SMU TIMEOUT: {e}")
        sys.exit(3)
    if resp != PPSMC_Result_OK:
        print(f"  FAILED: resp=0x{resp:x}  arg_out=0x{arg_out:x}")
        sys.exit(4)
    print(f"  SUCCESS: resp=0x{resp:x}  arg_out=0x{arg_out:x}")

    _, lo = smu_send(c, PPSMC_MSG_GetRunningSmuFeaturesLo, 0)
    _, hi = smu_send(c, PPSMC_MSG_GetRunningSmuFeaturesHi, 0)
    print(f"  RunningFeaturesLow  = 0x{lo:08x}")
    print(f"  RunningFeaturesHigh = 0x{hi:08x}")


if __name__ == "__main__":
    main()
