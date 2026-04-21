"""Hybrid autoload test:
  - VRAM autoload buffer (gfx_autoload.build_autoload_buffer) holds
    RLC + SDMA + RS64 CP + MES payloads, since PSP refuses those
    fw_types via LOAD_IP_FW on this ASIC.
  - PSP LOAD_IP_FW is used for IMU_I + IMU_D (the only non-SMU types
    PSP accepts here — verified by try_psp_autoload_resume.py).
  - Then GFX_CMD_ID_AUTOLOAD_RLC kicks PSP into doing the actual
    backdoor autoload internally — it programs GFX_IMU_RLC_BOOTLOADER_ADDR
    and unhalts IMU from the secure side, where our GC writes are
    blocked.

Expected: no more host-side GC register writes for this path.
"""
from __future__ import annotations

import ctypes
import logging
import os
import sys

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
from amd_gpu_driver.backends.macos.psp_bootloader import parse_psp_firmware
from amd_gpu_driver.backends.macos.iokit_client import IOKitClient
from amd_gpu_driver.backends.macos.psp_cmd import (
    GFX_FW_TYPE_IMU_D,
    GFX_FW_TYPE_IMU_I,
    alloc_cmd_ctx,
)
from amd_gpu_driver.backends.macos.smu import (
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

    print("\n== Step 1.5: DisallowGfxOff ==")
    r, _ = smu_send(c, PPSMC_MSG_DisallowGfxOff, 0, timeout_ms=1500)
    print(f"  DisallowGfxOff -> resp=0x{r:x}")

    print("\n== Step 2: parse gc_12_0_1_toc.bin ==")
    with open(os.path.join(FIRMWARE_DIR, "gc_12_0_1_toc.bin"), "rb") as f:
        toc_blob = f.read()
    layout = plan_autoload(toc_blob)
    print(f"  autoload buffer size = 0x{layout.buffer_size:x}")

    print("\n== Step 3: build VRAM autoload buffer (via BAR0) ==")
    build_autoload_buffer(c, FIRMWARE_DIR, layout, toc_blob)

    ctx = alloc_cmd_ctx(drv)

    print("\n== Step 4a: GFX_CMD_ID_LOAD_TOC (PSP TOC from SOS container) ==")
    sos_blob = open(os.path.join(FIRMWARE_DIR, "psp_14_0_3_sos.bin"), "rb").read()
    toc_comp = next(
        (c for c in parse_psp_firmware(sos_blob) if c.name == "TOC"),
        None,
    )
    if toc_comp is None:
        print("  WARN: no TOC component in SOS container; skipping LOAD_TOC")
    else:
        try:
            submit_load_toc(c, drv, MP0_BASE_DW, result.ring, ctx, toc_comp.data)
        except RuntimeError as e:
            print(f"  LOAD_TOC FAILED: {e}")
            sys.exit(6)

    print("\n== Step 4b: PSP LOAD_IP_FW(IMU_I, IMU_D) ==")
    imu_blob = open(os.path.join(FIRMWARE_DIR, "gc_12_0_1_imu.bin"), "rb").read()
    iram, dram = _extract_imu(imu_blob)
    _load_one(c, drv, MP0_BASE_DW, result.ring, ctx, iram,
              GFX_FW_TYPE_IMU_I, "IMU_I", strict=True)
    _load_one(c, drv, MP0_BASE_DW, result.ring, ctx, dram,
              GFX_FW_TYPE_IMU_D, "IMU_D", strict=True)

    print("\n== Step 5: GFX_CMD_ID_AUTOLOAD_RLC ==")
    try:
        resp = submit_autoload_rlc(c, drv, MP0_BASE_DW, result.ring, ctx)
    except TimeoutError as e:
        print(f"  TIMEOUT: {e}")
        sys.exit(2)
    print(f"  AUTOLOAD_RLC status = 0x{resp['status']:08x}")
    if resp["status"] != 0:
        print(f"  PSP rejected AUTOLOAD_RLC; raw_resp[0..32] = {resp['raw_resp'][:32].hex()}")
        sys.exit(3)

    print("\n== Step 6: post-autoload state ==")
    GC_B1 = 0xA000
    def gc_rd(off):
        return c.mmio_read32(5, (GC_B1 + off) * 4)
    print(f"  GFX_IMU_CORE_CTRL       = 0x{gc_rd(0x40b6):08x}")
    print(f"  GFX_IMU_GFX_RESET_CTRL  = 0x{gc_rd(0x40bc):08x}")
    print(f"  RLC_CNTL                = 0x{gc_rd(0x4c00):08x}")
    print(f"  RLC_GPM_THREAD_ENABLE   = 0x{gc_rd(0x4c45):08x}")
    print(f"  GRBM_STATUS (idx0)      = 0x{c.mmio_read32(5, (0x1260 + 0x0da4) * 4):08x}")

    print("\n== Step 7: EnableAllSmuFeatures(FEATURE_PWR_GFX) ==")
    try:
        resp, arg_out = smu_send(c, PPSMC_MSG_EnableAllSmuFeatures,
                                 FEATURE_PWR_GFX, timeout_ms=5000)
    except TimeoutError as e:
        print(f"  TIMEOUT: {e}")
        sys.exit(4)
    if resp != PPSMC_Result_OK:
        print(f"  FAILED: resp=0x{resp:x} arg_out=0x{arg_out:x}")
        sys.exit(5)
    print(f"  SUCCESS: resp=0x{resp:x} arg_out=0x{arg_out:x}")
    _, lo = smu_send(c, PPSMC_MSG_GetRunningSmuFeaturesLo, 0)
    _, hi = smu_send(c, PPSMC_MSG_GetRunningSmuFeaturesHi, 0)
    print(f"  RunningFeaturesLow  = 0x{lo:08x}")
    print(f"  RunningFeaturesHigh = 0x{hi:08x}")


if __name__ == "__main__":
    main()
