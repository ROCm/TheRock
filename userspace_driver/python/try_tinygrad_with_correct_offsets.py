"""Tinygrad order + CORRECT RLC register offsets.

Prior test (try_tinygrad_plus_bootloader.py, 2026-04-22) showed:
  After AUTOLOAD_RLC: RESET=0x7F, BOOTLOAD=0x3F, RLC_CNTL=0x0.
  But memory file says at e23bb507 we saw RLC_CNTL=0x1 (enabled).

Root cause: RLC_CNTL is at 0x4c00 (BASE_IDX=1), not 0x4b00 as our
scripts had. Per gc_12_0_0_offset.h:

  regRLC_CNTL                     0x4c00  BASE_IDX=1 (=0xA000)
  regRLC_STAT                     0x4c04  BASE_IDX=1
  regRLC_GPM_THREAD_RESET         0x4c28  BASE_IDX=1
  regRLC_PG_CNTL                  0x4c43  BASE_IDX=1
  regRLC_GPM_THREAD_ENABLE        0x4c45  BASE_IDX=1
  regRLC_GPM_STAT                 0x4e6c  BASE_IDX=1
  regRLC_RLCS_BOOTLOAD_STATUS     0x4e7c  BASE_IDX=1
  regGRBM_STATUS                  0x0da4  BASE_IDX=0 (=0x1260)
  regCP_STAT                      0x0f40  BASE_IDX=0

All the IMU regs we've been reading (CORE_CTRL=0x40b6,
GFX_RESET_CTRL=0x40bc) are correct and in BASE_IDX=1.

This script reruns tinygrad's order and reads the ACTUAL registers
so we can see real RLC state.
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
    MP0_BASE_DW,
    PPSMC_MSG_EnableAllSmuFeatures,
    smu_bring_up,
    smu_send,
)

FIRMWARE_DIR = os.path.expanduser("~/firmware/linux-firmware/amdgpu")
GC_B0 = 0x1260   # BASE_IDX=0 (GFXHUB side)
GC_B1 = 0xA000   # BASE_IDX=1 (main GFX + IMU + RLC)

# Register offsets (from gc_12_0_0_offset.h)
REGS = {
    # GFXHUB side (BASE_IDX=0):
    "GRBM_STATUS":               (GC_B0, 0x0da4),
    "CP_STAT":                   (GC_B0, 0x0f40),
    # Main GFX side (BASE_IDX=1):
    "GFX_IMU_CORE_CTRL":         (GC_B1, 0x40b6),
    "GFX_IMU_GFX_RESET_CTRL":    (GC_B1, 0x40bc),
    "GFX_IMU_RLC_BOOTLOADER_ADDR_HI": (GC_B1, 0x5f81),
    "GFX_IMU_RLC_BOOTLOADER_ADDR_LO": (GC_B1, 0x5f82),
    "GFX_IMU_RLC_BOOTLOADER_SIZE":    (GC_B1, 0x5f83),
    "RLC_CNTL":                  (GC_B1, 0x4c00),
    "RLC_STAT":                  (GC_B1, 0x4c04),
    "RLC_GPM_THREAD_RESET":      (GC_B1, 0x4c28),
    "RLC_PG_CNTL":               (GC_B1, 0x4c43),
    "RLC_GPM_THREAD_ENABLE":     (GC_B1, 0x4c45),
    "RLC_GPM_STAT":              (GC_B1, 0x4e6c),
    "RLC_RLCS_BOOTLOAD_STATUS":  (GC_B1, 0x4e7c),
}


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

    def reg_rd(base, off): return c.mmio_read32(5, (base + off) * 4)

    def snapshot(label):
        print(f"\n  [{label}]")
        for name, (base, off) in REGS.items():
            v = reg_rd(base, off)
            print(f"    {name:32s} = 0x{v:08x}")

    print("\n== 1: smu_bring_up(enable_domain=None) ==")
    result = smu_bring_up(c, drv, firmware_dir=FIRMWARE_DIR, enable_domain=None)
    snapshot("after smu_bring_up")

    ctx = alloc_cmd_ctx(drv)

    print("\n== 2: LOAD_TOC ==")
    sos_blob = open(os.path.join(FIRMWARE_DIR, "psp_14_0_3_sos.bin"), "rb").read()
    toc_comp = next(x for x in parse_psp_firmware(sos_blob) if x.name == "TOC")
    submit_load_toc(c, drv, MP0_BASE_DW, result.ring, ctx, toc_comp.data)

    print("\n== 3: LOAD_IP_FW(IMU_I, IMU_D) ==")
    imu_blob = open(os.path.join(FIRMWARE_DIR, "gc_12_0_1_imu.bin"), "rb").read()
    iram, dram = _extract_imu(imu_blob)
    _load_one(c, drv, MP0_BASE_DW, result.ring, ctx, iram, GFX_FW_TYPE_IMU_I, "IMU_I", strict=True)
    _load_one(c, drv, MP0_BASE_DW, result.ring, ctx, dram, GFX_FW_TYPE_IMU_D, "IMU_D", strict=True)

    print("\n== 4: build autoload buffer ==")
    with open(os.path.join(FIRMWARE_DIR, "gc_12_0_1_toc.bin"), "rb") as f:
        toc_blob = f.read()
    layout = plan_autoload(toc_blob)
    build_autoload_buffer(c, FIRMWARE_DIR, layout, toc_blob)

    print("\n== 5: AUTOLOAD_RLC ==")
    resp = submit_autoload_rlc(c, drv, MP0_BASE_DW, result.ring, ctx)
    print(f"  AUTOLOAD_RLC status = 0x{resp['status']:08x}")

    snapshot("just after AUTOLOAD_RLC")

    print("\n== 6: EnableAllSmuFeatures(0) [tinygrad's arg] ==")
    try:
        r, a = smu_send(c, PPSMC_MSG_EnableAllSmuFeatures, 0, timeout_ms=8000)
        print(f"  EnableAllSmuFeatures(0) resp=0x{r:x} arg_out=0x{a:x}")
    except TimeoutError:
        print("  EnableAllSmuFeatures(0) TIMEOUT (expected — GFX still gets powered)")

    snapshot("just after EnableAllSmuFeatures(0)")

    print("\n== 7: poll BOOTLOAD_COMPLETE (30s) ==")
    deadline = time.time() + 30
    last = None
    start = time.time()
    while time.time() < deadline:
        bl = reg_rd(GC_B1, 0x4e7c)     # RLC_RLCS_BOOTLOAD_STATUS
        rst = reg_rd(GC_B1, 0x40bc)    # GFX_IMU_GFX_RESET_CTRL
        core = reg_rd(GC_B1, 0x40b6)   # GFX_IMU_CORE_CTRL
        cntl = reg_rd(GC_B1, 0x4c00)   # RLC_CNTL
        stat = reg_rd(GC_B1, 0x4c04)   # RLC_STAT
        gpm = reg_rd(GC_B1, 0x4e6c)    # RLC_GPM_STAT
        snap = (bl, rst, core, cntl, stat, gpm)
        if snap != last:
            t = time.time() - start
            print(f"  t={t:6.3f}s CORE=0x{core:x} RESET=0x{rst:08x} "
                  f"BOOTLOAD=0x{bl:08x} RLC_CNTL=0x{cntl:x} "
                  f"RLC_STAT=0x{stat:08x} GPM_STAT=0x{gpm:08x}")
            last = snap
        if bl & 0x80000000:
            print("  BOOTLOAD_COMPLETE ✓")
            break
        time.sleep(0.05)

    snapshot("final state")


if __name__ == "__main__":
    main()
