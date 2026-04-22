"""PWR_SOC warm-up BEFORE AUTOLOAD_RLC.

Observation (try_autoload_only_toc_fix.py, 2026-04-21):
  Post-AUTOLOAD_RLC: CORE=0x8 IMU running, RESET=0x30 GFX in reset,
  BOOTLOAD=0x00 — and the state doesn't change no matter how long we
  wait or what SMU chatter we add (short of feature enables). IMU can
  clear CRESET on itself, but GFX blocks stay gated; RLC can't boot
  because it lives inside a gated GFX block.

The previous "0x3F with RESET=0x7F" readings were from runs that
did EnableAll(PWR_SOC=3) somewhere in the flow — that warm-up is
what releases GFX from reset.

New order:
  1. smu_bring_up.
  2. LOAD_TOC + LOAD_IP_FW(IMU_I/D).
  3. SetToolsDramAddr + SystemVirtualDramAddr.
  4. UseDefaultPPTable + TransferTableSmu2Dram(COMBO_PPTABLE).
  5. RunDcBtc.
  6. OverridePcieParameters × 3.
  7. EnableAll(PWR_SOC=3)              ← releases GFX from reset.
  8. Build autoload buffer.
  9. AUTOLOAD_RLC.
  10. Poll BOOTLOAD_COMPLETE (30s).
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
    FEATURE_PWR_SOC,
    MP0_BASE_DW,
    PPSMC_MSG_DisallowGfxOff,
    PPSMC_MSG_EnableAllSmuFeatures,
    PPSMC_MSG_SetSystemVirtualDramAddrHigh,
    PPSMC_MSG_SetSystemVirtualDramAddrLow,
    PPSMC_MSG_SetToolsDramAddrHigh,
    PPSMC_MSG_SetToolsDramAddrLow,
    PPSMC_MSG_TransferTableSmu2Dram,
    PPSMC_MSG_UseDefaultPPTable,
    smu_bring_up,
    smu_send,
)

FIRMWARE_DIR = os.path.expanduser("~/firmware/linux-firmware/amdgpu")
PPSMC_MSG_RunDcBtc               = 0x36
PPSMC_MSG_OverridePcieParameters = 0x20
TABLE_COMBO_PPTABLE = 1
TOOL_VRAM_OFF = 0x1810000
POOL_VRAM_OFF = 0x1900000


class _DriverShim:
    def __init__(self, client): self.client = client
    def alloc_dma(self, size):
        dma = self.client.alloc_dma(size)
        bus = dma.segments[0][0] if dma.segments else 0
        return (dma.cpu_addr, bus, dma.buffer_id)
    def free_dma(self, h): self.client.free_dma(h)


def _try(c, msg, arg, name, *, timeout=3000):
    try:
        r, a = smu_send(c, msg, arg, timeout_ms=timeout)
        print(f"  {name:52s} resp=0x{r:x} arg_out=0x{a:x}")
        return r, a
    except TimeoutError:
        print(f"  {name:52s} TIMEOUT")
        return None, None


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    c = IOKitClient(); c.open()
    info = c.get_info()
    print(f"device=0x{info.device_id:04x} rev=0x{info.revision_id:02x}")
    if c.mmio_read32(5, (0x16000 + 0x40 + 81) * 4) != 0:
        print("SOS already alive — replug first.")
        sys.exit(0)
    drv = _DriverShim(c)

    # Match the working recipe from commit e23bb507 (the one that
    # reached BOOTLOAD=0x3F / RESET=0x7F). Key differences vs our
    # recent attempts:
    #   - PWR_SOC warm-up is done INSIDE smu_bring_up, not separately
    #     later. So it happens right after SetDriverDramAddr, before
    #     LOAD_TOC / LOAD_IP_FW(IMU).
    #   - DisallowGfxOff is sent BEFORE building the autoload buffer.
    #   - build_autoload_buffer runs BEFORE LOAD_TOC.
    #   - None of the pptable/tools/pool/btc/pcie chatter (that's for
    #     SMU feature enables AFTER autoload, not prerequisites).
    print("\n== 1: smu_bring_up(FEATURE_PWR_SOC) ==")
    result = smu_bring_up(c, drv, firmware_dir=FIRMWARE_DIR,
                          enable_domain=FEATURE_PWR_SOC)

    ctx = alloc_cmd_ctx(drv)

    GC_B1 = 0xA000
    def gc_rd(o): return c.mmio_read32(5, (GC_B1 + o) * 4)
    def snapshot(label):
        print(f"\n  [{label}] CORE=0x{gc_rd(0x40b6):x} "
              f"RESET=0x{gc_rd(0x40bc):08x} "
              f"BOOTLOAD=0x{gc_rd(0x4e7c):08x}")

    snapshot("after smu_bring_up(PWR_SOC)")

    print("\n== 2: DisallowGfxOff ==")
    _try(c, PPSMC_MSG_DisallowGfxOff, 0, "DisallowGfxOff", timeout=1500)
    snapshot("after DisallowGfxOff")

    print("\n== 3: build autoload buffer ==")
    with open(os.path.join(FIRMWARE_DIR, "gc_12_0_1_toc.bin"), "rb") as f:
        toc_blob = f.read()
    layout = plan_autoload(toc_blob)
    build_autoload_buffer(c, FIRMWARE_DIR, layout, toc_blob)
    snapshot("after build_autoload_buffer")

    print("\n== 4: LOAD_TOC + LOAD_IP_FW(IMU_I/D) ==")
    sos_blob = open(os.path.join(FIRMWARE_DIR, "psp_14_0_3_sos.bin"), "rb").read()
    toc_comp = next(x for x in parse_psp_firmware(sos_blob) if x.name == "TOC")
    submit_load_toc(c, drv, MP0_BASE_DW, result.ring, ctx, toc_comp.data)
    imu_blob = open(os.path.join(FIRMWARE_DIR, "gc_12_0_1_imu.bin"), "rb").read()
    iram, dram = _extract_imu(imu_blob)
    _load_one(c, drv, MP0_BASE_DW, result.ring, ctx, iram, GFX_FW_TYPE_IMU_I, "IMU_I", strict=True)
    _load_one(c, drv, MP0_BASE_DW, result.ring, ctx, dram, GFX_FW_TYPE_IMU_D, "IMU_D", strict=True)
    snapshot("after LOAD_TOC + IMU")

    print("\n== 5: AUTOLOAD_RLC ==")
    resp = submit_autoload_rlc(c, drv, MP0_BASE_DW, result.ring, ctx)
    print(f"  AUTOLOAD_RLC status = 0x{resp['status']:08x}")

    snapshot("just after AUTOLOAD_RLC")

    print("\n== 6: poll BOOTLOAD_COMPLETE (30s) ==")
    deadline = time.time() + 30
    last = None
    start = time.time()
    while time.time() < deadline:
        bl = gc_rd(0x4e7c); rst = gc_rd(0x40bc); core = gc_rd(0x40b6)
        stat = gc_rd(0x4b20); gpm = gc_rd(0x4b48)
        snap = (bl, rst, core, stat, gpm)
        if snap != last:
            t = time.time() - start
            print(f"  t={t:6.3f}s CORE=0x{core:x} RESET=0x{rst:08x} "
                  f"BOOTLOAD=0x{bl:08x} RLC_STAT=0x{stat:x} GPM_STAT=0x{gpm:08x}")
            last = snap
        if bl & 0x80000000:
            print("  BOOTLOAD_COMPLETE ✓")
            break
        time.sleep(0.05)

    print("\n== Final RLC debug ==")
    for name, off in [
        ("RLC_CNTL",                     0x4b00),
        ("RLC_STAT",                     0x4b20),
        ("RLC_GPM_STAT",                 0x4b48),
        ("RLC_GPM_THREAD_ENABLE",        0x4b3e),
        ("RLC_GPM_THREAD_RESET",         0x4b3d),
        ("RLC_RLCS_BOOTLOAD_STATUS",     0x4e7c),
        ("RLC_RLCS_EXCEPTION_REG_1",     0x4e80),
        ("RLC_RLCS_EXCEPTION_REG_2",     0x4e81),
        ("RLC_RLCS_EXCEPTION_REG_3",     0x4e82),
        ("RLC_RLCS_EXCEPTION_REG_4",     0x4e83),
        ("CP_STAT",                      0x7e68),
        ("GRBM_STATUS",                  0x8010),
    ]:
        print(f"  {name:30s} = 0x{gc_rd(off):08x}")


if __name__ == "__main__":
    main()
