"""Minimal test: just autoload with the fixed TOC payload copy.

Hypothesis: previous RLC stall after bit 5 of RLC_RLCS_BOOTLOAD_STATUS
was caused by placing the common_firmware_header at the start of the
RLC_TOC slot. At runtime RLC reads its TOC at the slot offset and was
seeing 256 bytes of header garbage followed by TOC entries shifted
256 bytes, so it couldn't find any of the CP/MES/SDMA firmware
pieces in the autoload buffer.

Linux `gfx_v12_0_rlc_backdoor_autoload_copy_toc_ucode` copies only
the ucode payload (from `ucode_array_offset_bytes` onward) into the
RLC_TOC slot. The 2048-byte gc_12_0_1_toc.bin is 256 bytes header +
1792 bytes of TOC entries → payload exactly fits the 0x700 TOC slot.

No SMU feature enables here — in Linux, BOOTLOAD_COMPLETE fires
purely from PSP-driven autoload BEFORE hw_init runs smu prep. If
this works, BOOTLOAD_STATUS bits 6-30 and 31 should light up on
their own once IMU finishes streaming RLC_G and RLC starts loading
the rest of the firmware from the (now correctly laid out) VRAM
buffer.
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
    MP0_BASE_DW,
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


def _try(c, msg, arg, name, *, timeout=3000):
    try:
        r, a = smu_send(c, msg, arg, timeout_ms=timeout)
        print(f"  {name:52s} resp=0x{r:x} arg_out=0x{a:x}")
        return r, a
    except TimeoutError:
        print(f"  {name:52s} TIMEOUT")
        return None, None


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

    print("\n== 1: smu_bring_up ==")
    result = smu_bring_up(c, drv, firmware_dir=FIRMWARE_DIR, enable_domain=None)
    fb_base = (c.mmio_read32(5, (0x1A000 + 0x0554) * 4) & 0xFFFFFF) << 24
    tool_mc = fb_base + TOOL_VRAM_OFF
    pool_mc = fb_base + POOL_VRAM_OFF

    ctx = alloc_cmd_ctx(drv)

    print("\n== 2: LOAD_TOC + LOAD_IP_FW(IMU) + autoload buffer ==")
    sos_blob = open(os.path.join(FIRMWARE_DIR, "psp_14_0_3_sos.bin"), "rb").read()
    toc_comp = next(x for x in parse_psp_firmware(sos_blob) if x.name == "TOC")
    submit_load_toc(c, drv, MP0_BASE_DW, result.ring, ctx, toc_comp.data)
    imu_blob = open(os.path.join(FIRMWARE_DIR, "gc_12_0_1_imu.bin"), "rb").read()
    iram, dram = _extract_imu(imu_blob)
    _load_one(c, drv, MP0_BASE_DW, result.ring, ctx, iram, GFX_FW_TYPE_IMU_I, "IMU_I", strict=True)
    _load_one(c, drv, MP0_BASE_DW, result.ring, ctx, dram, GFX_FW_TYPE_IMU_D, "IMU_D", strict=True)
    with open(os.path.join(FIRMWARE_DIR, "gc_12_0_1_toc.bin"), "rb") as f:
        toc_blob = f.read()
    layout = plan_autoload(toc_blob)
    build_autoload_buffer(c, FIRMWARE_DIR, layout, toc_blob)

    print("\n== 3: AUTOLOAD_RLC ==")
    resp = submit_autoload_rlc(c, drv, MP0_BASE_DW, result.ring, ctx)
    print(f"  AUTOLOAD_RLC status = 0x{resp['status']:08x}")

    # Snapshot GFX state right after AUTOLOAD_RLC, before any SMU chatter.
    GC_B1 = 0xA000
    def gc_rd(o): return c.mmio_read32(5, (GC_B1 + o) * 4)
    print("\n  GFX snapshot post-AUTOLOAD (pre-SMU-prep):")
    print(f"    IMU_CORE_CTRL = 0x{gc_rd(0x40b6):x}  "
          f"IMU_GFX_RESET_CTRL = 0x{gc_rd(0x40bc):08x}  "
          f"BOOTLOAD = 0x{gc_rd(0x4e7c):08x}")

    print("\n== 4: SMU post-autoload prep (tools/pool/pptable/btc/pcie, NO feature enables) ==")
    _try(c, PPSMC_MSG_SetToolsDramAddrHigh, (tool_mc >> 32) & 0xFFFFFFFF, "SetToolsDramAddrHigh")
    _try(c, PPSMC_MSG_SetToolsDramAddrLow, tool_mc & 0xFFFFFFFF, "SetToolsDramAddrLow")
    _try(c, PPSMC_MSG_SetSystemVirtualDramAddrHigh, (pool_mc >> 32) & 0xFFFFFFFF, "SetSystemVirtualDramAddrHigh")
    _try(c, PPSMC_MSG_SetSystemVirtualDramAddrLow, pool_mc & 0xFFFFFFFF, "SetSystemVirtualDramAddrLow")
    _try(c, PPSMC_MSG_UseDefaultPPTable, 0, "UseDefaultPPTable")
    _try(c, PPSMC_MSG_TransferTableSmu2Dram, TABLE_COMBO_PPTABLE,
         "TransferTableSmu2Dram(COMBO_PPTABLE=1)", timeout=5000)
    _try(c, PPSMC_MSG_RunDcBtc, 0, "RunDcBtc", timeout=5000)
    for level in range(3):
        arg = (level << 16) | (2 << 8) | 3
        _try(c, PPSMC_MSG_OverridePcieParameters, arg,
             f"OverridePcieParameters(level={level})", timeout=5000)

    print("\n  GFX snapshot post-SMU-prep (pre-poll):")
    print(f"    IMU_CORE_CTRL = 0x{gc_rd(0x40b6):x}  "
          f"IMU_GFX_RESET_CTRL = 0x{gc_rd(0x40bc):08x}  "
          f"BOOTLOAD = 0x{gc_rd(0x4e7c):08x}")

    print("\n== 5: poll BOOTLOAD_COMPLETE (30s, log every transition) ==")
    deadline = time.time() + 30
    last = None
    start = time.time()
    while time.time() < deadline:
        bl = gc_rd(0x4e7c); rst = gc_rd(0x40bc); core = gc_rd(0x40b6)
        stat = gc_rd(0x4b20)   # RLC_STAT
        gpm = gc_rd(0x4b48)    # RLC_GPM_STAT
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

    print("\n== Final RLC debug registers ==")
    for name, off in [
        ("RLC_CNTL",              0x4b00),
        ("RLC_STAT",              0x4b20),
        ("RLC_GPM_STAT",          0x4b48),
        ("RLC_GPM_THREAD_ENABLE", 0x4b3e),
        ("RLC_GPM_THREAD_RESET",  0x4b3d),
        ("RLC_PG_CNTL",           0x4b60),
        ("RLC_SAFE_MODE",         0x4b40),
        ("RLC_RLCS_BOOTLOAD_STATUS", 0x4e7c),
        ("RLC_RLCS_EXCEPTION_REG_1", 0x4e80),
        ("RLC_RLCS_EXCEPTION_REG_2", 0x4e81),
        ("RLC_RLCS_EXCEPTION_REG_3", 0x4e82),
        ("RLC_RLCS_EXCEPTION_REG_4", 0x4e83),
        ("CP_STAT",               0x7e68),
        ("GRBM_STATUS",           0x8010),
    ]:
        v = gc_rd(off)
        print(f"  {name:30s} = 0x{v:08x}")


if __name__ == "__main__":
    main()
