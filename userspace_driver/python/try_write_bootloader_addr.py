"""Explicitly write GFX_IMU_RLC_BOOTLOADER_ADDR before AUTOLOAD_RLC.

Current state: even at commit e23bb507 (the "working" commit per its
message), AUTOLOAD_RLC leaves the GPU at CORE=0x8, RESET=0x30,
BOOTLOAD=0. IMU is unhalted but does nothing — no reset release, no
RLC load. Something the GPU/DEXT/Mac stack does differently now
compared to April 20 has regressed the PSP-driven autoload.

Hypothesis: PSP assumes a specific convention for where the autoload
buffer lives, but that convention isn't being honoured. Linux's
`amdgpu_gfx_rlc_init_microcode` explicitly writes
GFX_IMU_RLC_BOOTLOADER_ADDR_{LO,HI} and _SIZE before autoload. If we
also write them, IMU should find the RLC_G ucode at that address.

Experiment: after AUTOLOAD_RLC returns, check if GC writes stick.
Then write BOOTLOADER_ADDR from the host side. If the state changes
— even to BOOTLOAD != 0 — we've learned that the bootloader address
register is what gated RLC from loading.

Also try: write the registers BEFORE AUTOLOAD_RLC. We'll probe both.
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
    FEATURE_PWR_SOC,
    MP0_BASE_DW,
    PPSMC_MSG_DisallowGfxOff,
    smu_bring_up,
    smu_send,
)

FIRMWARE_DIR = os.path.expanduser("~/firmware/linux-firmware/amdgpu")
GC_B1 = 0xA000
regGFX_IMU_RLC_BOOTLOADER_ADDR_LO = 0x5f82
regGFX_IMU_RLC_BOOTLOADER_ADDR_HI = 0x5f81
regGFX_IMU_RLC_BOOTLOADER_SIZE    = 0x5f83
regGFX_IMU_CORE_CTRL              = 0x40b6
regGFX_IMU_GFX_RESET_CTRL         = 0x40bc
regRLC_RLCS_BOOTLOAD_STATUS       = 0x4e7c
regRLC_CNTL                       = 0x4b00


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

    def gc_rd(o): return c.mmio_read32(5, (GC_B1 + o) * 4)
    def gc_wr(o, v): c.mmio_write32(5, (GC_B1 + o) * 4, v & 0xFFFFFFFF)

    def snapshot(label):
        core = gc_rd(regGFX_IMU_CORE_CTRL)
        rst  = gc_rd(regGFX_IMU_GFX_RESET_CTRL)
        bl   = gc_rd(regRLC_RLCS_BOOTLOAD_STATUS)
        cntl = gc_rd(regRLC_CNTL)
        blo  = gc_rd(regGFX_IMU_RLC_BOOTLOADER_ADDR_LO)
        bhi  = gc_rd(regGFX_IMU_RLC_BOOTLOADER_ADDR_HI)
        bsz  = gc_rd(regGFX_IMU_RLC_BOOTLOADER_SIZE)
        print(f"  [{label}] CORE=0x{core:x} RESET=0x{rst:08x} "
              f"BOOTLOAD=0x{bl:08x} RLC_CNTL=0x{cntl:x} "
              f"BOOTLOADER=[hi=0x{bhi:x} lo=0x{blo:08x} sz=0x{bsz:x}]")

    print("\n== 1: smu_bring_up(FEATURE_PWR_SOC) ==")
    result = smu_bring_up(c, drv, firmware_dir=FIRMWARE_DIR,
                          enable_domain=FEATURE_PWR_SOC)

    snapshot("initial")

    smu_send(c, PPSMC_MSG_DisallowGfxOff, 0, timeout_ms=1500)
    snapshot("after DisallowGfxOff")

    print("\n== 2: build autoload buffer ==")
    with open(os.path.join(FIRMWARE_DIR, "gc_12_0_1_toc.bin"), "rb") as f:
        toc_blob = f.read()
    layout = plan_autoload(toc_blob)
    build_autoload_buffer(c, FIRMWARE_DIR, layout, toc_blob)

    snapshot("after buffer fill")

    # The buffer was written via BAR0 which windows VRAM starting at
    # MC address `fb_base + 0` (default vBIOS-POST). So RLC_G_UCODE is
    # at MC address fb_base + layout.rlc_g_offset (= fb_base + 0 in
    # our layout).
    fb_base = (c.mmio_read32(5, (0x1A000 + 0x0554) * 4) & 0xFFFFFF) << 24
    rlc_g_mc = fb_base + layout.rlc_g_offset
    print(f"\n  RLC_G_UCODE MC addr = 0x{rlc_g_mc:x} "
          f"(fb_base=0x{fb_base:x}, rlc_g_offset=0x{layout.rlc_g_offset:x})")
    print(f"  RLC_G_UCODE size    = 0x{layout.rlc_g_size:x}")

    print("\n== 3: try writing GFX_IMU_RLC_BOOTLOADER_ADDR before AUTOLOAD_RLC ==")
    # These writes might not stick if GC is gated, but we can try.
    gc_wr(regGFX_IMU_RLC_BOOTLOADER_ADDR_LO, rlc_g_mc & 0xFFFFFFFF)
    gc_wr(regGFX_IMU_RLC_BOOTLOADER_ADDR_HI, (rlc_g_mc >> 32) & 0xFFFFFFFF)
    gc_wr(regGFX_IMU_RLC_BOOTLOADER_SIZE, layout.rlc_g_size)
    snapshot("after writing BOOTLOADER_ADDR (pre-AUTOLOAD)")

    ctx = alloc_cmd_ctx(drv)

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

    print("\n== 6: try writing BOOTLOADER_ADDR AGAIN post-AUTOLOAD ==")
    # After AUTOLOAD_RLC IMU is unhalted; GC writes might now stick.
    gc_wr(regGFX_IMU_RLC_BOOTLOADER_ADDR_LO, rlc_g_mc & 0xFFFFFFFF)
    gc_wr(regGFX_IMU_RLC_BOOTLOADER_ADDR_HI, (rlc_g_mc >> 32) & 0xFFFFFFFF)
    gc_wr(regGFX_IMU_RLC_BOOTLOADER_SIZE, layout.rlc_g_size)
    snapshot("after post-AUTOLOAD BOOTLOADER_ADDR write")

    print("\n== 7: poll BOOTLOAD for 20s ==")
    deadline = time.time() + 20
    last = None
    start = time.time()
    while time.time() < deadline:
        core = gc_rd(regGFX_IMU_CORE_CTRL)
        rst  = gc_rd(regGFX_IMU_GFX_RESET_CTRL)
        bl   = gc_rd(regRLC_RLCS_BOOTLOAD_STATUS)
        cntl = gc_rd(regRLC_CNTL)
        snap = (core, rst, bl, cntl)
        if snap != last:
            t = time.time() - start
            print(f"  t={t:6.3f}s CORE=0x{core:x} RESET=0x{rst:08x} "
                  f"BOOTLOAD=0x{bl:08x} RLC_CNTL=0x{cntl:x}")
            last = snap
        if bl & 0x80000000:
            print("  BOOTLOAD_COMPLETE ✓")
            break
        time.sleep(0.05)


if __name__ == "__main__":
    main()
