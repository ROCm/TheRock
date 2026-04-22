"""Tinygrad order + explicit BOOTLOADER_ADDR write.

Breakthrough from the prior run (try_tinygrad_order.py): even though
`EnableAllSmuFeatures(0)` timed out (no ACK from SMU within 10s), it
had real effects:
  - RESET_CTRL: 0x30 → 0x7F (all 7 GFX blocks out of reset)
  - BOOTLOAD_STATUS: 0x00 → 0x3F (bits 0-5 set; RLC GPM IRAM loaded)

But bits 6-30 (and 31 BOOTLOAD_COMPLETE) never fire. Hypothesis:
RLC needs the `GFX_IMU_RLC_BOOTLOADER_ADDR_{HI,LO,SIZE}` registers
to point at the autoload buffer so it can find the rest of the
firmware (CP PFP/ME/MEC, MES, SDMA) past its own GPM IRAM. Linux
writes them explicitly from host (gfx_v12_0.c:1322-1329) with the
FB-OFFSET form: `autoload_gpu_addr + rlc_g_offset - vram_start`.

For our autoload buffer at VRAM offset 0, rlc_g_offset = 0, so:
  BOOTLOADER_ADDR_HI = 0
  BOOTLOADER_ADDR_LO = 0
  BOOTLOADER_SIZE    = 0x6000 (RLC_G_UCODE TOC slot size)

Experimental order:
  1. smu_bring_up(enable_domain=None)
  2. EnableAllSmuFeatures(0)       -- powers up GFX block. Hangs SMU
                                      but GFX gets powered. Expect
                                      RESET to reach 0x7F.
  3. Wait briefly to let the state settle.
  4. Write BOOTLOADER_ADDR_HI/LO/SIZE now that GFX is powered.
     Verify by read-back.
  5. Build autoload buffer.
  6. LOAD_TOC (PSP, via ring).
  7. LOAD_IP_FW(IMU_I, IMU_D).
  8. AUTOLOAD_RLC.
  9. Poll BOOTLOAD_COMPLETE for 30 s.
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
GC_B1 = 0xA000

regGFX_IMU_CORE_CTRL              = 0x40b6
regGFX_IMU_GFX_RESET_CTRL         = 0x40bc
regGFX_IMU_RLC_BOOTLOADER_ADDR_HI = 0x5f81
regGFX_IMU_RLC_BOOTLOADER_ADDR_LO = 0x5f82
regGFX_IMU_RLC_BOOTLOADER_SIZE    = 0x5f83
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
        print(f"  [{label}]")
        print(f"    CORE=0x{core:x} RESET=0x{rst:08x} BOOTLOAD=0x{bl:08x} RLC_CNTL=0x{cntl:x}")
        print(f"    BOOTLOADER[hi=0x{bhi:08x} lo=0x{blo:08x} sz=0x{bsz:08x}]")

    print("\n== 1: smu_bring_up(enable_domain=None) ==")
    result = smu_bring_up(c, drv, firmware_dir=FIRMWARE_DIR, enable_domain=None)
    snapshot("after smu_bring_up")

    ctx = alloc_cmd_ctx(drv)

    print("\n== 2: EnableAllSmuFeatures(0) — power up GFX (may hang SMU) ==")
    try:
        r, a = smu_send(c, PPSMC_MSG_EnableAllSmuFeatures, 0, timeout_ms=8000)
        print(f"  EnableAllSmuFeatures(0) resp=0x{r:x} arg_out=0x{a:x}")
    except TimeoutError:
        print("  EnableAllSmuFeatures(0) TIMEOUT (expected — GFX still gets powered)")

    print("\n== 3: settle 500 ms ==")
    time.sleep(0.5)
    snapshot("after EnableAll(0) + settle")

    print("\n== 4: build autoload buffer (need buffer before BOOTLOADER_ADDR write) ==")
    with open(os.path.join(FIRMWARE_DIR, "gc_12_0_1_toc.bin"), "rb") as f:
        toc_blob = f.read()
    layout = plan_autoload(toc_blob)
    build_autoload_buffer(c, FIRMWARE_DIR, layout, toc_blob)
    snapshot("after build_autoload_buffer")

    # Write BOOTLOADER_ADDR with FB-offset form (Linux gfx_v12_0.c:1322-1329).
    # Autoload buffer is at VRAM offset 0 (fb_base + 0), RLC_G_UCODE at
    # rlc_g_offset within buffer (also 0 in our layout). Value to write is
    # (buffer_addr + rlc_g_offset - vram_start) = 0 + 0 - 0 = 0 (HI, LO).
    rlc_g_fb_offset = layout.rlc_g_offset  # absolute offset within VRAM
    print(f"\n== 5: write BOOTLOADER_ADDR (FB-offset form, rlc_g_fb_offset=0x{rlc_g_fb_offset:x}) ==")
    gc_wr(regGFX_IMU_RLC_BOOTLOADER_ADDR_HI, (rlc_g_fb_offset >> 32) & 0xFFFFFFFF)
    gc_wr(regGFX_IMU_RLC_BOOTLOADER_ADDR_LO, rlc_g_fb_offset & 0xFFFFFFFF)
    gc_wr(regGFX_IMU_RLC_BOOTLOADER_SIZE, layout.rlc_g_size)
    snapshot("after writing BOOTLOADER_ADDR")

    print("\n== 6: LOAD_TOC ==")
    sos_blob = open(os.path.join(FIRMWARE_DIR, "psp_14_0_3_sos.bin"), "rb").read()
    toc_comp = next(x for x in parse_psp_firmware(sos_blob) if x.name == "TOC")
    submit_load_toc(c, drv, MP0_BASE_DW, result.ring, ctx, toc_comp.data)

    print("\n== 7: LOAD_IP_FW(IMU_I, IMU_D) ==")
    imu_blob = open(os.path.join(FIRMWARE_DIR, "gc_12_0_1_imu.bin"), "rb").read()
    iram, dram = _extract_imu(imu_blob)
    _load_one(c, drv, MP0_BASE_DW, result.ring, ctx, iram, GFX_FW_TYPE_IMU_I, "IMU_I", strict=True)
    _load_one(c, drv, MP0_BASE_DW, result.ring, ctx, dram, GFX_FW_TYPE_IMU_D, "IMU_D", strict=True)
    snapshot("after LOAD_TOC + IMU")

    print("\n== 8: AUTOLOAD_RLC ==")
    resp = submit_autoload_rlc(c, drv, MP0_BASE_DW, result.ring, ctx)
    print(f"  AUTOLOAD_RLC status = 0x{resp['status']:08x}")
    snapshot("just after AUTOLOAD_RLC")

    print("\n== 9: poll BOOTLOAD_COMPLETE (30s) ==")
    deadline = time.time() + 30
    last = None
    start = time.time()
    while time.time() < deadline:
        core = gc_rd(regGFX_IMU_CORE_CTRL)
        rst  = gc_rd(regGFX_IMU_GFX_RESET_CTRL)
        bl   = gc_rd(regRLC_RLCS_BOOTLOAD_STATUS)
        cntl = gc_rd(regRLC_CNTL)
        blo  = gc_rd(regGFX_IMU_RLC_BOOTLOADER_ADDR_LO)
        snap = (core, rst, bl, cntl, blo)
        if snap != last:
            t = time.time() - start
            print(f"  t={t:6.3f}s CORE=0x{core:x} RESET=0x{rst:08x} "
                  f"BOOTLOAD=0x{bl:08x} RLC_CNTL=0x{cntl:x} BOOT_LO=0x{blo:08x}")
            last = snap
        if bl & 0x80000000:
            print("  BOOTLOAD_COMPLETE ✓")
            break
        time.sleep(0.05)


if __name__ == "__main__":
    main()
