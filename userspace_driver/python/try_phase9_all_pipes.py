"""Phase 9c (attempt 3): enable ALL MEC pipes (0-3), not just pipe 0.

Prior runs enabled only pipe 0, matching tinygrad's _enable_mec. But
Linux's `gfx_v12_0_cp_compute_enable` enables pipe0/1/2/3_ACTIVE
together (pipes 1-3 also need to be live for MEC's internal queue
management — maybe on gfx12, MEC uses multiple pipes internally even
when the driver only wants to use pipe 0).

Starting from PRE CP_MEC_RS64_CNTL = 0x40030000:
  - Clear MEC_HALT (bit 30).
  - Clear PIPE0_RESET (bit 16) and PIPE1_RESET (bit 17).
  - Set PIPE0_ACTIVE (bit 26), PIPE1_ACTIVE (27), PIPE2_ACTIVE (28),
    PIPE3_ACTIVE (29).
  - Clear MEC_INVALIDATE_ICACHE (bit 4) per Linux.
Result: 0x3c000000.
"""
from __future__ import annotations

import ctypes
import logging
import os
import struct
import sys
import time

from amd_gpu_driver.backends.macos.gfx_bringup import gfx_bring_up
from amd_gpu_driver.backends.macos.iokit_client import IOKitClient

FIRMWARE_DIR = os.path.expanduser("~/firmware/linux-firmware/amdgpu")
_MMIO_BAR = 5
GC_B1 = 0xA000

regGRBM_GFX_CNTL                   = 0x0900
regCP_MEC_RS64_PRGRM_CNTR_START    = 0x2900
regCP_MEC_RS64_PRGRM_CNTR_START_HI = 0x2938
regCP_MEC_RS64_CNTL                = 0x2904
regCP_MEC_RS64_INSTR_PNTR          = 0x2908
regCP_HQD_ACTIVE                   = 0x1fab
regCP_HQD_PQ_BASE                  = 0x1fb1


class _DriverShim:
    def __init__(self, client): self.client = client
    def alloc_dma(self, size):
        dma = self.client.alloc_dma(size)
        bus = dma.segments[0][0] if dma.segments else 0
        return (dma.cpu_addr, bus, dma.buffer_id)
    def free_dma(self, h): self.client.free_dma(h)


def _parse_rs64(blob):
    _u_feat, u_sz, u_off, d_sz, d_off, u_start_lo, u_start_hi = \
        struct.unpack_from("<IIIIIII", blob, 32)
    return (u_start_hi << 32) | u_start_lo


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    c = IOKitClient(); c.open()
    info = c.get_info()
    print(f"device=0x{info.device_id:04x} rev=0x{info.revision_id:02x}")
    drv = _DriverShim(c)

    def gc1_rd(o): return c.mmio_read32(_MMIO_BAR, (GC_B1 + o) * 4)
    def gc1_wr(o, v): c.mmio_write32(_MMIO_BAR, (GC_B1 + o) * 4, v & 0xFFFFFFFF)

    r = gfx_bring_up(c, drv, firmware_dir=FIRMWARE_DIR)
    if not (r.bootload_status & 0x80000000):
        print("BOOTLOAD_COMPLETE not set — aborting.")
        sys.exit(1)

    # Parse MEC firmware for PRGRM_CNTR.
    mec_blob = open(os.path.join(FIRMWARE_DIR, "gc_12_0_1_mec.bin"), "rb").read()
    us = _parse_rs64(mec_blob)

    print(f"\n== _config_mec for all 4 pipes ==")
    # Write PRGRM_CNTR_START for each pipe.
    for pipe in range(4):
        gc1_wr(regGRBM_GFX_CNTL, (1 << 2) | pipe)  # me=1, pipe=N
        gc1_wr(regCP_MEC_RS64_PRGRM_CNTR_START, (us >> 2) & 0xFFFFFFFF)
        gc1_wr(regCP_MEC_RS64_PRGRM_CNTR_START_HI, (us >> 34) & 0xFFFFFFFF)
    gc1_wr(regGRBM_GFX_CNTL, 0)

    # Pulse all pipe resets (RMW: set reset bits, then clear).
    cntl = gc1_rd(regCP_MEC_RS64_CNTL)
    gc1_wr(regCP_MEC_RS64_CNTL, cntl | 0x000F0010)   # pipe0..3_reset=1, invalidate_icache=1
    cntl = gc1_rd(regCP_MEC_RS64_CNTL)
    gc1_wr(regCP_MEC_RS64_CNTL, cntl & ~0x000F0010)  # clear those bits
    time.sleep(0.01)

    print(f"\n== _enable_mec: activate pipes 0-3, clear halt (Linux pattern) ==")
    cntl = gc1_rd(regCP_MEC_RS64_CNTL)
    # Build new value: clear halt (bit 30), clear icache invalidate (bit 4),
    # clear all pipe resets (bits 16-19), set all pipe actives (bits 26-29).
    new_cntl = (cntl
                & ~(0x40000000 | 0x00000010 | 0x000F0000))  # clear halt, icache, pipe_resets
    new_cntl |= 0x3C000000                                   # pipe0-3_active
    gc1_wr(regCP_MEC_RS64_CNTL, new_cntl)
    time.sleep(0.1)  # give MEC time to start

    post = gc1_rd(regCP_MEC_RS64_CNTL)
    print(f"  CP_MEC_RS64_CNTL = 0x{post:08x}  (wrote 0x{new_cntl:08x})")

    # MEC liveness probe.
    print(f"\n== MEC liveness (INSTR_PNTR, 5 samples over 50 ms) ==")
    for i in range(5):
        ip = gc1_rd(regCP_MEC_RS64_INSTR_PNTR)
        print(f"  t={i*10}ms INSTR_PNTR=0x{ip:08x}")
        time.sleep(0.01)

    # Canary: HQD write after all-pipe enable.
    print(f"\n== HQD canary for each (me, pipe, queue) ==")
    for me in [1]:
        for pipe in [0, 1, 2, 3]:
            for queue in [0]:
                grbm = (queue << 8) | (me << 2) | pipe
                gc1_wr(regGRBM_GFX_CNTL, grbm)
                gc1_wr(regCP_HQD_PQ_BASE, 0xdeadbeef)
                rb = gc1_rd(regCP_HQD_PQ_BASE)
                gc1_wr(regCP_HQD_PQ_BASE, 0)
                stuck = (rb == 0xdeadbeef)
                mark = "STUCK ✓" if stuck else "dropped"
                print(f"  me={me} pipe={pipe} queue={queue} GRBM=0x{grbm:04x}: "
                      f"read 0x{rb:08x}  {mark}")

    gc1_wr(regGRBM_GFX_CNTL, 0)


if __name__ == "__main__":
    main()
