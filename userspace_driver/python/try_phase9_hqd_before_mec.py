"""Phase 9c (attempt 2): program HQD BEFORE enabling MEC.

Prior run (try_phase9_compute_ring.py): MEC was enabled (INSTR_PNTR=0x4070
stable — firmware executing) but CP_HQD_PQ_BASE canary failed (write
0xdeadbeef, read 0). HQD writes dropped once MEC is active.

New order:
  1. BOOTLOAD_COMPLETE
  2. _config_mec: PRGRM_CNTR_START + reset pulse (MEC still HALTED)
  3. Write HQD registers (MEC not running, driver owns them)
  4. _enable_mec: clear halt, set pipe0_active=1
  5. Observe: does MEC pick up the pre-programmed queue and execute?

This matches the hardware model: driver programs HQD state while MEC
is halted; MEC reads HQD on startup/resume and honors the queue.

Also add a full canary battery (read + write + read) at each stage
so we know exactly when HQD writes start/stop taking effect.
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

GC_B0 = 0x1260
GC_B1 = 0xA000

regGRBM_GFX_CNTL                   = 0x0900
regCP_MEC_RS64_PRGRM_CNTR_START    = 0x2900
regCP_MEC_RS64_PRGRM_CNTR_START_HI = 0x2938
regCP_MEC_RS64_CNTL                = 0x2904
regCP_MEC_RS64_INSTR_PNTR          = 0x2908
regCP_MQD_BASE_ADDR                = 0x1fa9
regCP_HQD_ACTIVE                   = 0x1fab
regCP_HQD_PQ_BASE                  = 0x1fb1
regCP_HQD_PQ_WPTR_HI               = 0x1fe0
regCP_HQD_PQ_RPTR                  = 0x1fb3
regCP_HQD_PQ_WPTR_LO               = 0x1fdf

MQD_VRAM_OFF   = 0x1800000
RING_VRAM_OFF  = 0x1802000
EOP_VRAM_OFF   = 0x1810000
RPTR_VRAM_OFF  = 0x1820000
WPTR_VRAM_OFF  = 0x1821000

MQD_SIZE  = 0x1000
RING_SIZE = 0x1000
EOP_SIZE  = 0x10000


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

    # Phase 9a: bring up
    r = gfx_bring_up(c, drv, firmware_dir=FIRMWARE_DIR)
    if not (r.bootload_status & 0x80000000):
        print("BOOTLOAD_COMPLETE not set — aborting.")
        sys.exit(1)

    fb_base = (c.mmio_read32(_MMIO_BAR, (0x1A000 + 0x0554) * 4) & 0xFFFFFF) << 24
    bar0_cpu, _ = c.map_bar(0)
    print(f"\nfb_base = 0x{fb_base:x}")

    # MC addresses
    mqd_mc  = fb_base + MQD_VRAM_OFF
    ring_mc = fb_base + RING_VRAM_OFF
    eop_mc  = fb_base + EOP_VRAM_OFF
    rptr_mc = fb_base + RPTR_VRAM_OFF
    wptr_mc = fb_base + WPTR_VRAM_OFF

    # Parse MEC firmware for PRGRM_CNTR.
    mec_blob = open(os.path.join(FIRMWARE_DIR, "gc_12_0_1_mec.bin"), "rb").read()
    us = _parse_rs64(mec_blob)

    def canary(label):
        gc1_wr(regCP_HQD_PQ_BASE, 0xdeadbeef)
        rb = gc1_rd(regCP_HQD_PQ_BASE)
        stuck = (rb == 0xdeadbeef)
        print(f"  canary [{label}]: wrote 0xdeadbeef to CP_HQD_PQ_BASE, "
              f"read 0x{rb:08x}  {'STUCK ✓' if stuck else 'DROPPED ✗'}")
        # clean up
        gc1_wr(regCP_HQD_PQ_BASE, 0x0)
        return stuck

    # Phase 9b (step 1): _config_mec only — write PRGRM_CNTR, pulse reset,
    # but DO NOT clear halt yet.
    print(f"\n== Phase 9b-1: _config_mec (no enable yet) ==")
    pre_cntl = gc1_rd(regCP_MEC_RS64_CNTL)
    print(f"  PRE  CP_MEC_RS64_CNTL = 0x{pre_cntl:08x}  INSTR_PNTR=0x{gc1_rd(regCP_MEC_RS64_INSTR_PNTR):08x}")

    # Select me=1, pipe=0 for PRGRM_CNTR writes.
    gc1_wr(regGRBM_GFX_CNTL, (1 << 2))
    gc1_wr(regCP_MEC_RS64_PRGRM_CNTR_START, (us >> 2) & 0xFFFFFFFF)
    gc1_wr(regCP_MEC_RS64_PRGRM_CNTR_START_HI, (us >> 34) & 0xFFFFFFFF)
    gc1_wr(regGRBM_GFX_CNTL, 0)
    # Pulse pipe0_reset.
    cntl = gc1_rd(regCP_MEC_RS64_CNTL)
    gc1_wr(regCP_MEC_RS64_CNTL, cntl | 0x00010000)
    cntl = gc1_rd(regCP_MEC_RS64_CNTL)
    gc1_wr(regCP_MEC_RS64_CNTL, cntl & ~0x00010000)
    post_cntl = gc1_rd(regCP_MEC_RS64_CNTL)
    print(f"  POST CP_MEC_RS64_CNTL = 0x{post_cntl:08x} (halt still set, pipe0_reset pulsed)")

    # Canary #1: HQD writes while MEC is halted & queue0 selected.
    gc1_wr(regGRBM_GFX_CNTL, (1 << 2))  # me=1, pipe=0, queue=0
    print(f"\n== canary #1: HQD write w/ MEC HALTED ==")
    canary("MEC halted, q=0 selected")

    # Program HQD now — MEC halted.
    print(f"\n== program HQD registers (MEC halted) ==")
    def _write_vram_dw(off, dw_values):
        for i, v in enumerate(dw_values):
            (ctypes.c_uint32 * 1).from_address(bar0_cpu + off + i * 4)[0] = v & 0xFFFFFFFF

    def _zero_vram(off, size):
        for i in range(0, size, 4):
            (ctypes.c_uint32 * 1).from_address(bar0_cpu + off + i)[0] = 0

    _zero_vram(MQD_VRAM_OFF, MQD_SIZE)
    _zero_vram(RING_VRAM_OFF, RING_SIZE)
    _zero_vram(RPTR_VRAM_OFF, 0x10)
    _zero_vram(WPTR_VRAM_OFF, 0x10)

    mqd = [0] * (MQD_SIZE // 4)
    mqd[0] = 0xC0310800

    ring_dw = RING_SIZE // 4
    queue_size_val = (ring_dw.bit_length() - 2) & 0x3f
    mqd[0x80] = mqd_mc & 0xFFFFFFFF
    mqd[0x81] = (mqd_mc >> 32) & 0xFFFFFFFF
    mqd[0x84] = 0x1 | (0x55 << 8)                  # persistent_state
    mqd[0x85] = 2                                   # pipe_priority
    mqd[0x86] = 0xf                                 # queue_priority
    mqd[0x87] = 0x111                               # quantum
    mqd[0x88] = (ring_mc >> 8) & 0xFFFFFFFF
    mqd[0x89] = (ring_mc >> 40) & 0xFFFFFFFF
    mqd[0x8B] = rptr_mc & 0xFFFFFFFF
    mqd[0x8C] = (rptr_mc >> 32) & 0xFFFFFFFF
    mqd[0x8D] = wptr_mc & 0xFFFFFFFF
    mqd[0x8E] = (wptr_mc >> 32) & 0xFFFFFFFF
    mqd[0x8F] = 0x40000000                          # doorbell_en=1, offset=0
    mqd[0x91] = queue_size_val | (5 << 8)           # PQ_CONTROL: queue_size + rptr_block_size=5
    mqd[0x95] = 3 << 20                             # ib_control: min_ib_avail_size=3
    mqd[0xA0] = 0x20004000                          # hq_status0
    mqd[0xA2] = 1 << 8                              # mqd_control.priv_state=1
    mqd[0xA5] = (eop_mc >> 8) & 0xFFFFFFFF
    mqd[0xA6] = (eop_mc >> 40) & 0xFFFFFFFF
    mqd[0xA7] = ((EOP_SIZE // 4).bit_length() - 2) & 0x3f
    # compute_static_thread_mgmt_se*: all SEs enabled
    for dw_idx in (0x17, 0x18, 0x1A, 0x1B, 0x2C, 0x2D, 0x2E, 0x2F):
        mqd[dw_idx] = 0xFFFFFFFF
    _write_vram_dw(MQD_VRAM_OFF, mqd)

    # Program HQD registers.
    for i in range(regCP_HQD_PQ_WPTR_HI - regCP_MQD_BASE_ADDR + 1):
        gc1_wr(regCP_MQD_BASE_ADDR + i, mqd[0x80 + i])

    # Check: did the block-copy writes stick?
    rb_base = gc1_rd(regCP_MQD_BASE_ADDR)
    rb_pq_base = gc1_rd(regCP_HQD_PQ_BASE)
    print(f"  post-HQD-write: CP_MQD_BASE_ADDR=0x{rb_base:08x} "
          f"CP_HQD_PQ_BASE=0x{rb_pq_base:08x}")

    gc1_wr(regCP_HQD_ACTIVE, 0x1)
    active_before_enable = gc1_rd(regCP_HQD_ACTIVE)
    print(f"  CP_HQD_ACTIVE (before MEC enable) = 0x{active_before_enable:08x}")

    # Deselect before MEC enable.
    gc1_wr(regGRBM_GFX_CNTL, 0)

    # Phase 9b (step 2): _enable_mec
    print(f"\n== Phase 9b-2: _enable_mec (clear halt) ==")
    cntl = gc1_rd(regCP_MEC_RS64_CNTL)
    new_cntl = (cntl & ~0x40000000) | 0x04000000
    gc1_wr(regCP_MEC_RS64_CNTL, new_cntl)
    time.sleep(0.05)
    post = gc1_rd(regCP_MEC_RS64_CNTL)
    print(f"  CP_MEC_RS64_CNTL = 0x{post:08x}")

    # Probe MEC liveness.
    for i in range(5):
        ip = gc1_rd(regCP_MEC_RS64_INSTR_PNTR)
        print(f"  t={i*10}ms INSTR_PNTR=0x{ip:08x}")
        time.sleep(0.01)

    # Canary #2: HQD write after MEC enable.
    gc1_wr(regGRBM_GFX_CNTL, (1 << 2))
    print(f"\n== canary #2: HQD write w/ MEC RUNNING ==")
    canary("MEC running, q=0 selected")

    # Re-check HQD state after MEC enable.
    active = gc1_rd(regCP_HQD_ACTIVE)
    pq_base = gc1_rd(regCP_HQD_PQ_BASE)
    print(f"  post-enable: CP_HQD_ACTIVE=0x{active:08x} CP_HQD_PQ_BASE=0x{pq_base:08x}")

    # Submit a NOP packet.
    pm4_nop = 0xC0001000
    _write_vram_dw(RING_VRAM_OFF, [pm4_nop])
    gc1_wr(regCP_HQD_PQ_WPTR_LO, 1)
    time.sleep(0.05)

    # Final state.
    print(f"\n== final state ==")
    for i in range(5):
        rptr = gc1_rd(regCP_HQD_PQ_RPTR)
        wptr = gc1_rd(regCP_HQD_PQ_WPTR_LO)
        active = gc1_rd(regCP_HQD_ACTIVE)
        rptr_report = (ctypes.c_uint32 * 1).from_address(bar0_cpu + RPTR_VRAM_OFF)[0]
        print(f"  sample {i}: rptr=0x{rptr:x} wptr=0x{wptr:x} "
              f"active=0x{active:x} rptr_report_vram=0x{rptr_report:08x}")
        time.sleep(0.02)

    gc1_wr(regGRBM_GFX_CNTL, 0)


if __name__ == "__main__":
    main()
