"""Phase 9c (attempt 6): register KIQ with RLC, then try HQD writes.

Prior attempts with MES enabled still had HQD canary drops. Missing
step found in gfx_v12_0_kiq_setting (gfx_v12_0.c:2942-2952):

  tmp = RREG32(regRLC_CP_SCHEDULERS) & 0xffffff00
  tmp |= (ring->me << 5) | (ring->pipe << 3) | ring->queue
  WREG32(regRLC_CP_SCHEDULERS, tmp | 0x80)

RLC tracks the KIQ's (me, pipe, queue) coordinates and coordinates
with MEC to gate access to other HQDs. Without this, MEC keeps all
HQDs locked.

`regRLC_CP_SCHEDULERS = 0x098a` (BASE_IDX=1).

For KIQ we use me=1 (mec+1 with mec=0), pipe=0, queue=0 — the simplest
valid configuration. Value to OR in: (1<<5) | (0<<3) | 0 | 0x80 = 0xA0.

This script:
  1. gfx_bring_up().
  2. Load MES uni_mes firmware.
  3. Enable MEC all 4 pipes.
  4. Enable MES both pipes.
  5. Write RLC_CP_SCHEDULERS to designate me=1, pipe=0, queue=0 as KIQ.
  6. HQD canary on (me=1, pipe=0, queue=0).

If canary sticks, we have write access to the KIQ HQD and can proceed
with proper MQD programming.
"""
from __future__ import annotations

import logging
import os
import struct
import sys
import time

from amd_gpu_driver.backends.macos.gfx_bringup import gfx_bring_up
from amd_gpu_driver.backends.macos.gfx_psp_autoload import _load_one
from amd_gpu_driver.backends.macos.iokit_client import IOKitClient
from amd_gpu_driver.backends.macos.psp_cmd import alloc_cmd_ctx
from amd_gpu_driver.backends.macos.smu import MP0_BASE_DW

FIRMWARE_DIR = os.path.expanduser("~/firmware/linux-firmware/amdgpu")
_MMIO_BAR = 5
GC_B1 = 0xA000

GFX_FW_TYPE_CP_MES           = 33
GFX_FW_TYPE_MES_STACK        = 34
GFX_FW_TYPE_CP_MES_KIQ       = 81
GFX_FW_TYPE_MES_KIQ_STACK    = 82

regGRBM_GFX_CNTL                 = 0x0900
regRLC_CP_SCHEDULERS             = 0x098a
regCP_MEC_RS64_PRGRM_CNTR_START  = 0x2900
regCP_MEC_RS64_PRGRM_CNTR_START_HI = 0x2938
regCP_MEC_RS64_CNTL              = 0x2904
regCP_MEC_RS64_INSTR_PNTR        = 0x2908
regCP_MES_PRGRM_CNTR_START       = 0x2800
regCP_MES_PRGRM_CNTR_START_HI    = 0x289d
regCP_MES_CNTL                   = 0x2807
regCP_HQD_PQ_BASE                = 0x1fb1
regCP_HQD_ACTIVE                 = 0x1fab


class _DriverShim:
    def __init__(self, client): self.client = client
    def alloc_dma(self, size):
        dma = self.client.alloc_dma(size)
        bus = dma.segments[0][0] if dma.segments else 0
        return (dma.cpu_addr, bus, dma.buffer_id)
    def free_dma(self, h): self.client.free_dma(h)


def _parse_mes(blob: bytes):
    (u_ver, u_sz, u_off, d_ver, d_sz, d_off,
     uc_lo, uc_hi, dd_lo, dd_hi) = struct.unpack_from("<IIIIIIIIII", blob, 32)
    return {"ucode_size": u_sz, "ucode_offset": u_off,
            "data_size": d_sz, "data_offset": d_off,
            "uc_start_addr": (uc_hi << 32) | uc_lo}


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

    # 1. Bring GFX up.
    r = gfx_bring_up(c, drv, firmware_dir=FIRMWARE_DIR)
    if not (r.bootload_status & 0x80000000):
        print("BOOTLOAD_COMPLETE not set — aborting.")
        sys.exit(1)

    # 2. Load MES firmware.
    uni_blob = open(os.path.join(FIRMWARE_DIR, "gc_12_0_1_uni_mes.bin"), "rb").read()
    h = _parse_mes(uni_blob)
    ucode = uni_blob[h["ucode_offset"]:h["ucode_offset"] + h["ucode_size"]]
    data  = uni_blob[h["data_offset"]:h["data_offset"] + h["data_size"]]

    ctx = alloc_cmd_ctx(drv)
    print(f"\n== LOAD_IP_FW (MES) ==")
    for label, fw_type, payload in [
        ("CP_MES", GFX_FW_TYPE_CP_MES, ucode),
        ("MES_STACK", GFX_FW_TYPE_MES_STACK, data),
        ("CP_MES_KIQ", GFX_FW_TYPE_CP_MES_KIQ, ucode),
        ("MES_KIQ_STACK", GFX_FW_TYPE_MES_KIQ_STACK, data),
    ]:
        _load_one(c, drv, MP0_BASE_DW, r.ring, ctx, payload, fw_type, label, strict=True)

    # 3. Enable MEC (all 4 pipes).
    mec_blob = open(os.path.join(FIRMWARE_DIR, "gc_12_0_1_mec.bin"), "rb").read()
    us = _parse_rs64(mec_blob)
    mec_lo = (us >> 2) & 0xFFFFFFFF
    mec_hi = (us >> 34) & 0xFFFFFFFF

    print(f"\n== MEC config + enable (all 4 pipes) ==")
    # Write PRGRM_CNTR for each pipe.
    for pipe in range(4):
        gc1_wr(regGRBM_GFX_CNTL, (1 << 2) | pipe)
        gc1_wr(regCP_MEC_RS64_PRGRM_CNTR_START, mec_lo)
        gc1_wr(regCP_MEC_RS64_PRGRM_CNTR_START_HI, mec_hi)
    gc1_wr(regGRBM_GFX_CNTL, 0)

    # Pulse all resets.
    cntl = gc1_rd(regCP_MEC_RS64_CNTL)
    gc1_wr(regCP_MEC_RS64_CNTL, cntl | 0x000F0010)
    cntl = gc1_rd(regCP_MEC_RS64_CNTL)
    gc1_wr(regCP_MEC_RS64_CNTL, cntl & ~0x000F0010)
    # Enable all pipes, clear halt + invalidate.
    cntl = gc1_rd(regCP_MEC_RS64_CNTL)
    new_cntl = (cntl & ~(0x40000000 | 0x00000010 | 0x000F0000)) | 0x3C000000
    gc1_wr(regCP_MEC_RS64_CNTL, new_cntl)
    time.sleep(0.05)
    print(f"  CP_MEC_RS64_CNTL = 0x{gc1_rd(regCP_MEC_RS64_CNTL):08x}")

    # 4. Enable MES (both pipes per Linux pattern).
    print(f"\n== MES enable ==")
    uc = h["uc_start_addr"] >> 2
    m_lo = uc & 0xFFFFFFFF
    m_hi = (uc >> 32) & 0xFFFFFFFF
    # Disable first.
    pre = gc1_rd(regCP_MES_CNTL)
    disable = (pre & ~(0x0C000000)) | 0x40000000 | 0x00000010 | 0x00030000
    gc1_wr(regCP_MES_CNTL, disable)
    time.sleep(0.01)
    for pipe in range(2):
        gc1_wr(regGRBM_GFX_CNTL, (3 << 2) | pipe)
        cntl = gc1_rd(regCP_MES_CNTL)
        gc1_wr(regCP_MES_CNTL, cntl | (1 << (16 + pipe)))
        gc1_wr(regCP_MES_PRGRM_CNTR_START, m_lo)
        gc1_wr(regCP_MES_PRGRM_CNTR_START_HI, m_hi)
        new_cntl = (0x04000000 if pipe == 0 else 0x0C000000)
        gc1_wr(regCP_MES_CNTL, new_cntl)
    gc1_wr(regGRBM_GFX_CNTL, 0)
    time.sleep(0.5)
    print(f"  CP_MES_CNTL = 0x{gc1_rd(regCP_MES_CNTL):08x}")

    # 5. KEY STEP: register KIQ with RLC.
    # Pick KIQ at me=1, pipe=0, queue=0 (simplest valid KIQ slot).
    kiq_me, kiq_pipe, kiq_queue = 1, 0, 0
    print(f"\n== RLC_CP_SCHEDULERS register KIQ (me={kiq_me}, pipe={kiq_pipe}, queue={kiq_queue}) ==")
    pre_sched = gc1_rd(regRLC_CP_SCHEDULERS)
    print(f"  pre:  RLC_CP_SCHEDULERS = 0x{pre_sched:08x}")

    tmp = pre_sched & 0xFFFFFF00
    tmp |= (kiq_me << 5) | (kiq_pipe << 3) | kiq_queue
    gc1_wr(regRLC_CP_SCHEDULERS, tmp | 0x80)

    post_sched = gc1_rd(regRLC_CP_SCHEDULERS)
    print(f"  post: RLC_CP_SCHEDULERS = 0x{post_sched:08x} (wrote 0x{(tmp | 0x80):08x})")

    # 6. HQD canary on me=3 (MES) — with uni_mes enabled, KIQ HQDs are
    # actually on ME=3 (MES scheduler), not ME=1 (MEC). See
    # mes_v12_0_queue_init_register: soc21_grbm_select(3, ring->pipe, 0, 0).
    print(f"\n== HQD canary on me=3 (MES) ==")
    for me in [3]:
        for pipe in [0, 1]:   # AMDGPU_MES_SCHED_PIPE=0, KIQ_PIPE=1
            for queue in [0]:
                grbm = (queue << 8) | (me << 2) | pipe
                gc1_wr(regGRBM_GFX_CNTL, grbm)
                gc1_wr(regCP_HQD_PQ_BASE, 0xdeadbeef)
                rb = gc1_rd(regCP_HQD_PQ_BASE)
                gc1_wr(regCP_HQD_PQ_BASE, 0)
                stuck = (rb == 0xdeadbeef)
                print(f"  me={me} pipe={pipe} queue={queue}: read 0x{rb:08x}  "
                      f"{'STUCK ✓' if stuck else 'dropped'}")

    print(f"\n== HQD canary on me=1 (MEC) for comparison ==")
    for pipe in [0, 1, 2, 3]:
        for queue in [0]:
            grbm = (queue << 8) | (1 << 2) | pipe
            gc1_wr(regGRBM_GFX_CNTL, grbm)
            gc1_wr(regCP_HQD_PQ_BASE, 0xdeadbeef)
            rb = gc1_rd(regCP_HQD_PQ_BASE)
            gc1_wr(regCP_HQD_PQ_BASE, 0)
            stuck = (rb == 0xdeadbeef)
            print(f"  me=1 pipe={pipe} queue={queue}: {'STUCK ✓' if stuck else 'dropped'}")
    gc1_wr(regGRBM_GFX_CNTL, 0)


if __name__ == "__main__":
    main()
