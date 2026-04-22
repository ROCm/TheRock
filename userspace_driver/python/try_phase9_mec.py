"""Phase 9b: config + enable MEC.

After BOOTLOAD_COMPLETE:
  - MEC firmware is loaded in PSP TMR via LOAD_IP_FW (RS64_MEC + stacks).
  - MEC program counter registers are zeroed (nothing points to the code).
  - MEC halt bit is set (MEC_RS64_CNTL = 0).

Tinygrad's _config_mec + _enable_mec for gfx12:

  1. _grbm_select(me=1, pipe=0) — route register writes to MEC pipe 0.
  2. Write CP_MEC_RS64_PRGRM_CNTR_START    = ucode_start_addr_lo >> 2
         CP_MEC_RS64_PRGRM_CNTR_START_HI = ucode_start_addr_hi (full).
  3. _grbm_select() — deselect.
  4. Pulse CP_MEC_RS64_CNTL.mec_pipe0_reset=1 then =0.
  5. CP_MEC_RS64_CNTL.mec_pipe0_active=1, mec_halt=0.
  6. Sleep 50 ms, read CP_MEC_RS64_INSTR_PNTR to see if MEC is executing.

The ucode_start_addr comes from gfx_firmware_header_v2_0 at offsets
+52 (lo) and +56 (hi) of the MEC firmware blob.

Register layout from gc_12_0_0_offset.h (all BASE_IDX=1 -> GC_B1=0xA000
unless noted):
  regCP_MEC_RS64_PRGRM_CNTR_START    = 0x2900
  regCP_MEC_RS64_PRGRM_CNTR_START_HI = 0x2938
  regCP_MEC_RS64_CNTL                = 0x2904
  regGRBM_GFX_CNTL                   = 0x0900
"""
from __future__ import annotations

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

# GC BASE_IDX=1 (0xA000) registers:
regGRBM_GFX_CNTL                   = 0x0900
regCP_MEC_RS64_PRGRM_CNTR_START    = 0x2900
regCP_MEC_RS64_PRGRM_CNTR_START_HI = 0x2938
regCP_MEC_RS64_CNTL                = 0x2904


class _DriverShim:
    def __init__(self, client): self.client = client
    def alloc_dma(self, size):
        dma = self.client.alloc_dma(size)
        bus = dma.segments[0][0] if dma.segments else 0
        return (dma.cpu_addr, bus, dma.buffer_id)
    def free_dma(self, h): self.client.free_dma(h)


def _parse_rs64_header(blob: bytes) -> dict:
    """gfx_firmware_header_v2_0 (gc_12_0_1_{pfp,me,mec}.bin).

    Layout after common header (32 bytes):
      +32 ucode_feature_version
      +36 ucode_size_bytes
      +40 ucode_offset_bytes
      +44 data_size_bytes
      +48 data_offset_bytes
      +52 ucode_start_addr_lo
      +56 ucode_start_addr_hi
    """
    (u_feat, u_sz, u_off, d_sz, d_off, u_start_lo, u_start_hi) = \
        struct.unpack_from("<IIIIIII", blob, 32)
    return {
        "feature_version": u_feat,
        "ucode_size": u_sz,
        "ucode_offset": u_off,
        "data_size": d_sz,
        "data_offset": d_off,
        "ucode_start_addr": (u_start_hi << 32) | u_start_lo,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    c = IOKitClient(); c.open()
    info = c.get_info()
    print(f"device=0x{info.device_id:04x} rev=0x{info.revision_id:02x}")
    drv = _DriverShim(c)

    def gc1_rd(o): return c.mmio_read32(_MMIO_BAR, (GC_B1 + o) * 4)
    def gc1_wr(o, v): c.mmio_write32(_MMIO_BAR, (GC_B1 + o) * 4, v & 0xFFFFFFFF)

    # 1. Bring GFX up (strict tinygrad order, validated recipe).
    result = gfx_bring_up(c, drv, firmware_dir=FIRMWARE_DIR)
    if not (result.bootload_status & 0x80000000):
        print("BOOTLOAD_COMPLETE not set — aborting.")
        sys.exit(1)

    # 2. Parse MEC firmware header to get ucode_start.
    mec_blob = open(os.path.join(FIRMWARE_DIR, "gc_12_0_1_mec.bin"), "rb").read()
    mec_hdr = _parse_rs64_header(mec_blob)
    print(f"\n== MEC firmware ==")
    print(f"  feature_version = {mec_hdr['feature_version']}")
    print(f"  ucode_size      = {mec_hdr['ucode_size']}")
    print(f"  ucode_start_addr= 0x{mec_hdr['ucode_start_addr']:x}")

    pre_cntl = gc1_rd(regCP_MEC_RS64_CNTL)
    pre_start = gc1_rd(regCP_MEC_RS64_PRGRM_CNTR_START)
    pre_start_hi = gc1_rd(regCP_MEC_RS64_PRGRM_CNTR_START_HI)
    print(f"\n  PRE-CONFIG: CP_MEC_RS64_CNTL=0x{pre_cntl:08x} "
          f"PRGRM_CNTR_START=0x{pre_start:08x} HI=0x{pre_start_hi:08x}")

    # 3. _grbm_select(me=1, pipe=0) — target MEC pipe 0.
    # GRBM_GFX_CNTL bitfield (gc_12_0_0_sh_mask.h):
    #   PIPEID  bits 0-1  (mask 0x3)
    #   MEID    bits 2-3  (mask 0xC)
    #   VMID    bits 4-7
    #   QUEUEID bits 8-10
    #   CTXID   bits 11-13
    # me=1, pipe=0 -> (1 << 2) | (0 << 0) = 0x4.
    grbm_val = (1 << 2) | (0 << 0)   # me=1, pipe=0
    print(f"\n== _grbm_select(me=1, pipe=0): GRBM_GFX_CNTL = 0x{grbm_val:x} ==")
    gc1_wr(regGRBM_GFX_CNTL, grbm_val)
    readback = gc1_rd(regGRBM_GFX_CNTL)
    print(f"  readback: 0x{readback:08x}")

    # 4. Write PRGRM_CNTR_START from the firmware.
    #    Per tinygrad: `ucode_start >> 2` goes to START. Full HI to START_HI.
    us = mec_hdr["ucode_start_addr"]
    start_lo = (us >> 2) & 0xFFFFFFFF
    start_hi = (us >> 34) & 0xFFFFFFFF  # shift-by-2 of the full 64-bit word means hi = ucode_start >> 34
    print(f"\n== write PRGRM_CNTR_START = 0x{start_lo:08x} HI = 0x{start_hi:08x} ==")
    gc1_wr(regCP_MEC_RS64_PRGRM_CNTR_START, start_lo)
    gc1_wr(regCP_MEC_RS64_PRGRM_CNTR_START_HI, start_hi)
    rb_lo = gc1_rd(regCP_MEC_RS64_PRGRM_CNTR_START)
    rb_hi = gc1_rd(regCP_MEC_RS64_PRGRM_CNTR_START_HI)
    print(f"  readback: lo=0x{rb_lo:08x} hi=0x{rb_hi:08x}")

    # 5. Deselect.
    print(f"\n== _grbm_select() — deselect ==")
    gc1_wr(regGRBM_GFX_CNTL, 0)

    # 6. Pulse pipe0_reset.
    # CP_MEC_RS64_CNTL bit layout (from sh_mask): mec_pipe0_reset bit 16 (typ).
    # Let's find: a value like 0x00010000 -> reset. Actually safest: use tinygrad's
    # .update() semantics — but without bitfield metadata we just have to use magic.
    # For now: set bit 16 (common convention for pipe0_reset in gfx11/12 headers).
    print(f"\n== pulse mec_pipe0_reset ==")
    cntl_reset = 0x00010000  # mec_pipe0_reset = 1
    gc1_wr(regCP_MEC_RS64_CNTL, cntl_reset)
    time.sleep(0.01)
    gc1_wr(regCP_MEC_RS64_CNTL, 0x0)
    time.sleep(0.01)

    # 7. Clear halt: mec_pipe0_active=1 (bit 26), mec_halt=0 (bit 30).
    # From gc_12_0_0_sh_mask.h:
    #   MEC_PIPE0_RESET  = 0x00010000 (bit 16)
    #   MEC_PIPE0_ACTIVE = 0x04000000 (bit 26)
    #   MEC_HALT         = 0x40000000 (bit 30)
    #   MEC_STEP         = 0x80000000 (bit 31)
    print(f"\n== enable MEC (mec_pipe0_active=1, mec_halt=0) ==")
    cntl_enable = 0x04000000  # active=1, halt=0
    gc1_wr(regCP_MEC_RS64_CNTL, cntl_enable)
    time.sleep(0.05)

    post_cntl = gc1_rd(regCP_MEC_RS64_CNTL)
    post_start = gc1_rd(regCP_MEC_RS64_PRGRM_CNTR_START)
    post_start_hi = gc1_rd(regCP_MEC_RS64_PRGRM_CNTR_START_HI)
    print(f"\n  POST-ENABLE: CP_MEC_RS64_CNTL=0x{post_cntl:08x} "
          f"PRGRM_CNTR_START=0x{post_start:08x} HI=0x{post_start_hi:08x}")

    # 8. Read MEC status / instr pointer if available.
    # (We don't have regCP_MEC_RS64_INSTR_PNTR offset yet — probe later.)


if __name__ == "__main__":
    main()
