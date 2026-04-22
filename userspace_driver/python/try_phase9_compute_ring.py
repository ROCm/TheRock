"""Phase 9c: MQD + compute ring + PM4 NOP submission.

After MEC is enabled (Phase 9b), we set up one HSA-style async compute
queue and submit a PM4 NOP packet. Success criterion: the ring's
rptr_report address reflects progress past the NOP.

Layout (all in BAR0-window VRAM, MC addrs = fb_base + VRAM offset):

  Autoload buffer  : 0x0000000 .. 0x1700000  (23 MB, already in use)
  Driver table     : fb_top - 0x20000 (outside BAR0)
  MQD              : 0x1800000  (4 KB)
  Ring buffer      : 0x1802000  (4 KB)
  EOP buffer       : 0x1810000  (16 KB)
  RPTR report page : 0x1820000  (4 KB, ring rptr is written here by CP)
  WPTR poll page   : 0x1821000  (4 KB, CP polls for wptr updates here)

MQD layout is `v12_compute_mqd` from Linux v12_structs.h. The subset
we care about (DW indices shown):
  0x80/0x81: cp_mqd_base_addr_lo/hi
  0x82:      cp_hqd_active
  0x83:      cp_hqd_vmid
  0x84:      cp_hqd_persistent_state (preload_size=0x55, preload_req=1)
  0x85:      cp_hqd_pipe_priority (=2)
  0x86:      cp_hqd_queue_priority (=0xf)
  0x87:      cp_hqd_quantum (=0x111)
  0x88/0x89: cp_hqd_pq_base_lo/hi (ring >> 8)
  0x8A:      cp_hqd_pq_rptr
  0x8B/0x8C: cp_hqd_pq_rptr_report_addr_lo/hi
  0x8D/0x8E: cp_hqd_pq_wptr_poll_addr_lo/hi
  0x8F:      cp_hqd_pq_doorbell_control (doorbell_offset, doorbell_en=1)
  0x91:      cp_hqd_pq_control (queue_size, rptr_block_size=5)
  0x92/0x93: cp_hqd_ib_base_addr_lo/hi
  0x95:      cp_hqd_ib_control (min_ib_avail_size=3)
  0xA0:      cp_hqd_hq_status0 (=0x20004000)
  0xA2:      cp_mqd_control (priv_state=1)
  0xA5/0xA6: cp_hqd_eop_base_addr_lo/hi
  0xA7:      cp_hqd_eop_control (eop_size log2)
  0xB5:      cp_hqd_aql_control (0 for non-AQL)
  0xB6/0xB7: cp_hqd_pq_wptr_lo/hi

Programming sequence (tinygrad setup_ring + _enable_mec already done):
  1. Zero MQD, fill fields.
  2. Write MQD to VRAM (BAR0).
  3. HDP flush.
  4. _grbm_select(me=1, pipe=0, queue=0).
  5. Copy MQD DW[0x80..0xB7] -> CP_MQD_BASE_ADDR..CP_HQD_PQ_WPTR_HI regs.
  6. CP_HQD_ACTIVE = 1.
  7. _grbm_select() deselect.
  8. Write PM4 NOP to ring[0].
  9. Write CP_HQD_PQ_WPTR_LO = 4 (1 DWORD packet advances wptr by 1 but
     we'll write 4 bytes = 1 DWORD. Actually wptr is in DWORDs.).
  10. Poll cp_hqd_pq_rptr >= wptr or rptr_report_addr reflects rptr=1.
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

GC_B0 = 0x1260   # BASE_IDX=0
GC_B1 = 0xA000   # BASE_IDX=1

# BASE_IDX=1 (GC_B1):
regGRBM_GFX_CNTL                   = 0x0900
regCP_MEC_RS64_PRGRM_CNTR_START    = 0x2900
regCP_MEC_RS64_PRGRM_CNTR_START_HI = 0x2938
regCP_MEC_RS64_CNTL                = 0x2904
regCP_MEC_RS64_INSTR_PNTR          = 0x2908
regCP_PQ_WPTR_POLL_CNTL            = 0x1e23
regCP_MQD_BASE_ADDR                = 0x1fa9
regCP_HQD_ACTIVE                   = 0x1fab
regCP_HQD_PQ_WPTR_HI               = 0x1fe0
regCP_HQD_PQ_RPTR                  = 0x1fb3
regCP_HQD_PQ_WPTR_LO               = 0x1fdf

# BASE_IDX=0 (GC_B0):
regCP_MEC_DOORBELL_RANGE_LOWER     = 0x1dfc
regCP_MEC_DOORBELL_RANGE_UPPER     = 0x1dfd
# HDP flush (BIF BASE_IDX=2, per backends/windows/nbio_init.py convention):
# regBIF_BX0_REMAP_HDP_MEM_FLUSH_CNTL at offset 0x65 in BIF PF0 block.
# We'll skip HDP flush for now — BAR0 writes are usually coherent with GPU
# reads through the same aperture.

# VRAM offsets (all inside 256 MB BAR0 window):
MQD_VRAM_OFF   = 0x1800000
RING_VRAM_OFF  = 0x1802000
EOP_VRAM_OFF   = 0x1810000
RPTR_VRAM_OFF  = 0x1820000
WPTR_VRAM_OFF  = 0x1821000

MQD_SIZE   = 0x1000
RING_SIZE  = 0x1000       # 4 KB = 1024 DWORDs
EOP_SIZE   = 0x10000      # 64 KB
RPTR_SIZE  = 0x1000
WPTR_SIZE  = 0x1000

# CP_HQD_PQ_CONTROL bits (gc_12_0_0_sh_mask.h):
#   QUEUE_SIZE       bits 0-5    (log2(ring_dw) - 1, max 0x3f)
#   RPTR_BLOCK_SIZE  bits 8-13
#   UNORD_DISPATCH   bit 28
#   PRIV_STATE       bit 30
# CP_HQD_PERSISTENT_STATE bits:
#   PRELOAD_REQ bit 0
#   PRELOAD_SIZE bits 8-17
# CP_HQD_PQ_DOORBELL_CONTROL bits:
#   DOORBELL_OFFSET bits 2-27
#   DOORBELL_EN bit 30
# CP_HQD_EOP_CONTROL bits:
#   EOP_SIZE bits 0-5 (log2(eop_dw) - 1)
# CP_MQD_CONTROL bits:
#   PRIV_STATE bit 8 (actually varies by gen; use 0x100)


class _DriverShim:
    def __init__(self, client): self.client = client
    def alloc_dma(self, size):
        dma = self.client.alloc_dma(size)
        bus = dma.segments[0][0] if dma.segments else 0
        return (dma.cpu_addr, bus, dma.buffer_id)
    def free_dma(self, h): self.client.free_dma(h)


def _parse_rs64_header(blob: bytes) -> dict:
    (u_feat, u_sz, u_off, d_sz, d_off, u_start_lo, u_start_hi) = \
        struct.unpack_from("<IIIIIII", blob, 32)
    return {"ucode_start_addr": (u_start_hi << 32) | u_start_lo}


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    c = IOKitClient(); c.open()
    info = c.get_info()
    print(f"device=0x{info.device_id:04x} rev=0x{info.revision_id:02x}")
    drv = _DriverShim(c)

    def gc1_rd(o): return c.mmio_read32(_MMIO_BAR, (GC_B1 + o) * 4)
    def gc1_wr(o, v): c.mmio_write32(_MMIO_BAR, (GC_B1 + o) * 4, v & 0xFFFFFFFF)
    def gc0_wr(o, v): c.mmio_write32(_MMIO_BAR, (GC_B0 + o) * 4, v & 0xFFFFFFFF)

    # ========== Phase 9a: bring up ==========
    result = gfx_bring_up(c, drv, firmware_dir=FIRMWARE_DIR)
    if not (result.bootload_status & 0x80000000):
        print("BOOTLOAD_COMPLETE not set — aborting.")
        sys.exit(1)

    # Get VRAM base.
    fb_base_reg = c.mmio_read32(_MMIO_BAR, (0x1A000 + 0x0554) * 4) & 0xFFFFFF
    fb_base = fb_base_reg << 24
    print(f"\nfb_base = 0x{fb_base:x}")

    # Map BAR0 for VRAM writes.
    bar0_cpu, bar0_size = c.map_bar(0)
    print(f"BAR0 mapped at CPU=0x{bar0_cpu:x}  size=0x{bar0_size:x}")

    # Compute MC addresses.
    mqd_mc  = fb_base + MQD_VRAM_OFF
    ring_mc = fb_base + RING_VRAM_OFF
    eop_mc  = fb_base + EOP_VRAM_OFF
    rptr_mc = fb_base + RPTR_VRAM_OFF
    wptr_mc = fb_base + WPTR_VRAM_OFF
    print(f"  MQD  MC = 0x{mqd_mc:x}  ring MC = 0x{ring_mc:x}")
    print(f"  EOP  MC = 0x{eop_mc:x}   rptr MC = 0x{rptr_mc:x}  wptr MC = 0x{wptr_mc:x}")

    # ========== Phase 9b: config + enable MEC ==========
    mec_blob = open(os.path.join(FIRMWARE_DIR, "gc_12_0_1_mec.bin"), "rb").read()
    mec_hdr = _parse_rs64_header(mec_blob)
    us = mec_hdr["ucode_start_addr"]
    start_lo = (us >> 2) & 0xFFFFFFFF
    start_hi = (us >> 34) & 0xFFFFFFFF

    print(f"\n== MEC config/enable (me=1, pipe=0) ==")

    # Tinygrad _config_mec is read-modify-write on CP_MEC_RS64_CNTL.
    # Our prior raw-write path clobbered pipe1/2/3_reset bits that
    # initial state had set, which may have crashed MEC.
    # Bit layout: PIPE0_RESET=16 PIPE1_RESET=17 PIPE2_RESET=18
    #   PIPE3_RESET=19 PIPE0_ACTIVE=26 PIPE1_ACTIVE=27 PIPE2_ACTIVE=28
    #   PIPE3_ACTIVE=29 HALT=30 STEP=31.
    pre_cntl = gc1_rd(regCP_MEC_RS64_CNTL)
    print(f"  PRE  CP_MEC_RS64_CNTL = 0x{pre_cntl:08x}")

    # Step 1: select me=1, pipe=0.
    gc1_wr(regGRBM_GFX_CNTL, (1 << 2))
    # Step 2: write program counter (me=1,pipe=0 selected so per-pipe).
    gc1_wr(regCP_MEC_RS64_PRGRM_CNTR_START, start_lo)
    gc1_wr(regCP_MEC_RS64_PRGRM_CNTR_START_HI, start_hi)
    # Step 3: deselect.
    gc1_wr(regGRBM_GFX_CNTL, 0)
    # Step 4: set pipe0_reset bit (RMW).
    cntl = gc1_rd(regCP_MEC_RS64_CNTL)
    gc1_wr(regCP_MEC_RS64_CNTL, cntl | 0x00010000)
    # Step 5: clear pipe0_reset (RMW).
    cntl = gc1_rd(regCP_MEC_RS64_CNTL)
    gc1_wr(regCP_MEC_RS64_CNTL, cntl & ~0x00010000)
    # Step 6 (_enable_mec): RMW — clear halt (bit 30), set pipe0_active
    # (bit 26), leave other pipes' reset/active bits at their current
    # value.
    cntl = gc1_rd(regCP_MEC_RS64_CNTL)
    new_cntl = (cntl & ~0x40000000) | 0x04000000   # halt=0, pipe0_active=1
    gc1_wr(regCP_MEC_RS64_CNTL, new_cntl)
    time.sleep(0.05)
    post_cntl = gc1_rd(regCP_MEC_RS64_CNTL)
    print(f"  POST CP_MEC_RS64_CNTL = 0x{post_cntl:08x}  (wrote 0x{new_cntl:08x})")

    # Sample MEC INSTR_PNTR to see if firmware is actually executing.
    # If static (e.g., 0), MEC is idle / not running — HQD writes will
    # go nowhere. If it's advancing or stable at a non-zero address,
    # MEC is alive.
    print(f"\n== MEC liveness probe (INSTR_PNTR, 5 samples over 50 ms) ==")
    for i in range(5):
        ip = gc1_rd(regCP_MEC_RS64_INSTR_PNTR)
        print(f"  t={i*10}ms CP_MEC_RS64_INSTR_PNTR = 0x{ip:08x}")
        time.sleep(0.01)

    # Program MEC doorbell range (per tinygrad AM_GFX.init_hw).
    print(f"\n== program CP_MEC_DOORBELL_RANGE ==")
    gc0_wr(regCP_MEC_DOORBELL_RANGE_LOWER, 0x0)
    gc0_wr(regCP_MEC_DOORBELL_RANGE_UPPER, 0xf8)

    # ========== Phase 9c: MQD + ring + PM4 submission ==========

    # Zero VRAM regions we'll touch (via BAR0 DWORD writes — safer than
    # ctypes.memset which SIGBUSes past ~1 MB on Apple Silicon).
    def _zero_vram(off, size):
        for i in range(0, size, 4):
            (ctypes.c_uint32 * 1).from_address(bar0_cpu + off + i)[0] = 0

    def _write_vram_dw(off, dw_values):
        for i, v in enumerate(dw_values):
            (ctypes.c_uint32 * 1).from_address(bar0_cpu + off + i * 4)[0] = v & 0xFFFFFFFF

    def _read_vram_dw(off):
        return (ctypes.c_uint32 * 1).from_address(bar0_cpu + off)[0]

    print(f"\n== zero MQD + ring + EOP + rptr + wptr ==")
    _zero_vram(MQD_VRAM_OFF,  MQD_SIZE)
    _zero_vram(RING_VRAM_OFF, RING_SIZE)
    # EOP is big; just zero first 4 KB for now.
    _zero_vram(EOP_VRAM_OFF,  0x1000)
    _zero_vram(RPTR_VRAM_OFF, 0x10)
    _zero_vram(WPTR_VRAM_OFF, 0x10)

    # Build MQD as a 1024-DWORD array (4 KB). Only populate the fields
    # that matter for our minimal non-AQL compute queue. All other
    # fields stay zero.
    mqd = [0] * (MQD_SIZE // 4)

    # CP_HQD_PQ_CONTROL:
    #   QUEUE_SIZE      = log2(ring_dw) - 1
    #   RPTR_BLOCK_SIZE = 5 (shift 8)
    ring_dw = RING_SIZE // 4
    queue_size_val = (ring_dw.bit_length() - 2) & 0x3f
    rptr_block_size = 5
    cp_hqd_pq_control = queue_size_val | (rptr_block_size << 8)

    # CP_HQD_PERSISTENT_STATE: preload_size=0x55 (shift 8), preload_req=1
    cp_hqd_persistent_state = 0x1 | (0x55 << 8)

    # CP_HQD_PQ_DOORBELL_CONTROL: doorbell_en=1 (bit 30), offset=0.
    # tinygrad sets doorbell_en=1 even when not yet using doorbells —
    # activation may require this even if we advance wptr by register.
    cp_hqd_pq_doorbell_control = (0 << 2) | (1 << 30)

    # CP_HQD_EOP_CONTROL: eop_size = log2(eop_dw) - 1
    eop_dw = EOP_SIZE // 4
    cp_hqd_eop_control = (eop_dw.bit_length() - 2) & 0x3f

    # CP_HQD_IB_CONTROL: min_ib_avail_size=3 (shift 0x14 = 20)
    cp_hqd_ib_control = 3 << 20

    # CP_MQD_CONTROL: priv_state=1 (shift 8)
    cp_mqd_control = 1 << 8

    # MQD[0] = type-3 PM4 packet header identifying the MQD to CP.
    # Value 0xC0310800 from tinygrad (struct_v12_compute_mqd.header).
    mqd[0] = 0xC0310800

    mqd[0x80] = mqd_mc & 0xFFFFFFFF                    # cp_mqd_base_addr_lo
    mqd[0x81] = (mqd_mc >> 32) & 0xFFFFFFFF            # cp_mqd_base_addr_hi
    mqd[0x82] = 0                                       # cp_hqd_active (will set separately)
    mqd[0x83] = 0                                       # cp_hqd_vmid
    mqd[0x84] = cp_hqd_persistent_state
    mqd[0x85] = 2                                       # cp_hqd_pipe_priority
    mqd[0x86] = 0xf                                     # cp_hqd_queue_priority
    mqd[0x87] = 0x111                                   # cp_hqd_quantum
    mqd[0x88] = (ring_mc >> 8) & 0xFFFFFFFF             # cp_hqd_pq_base_lo
    mqd[0x89] = (ring_mc >> 40) & 0xFFFFFFFF            # cp_hqd_pq_base_hi
    mqd[0x8A] = 0                                       # cp_hqd_pq_rptr
    mqd[0x8B] = rptr_mc & 0xFFFFFFFF                    # cp_hqd_pq_rptr_report_addr_lo
    mqd[0x8C] = (rptr_mc >> 32) & 0xFFFFFFFF            # cp_hqd_pq_rptr_report_addr_hi
    mqd[0x8D] = wptr_mc & 0xFFFFFFFF                    # cp_hqd_pq_wptr_poll_addr_lo
    mqd[0x8E] = (wptr_mc >> 32) & 0xFFFFFFFF            # cp_hqd_pq_wptr_poll_addr_hi
    mqd[0x8F] = cp_hqd_pq_doorbell_control
    mqd[0x91] = cp_hqd_pq_control
    mqd[0x95] = cp_hqd_ib_control
    mqd[0xA0] = 0x20004000                              # cp_hqd_hq_status0
    mqd[0xA2] = cp_mqd_control
    mqd[0xA5] = (eop_mc >> 8) & 0xFFFFFFFF              # cp_hqd_eop_base_addr_lo
    mqd[0xA6] = (eop_mc >> 40) & 0xFFFFFFFF             # cp_hqd_eop_base_addr_hi
    mqd[0xA7] = cp_hqd_eop_control
    mqd[0xB5] = 0                                       # cp_hqd_aql_control
    mqd[0xB6] = 0                                       # cp_hqd_pq_wptr_lo
    mqd[0xB7] = 0                                       # cp_hqd_pq_wptr_hi

    # compute_static_thread_mgmt_se[0..7] = 0xffffffff (enable all SEs).
    # Per v12_compute_mqd (v12_structs.h:698-722):
    #   se0 at DW 0x17, se1 at 0x18, se2 at 0x1A, se3 at 0x1B
    #   se4 at 0x2C, se5 at 0x2D, se6 at 0x2E, se7 at 0x2F
    # These are MQD-only (no direct registers) — CP reads them from MQD VRAM
    # at context save/restore time.
    for dw_idx in (0x17, 0x18, 0x1A, 0x1B, 0x2C, 0x2D, 0x2E, 0x2F):
        mqd[dw_idx] = 0xFFFFFFFF

    print(f"\n== populate + write MQD to VRAM at offset 0x{MQD_VRAM_OFF:x} ==")
    print(f"  queue_size (log2-1)   = {queue_size_val}  (ring = {ring_dw} DWs)")
    print(f"  cp_hqd_pq_control     = 0x{cp_hqd_pq_control:08x}")
    print(f"  cp_hqd_eop_control    = 0x{cp_hqd_eop_control:08x}")
    _write_vram_dw(MQD_VRAM_OFF, mqd)

    # Read back a few fields to verify.
    rb_base_lo = _read_vram_dw(MQD_VRAM_OFF + 0x80 * 4)
    rb_pq_base_lo = _read_vram_dw(MQD_VRAM_OFF + 0x88 * 4)
    print(f"  MQD readback: cp_mqd_base_addr_lo = 0x{rb_base_lo:08x}  "
          f"cp_hqd_pq_base_lo = 0x{rb_pq_base_lo:08x}")

    # Disable wptr polling (Linux gfx_v12_0_kiq_init_register step 1).
    print(f"\n== disable CP_PQ_WPTR_POLL_CNTL.EN ==")
    wp = gc1_rd(regCP_PQ_WPTR_POLL_CNTL)
    print(f"  pre: 0x{wp:08x}")
    # EN is typically the top bit (0x80000000) or similar — safest: clear bit 31.
    gc1_wr(regCP_PQ_WPTR_POLL_CNTL, wp & 0x7FFFFFFF)

    # ========== Copy MQD DW[0x80..0xB7] to CP_HQD_* registers ==========
    print(f"\n== program HQD registers from MQD ==")
    gc1_wr(regGRBM_GFX_CNTL, (1 << 2))  # me=1, pipe=0, queue=0
    time.sleep(0.001)

    # Sanity: write and read back CP_HQD_PQ_BASE to confirm HQD writes
    # are routing to the selected pipe/queue.
    gc1_wr(0x1fb1, 0xdeadbeef)  # regCP_HQD_PQ_BASE
    rb = gc1_rd(0x1fb1)
    print(f"  canary: wrote 0xdeadbeef to CP_HQD_PQ_BASE, readback 0x{rb:08x}")

    for i in range(regCP_HQD_PQ_WPTR_HI - regCP_MQD_BASE_ADDR + 1):
        mqd_idx = 0x80 + i
        reg = regCP_MQD_BASE_ADDR + i
        gc1_wr(reg, mqd[mqd_idx])

    # Enable the queue: CP_HQD_ACTIVE = 1.
    print(f"  writing CP_HQD_ACTIVE = 1")
    gc1_wr(regCP_HQD_ACTIVE, 0x1)
    time.sleep(0.01)
    active = gc1_rd(regCP_HQD_ACTIVE)
    print(f"  CP_HQD_ACTIVE readback = 0x{active:08x}")

    rptr_pre = gc1_rd(regCP_HQD_PQ_RPTR)
    wptr_pre = gc1_rd(regCP_HQD_PQ_WPTR_LO)
    print(f"  CP_HQD_PQ_RPTR = 0x{rptr_pre:08x}  CP_HQD_PQ_WPTR_LO = 0x{wptr_pre:08x}")

    # Deselect while we still have context — will re-select before wptr write.
    gc1_wr(regGRBM_GFX_CNTL, 0)

    # ========== Write a PM4 NOP packet to the ring ==========
    # PM4 NOP: type-3 packet, opcode 0x10, count_dw = 0 (header only).
    # Header layout: [31:30]=type (3), [29:16]=count-1 (0 for 1 DW total),
    #                [15:8]=opcode, [7:0]=... (predicates etc).
    # For a pure NOP with no body: actually a type-3 packet with count=0
    # means header only = 1 DW total. Opcode for NOP is 0x10.
    # Header = 0xC0001000  (type=3, count=0, opcode=0x10)
    pm4_nop = 0xC0001000

    print(f"\n== write PM4 NOP 0x{pm4_nop:08x} to ring[0] ==")
    _write_vram_dw(RING_VRAM_OFF, [pm4_nop])

    # Re-select to write wptr.
    gc1_wr(regGRBM_GFX_CNTL, (1 << 2))
    time.sleep(0.001)

    # Advance wptr by 1 DW.
    print(f"  advance wptr to 1")
    gc1_wr(regCP_HQD_PQ_WPTR_LO, 1)
    time.sleep(0.05)

    # Read back rptr / wptr / cp_hqd_active a few times.
    print(f"\n== poll ring state (10 samples, 20 ms apart) ==")
    for i in range(10):
        rptr = gc1_rd(regCP_HQD_PQ_RPTR)
        wptr = gc1_rd(regCP_HQD_PQ_WPTR_LO)
        active = gc1_rd(regCP_HQD_ACTIVE)
        rptr_report = _read_vram_dw(RPTR_VRAM_OFF)
        print(f"  sample {i}: rptr=0x{rptr:x} wptr=0x{wptr:x} "
              f"active=0x{active:x} rptr_report_vram=0x{rptr_report:08x}")
        time.sleep(0.02)

    gc1_wr(regGRBM_GFX_CNTL, 0)


if __name__ == "__main__":
    main()
