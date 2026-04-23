"""Phase 9c (attempt 7): full MES-KIQ HQD bringup per Linux exact sequence.

Previous attempts: HQD canary read-back was always 0. Newer
understanding: GRBM_GFX_CNTL is write-latched (reads return 0),
and likely CP_HQD_* registers have similar per-queue-context behavior
— writes go into CP's context store for the selected queue, reads
return 0 until queue is active and state is consistent.

This script stops relying on readback and watches side effects:
  - CP_STAT (0x0f40 BASE_IDX=0) — non-zero means CP is busy
  - GRBM_STATUS (0x0da4 BASE_IDX=0) — compute block bits
  - rptr_report VRAM — CP writes current rptr here; changes => queue active
  - CP_HQD_PQ_RPTR readback — might work once queue is truly active

Follows mes_v12_0_mqd_init (mes_v12_0.c:1270-1377) +
mes_v12_0_queue_init_register (1379-1439) exactly:

  soc21_grbm_select(3, 1, 0, 0)  // me=3, pipe=KIQ_PIPE=1

  Step 1: RMW CP_HQD_VMID.VMID = 0
  Step 2: RMW CP_HQD_PQ_DOORBELL_CONTROL.DOORBELL_EN = 0 (disable)
  Step 3: CP_MQD_BASE_ADDR_LO/HI = MQD address
  Step 4: CP_MQD_CONTROL = 0 (not `data` as the comment says!)
  Step 5: CP_HQD_PQ_BASE_LO/HI = ring_mc >> 8
  Step 6: CP_HQD_PQ_RPTR_REPORT_ADDR_LO/HI = rptr_mc
  Step 7: CP_HQD_PQ_CONTROL = computed (QUEUE_SIZE, RPTR_BLOCK, etc)
  Step 8: CP_HQD_PQ_WPTR_POLL_ADDR_LO/HI = wptr_mc
  Step 9: CP_HQD_PQ_DOORBELL_CONTROL = final (DOORBELL_EN=1 if use_doorbell)
  Step 10: CP_HQD_PERSISTENT_STATE (PRELOAD_SIZE=0x55)
  Step 11: CP_HQD_ACTIVE = 1

Defaults from gfx_v12_0.c:
  CP_HQD_EOP_CONTROL_DEFAULT     = 0x00000006
  CP_MQD_CONTROL_DEFAULT         = 0x00000100
  CP_HQD_PQ_CONTROL_DEFAULT      = 0x00308509
  CP_HQD_PERSISTENT_STATE_DEFAULT= 0x0be05501
  CP_HQD_IB_CONTROL_DEFAULT      = 0x00300000
"""
from __future__ import annotations

import ctypes
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
GC_B0 = 0x1260
GC_B1 = 0xA000

# MES firmware types
GFX_FW_TYPE_CP_MES           = 33
GFX_FW_TYPE_MES_STACK        = 34
GFX_FW_TYPE_CP_MES_KIQ       = 81
GFX_FW_TYPE_MES_KIQ_STACK    = 82

# BASE_IDX=1 regs
regGRBM_GFX_CNTL                 = 0x0900
regRLC_CP_SCHEDULERS             = 0x098a
regCP_MEC_RS64_PRGRM_CNTR_START  = 0x2900
regCP_MEC_RS64_PRGRM_CNTR_START_HI = 0x2938
regCP_MEC_RS64_CNTL              = 0x2904
regCP_MES_PRGRM_CNTR_START       = 0x2800
regCP_MES_PRGRM_CNTR_START_HI    = 0x289d
regCP_MES_CNTL                   = 0x2807
# HQD block (BASE_IDX=1, contiguous 0x1fa9-0x1fe0):
regCP_MQD_BASE_ADDR              = 0x1fa9
regCP_MQD_BASE_ADDR_HI           = 0x1faa
regCP_HQD_ACTIVE                 = 0x1fab
regCP_HQD_VMID                   = 0x1fac
regCP_HQD_PERSISTENT_STATE       = 0x1fad
regCP_HQD_PQ_BASE                = 0x1fb1
regCP_HQD_PQ_BASE_HI             = 0x1fb2
regCP_HQD_PQ_RPTR                = 0x1fb3
regCP_HQD_PQ_RPTR_REPORT_ADDR    = 0x1fb4
regCP_HQD_PQ_RPTR_REPORT_ADDR_HI = 0x1fb5
regCP_HQD_PQ_WPTR_POLL_ADDR      = 0x1fb6
regCP_HQD_PQ_WPTR_POLL_ADDR_HI   = 0x1fb7
regCP_HQD_PQ_DOORBELL_CONTROL    = 0x1fb8
regCP_HQD_PQ_CONTROL             = 0x1fba
regCP_HQD_IB_CONTROL             = 0x1fbe
regCP_HQD_PQ_WPTR_LO             = 0x1fdf
regCP_HQD_PQ_WPTR_HI             = 0x1fe0
regCP_MQD_CONTROL                = 0x1fcb
# MES status registers (BASE_IDX=1)
regCP_MES_HEADER_DUMP            = 0x280d
regCP_MES_INSTR_PNTR             = 0x2813
regCP_MES_MINSTRET_LO            = 0x282a
regCP_MES_MINSTRET_HI            = 0x282b
# BASE_IDX=0 regs
regGRBM_STATUS                   = 0x0da4
regCP_STAT                       = 0x0f40

# Register defaults (from gfx_v12_0.c:63-70)
CP_HQD_EOP_CONTROL_DEFAULT       = 0x00000006
CP_MQD_CONTROL_DEFAULT           = 0x00000100
CP_HQD_PQ_CONTROL_DEFAULT        = 0x00308509
CP_HQD_PERSISTENT_STATE_DEFAULT  = 0x0be05501

# VRAM layout (all in BAR0 window)
MQD_VRAM_OFF   = 0x1800000  # 4 KB MQD
RING_VRAM_OFF  = 0x1802000  # 4 KB ring
EOP_VRAM_OFF   = 0x1810000  # 64 KB EOP
RPTR_VRAM_OFF  = 0x1820000  # 4 KB rptr_report (CP writes here)
WPTR_VRAM_OFF  = 0x1821000  # 4 KB wptr_poll (CP reads here)

MQD_SIZE  = 0x1000
RING_SIZE = 0x1000
EOP_SIZE  = 0x10000
MES_EOP_SIZE = 0x1000  # MES_EOP_SIZE from amdgpu


class _DriverShim:
    def __init__(self, client): self.client = client
    def alloc_dma(self, size):
        dma = self.client.alloc_dma(size)
        bus = dma.segments[0][0] if dma.segments else 0
        return (dma.cpu_addr, bus, dma.buffer_id)
    def free_dma(self, h): self.client.free_dma(h)


def _parse_mes(blob):
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
    def gc0_rd(o): return c.mmio_read32(_MMIO_BAR, (GC_B0 + o) * 4)

    # ==== 1. Bring up GFX ====
    r = gfx_bring_up(c, drv, firmware_dir=FIRMWARE_DIR)
    if not (r.bootload_status & 0x80000000):
        print("BOOTLOAD_COMPLETE not set — aborting.")
        sys.exit(1)

    fb_base = (c.mmio_read32(_MMIO_BAR, (0x1A000 + 0x0554) * 4) & 0xFFFFFF) << 24
    bar0_cpu, _ = c.map_bar(0)
    print(f"\nfb_base = 0x{fb_base:x}")

    mqd_mc  = fb_base + MQD_VRAM_OFF
    ring_mc = fb_base + RING_VRAM_OFF
    eop_mc  = fb_base + EOP_VRAM_OFF
    rptr_mc = fb_base + RPTR_VRAM_OFF
    wptr_mc = fb_base + WPTR_VRAM_OFF
    print(f"  MQD  MC = 0x{mqd_mc:x}")
    print(f"  ring MC = 0x{ring_mc:x}")
    print(f"  EOP  MC = 0x{eop_mc:x}")
    print(f"  rptr MC = 0x{rptr_mc:x}")
    print(f"  wptr MC = 0x{wptr_mc:x}")

    def vram_wr(off, val):
        (ctypes.c_uint32 * 1).from_address(bar0_cpu + off)[0] = val & 0xFFFFFFFF

    def vram_rd(off):
        return (ctypes.c_uint32 * 1).from_address(bar0_cpu + off)[0]

    # Zero all VRAM regions.
    for base, size in [(MQD_VRAM_OFF, MQD_SIZE), (RING_VRAM_OFF, RING_SIZE),
                       (EOP_VRAM_OFF, 0x1000), (RPTR_VRAM_OFF, 0x20),
                       (WPTR_VRAM_OFF, 0x20)]:
        for i in range(0, size, 4):
            vram_wr(base + i, 0)

    # ==== 2. Load MES firmware ====
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

    # ==== 3. Enable MEC (all 4 pipes) ====
    mec_blob = open(os.path.join(FIRMWARE_DIR, "gc_12_0_1_mec.bin"), "rb").read()
    us = _parse_rs64(mec_blob)
    mec_lo = (us >> 2) & 0xFFFFFFFF
    mec_hi = (us >> 34) & 0xFFFFFFFF

    for pipe in range(4):
        gc1_wr(regGRBM_GFX_CNTL, (1 << 2) | pipe)
        gc1_wr(regCP_MEC_RS64_PRGRM_CNTR_START, mec_lo)
        gc1_wr(regCP_MEC_RS64_PRGRM_CNTR_START_HI, mec_hi)
    gc1_wr(regGRBM_GFX_CNTL, 0)
    cntl = gc1_rd(regCP_MEC_RS64_CNTL)
    gc1_wr(regCP_MEC_RS64_CNTL, cntl | 0x000F0010)
    cntl = gc1_rd(regCP_MEC_RS64_CNTL)
    gc1_wr(regCP_MEC_RS64_CNTL, cntl & ~0x000F0010)
    cntl = gc1_rd(regCP_MEC_RS64_CNTL)
    gc1_wr(regCP_MEC_RS64_CNTL, (cntl & ~(0x40000000 | 0x00000010 | 0x000F0000)) | 0x3C000000)
    time.sleep(0.05)
    print(f"\nCP_MEC_RS64_CNTL = 0x{gc1_rd(regCP_MEC_RS64_CNTL):08x}")

    # ==== 4. Enable MES (both pipes per Linux) ====
    uc = h["uc_start_addr"] >> 2
    m_lo = uc & 0xFFFFFFFF
    m_hi = (uc >> 32) & 0xFFFFFFFF
    pre = gc1_rd(regCP_MES_CNTL)
    disable = (pre & ~0x0C000000) | 0x40000000 | 0x00000010 | 0x00030000
    gc1_wr(regCP_MES_CNTL, disable)
    time.sleep(0.01)
    for pipe in range(2):
        gc1_wr(regGRBM_GFX_CNTL, (3 << 2) | pipe)
        cntl = gc1_rd(regCP_MES_CNTL)
        gc1_wr(regCP_MES_CNTL, cntl | (1 << (16 + pipe)))
        gc1_wr(regCP_MES_PRGRM_CNTR_START, m_lo)
        gc1_wr(regCP_MES_PRGRM_CNTR_START_HI, m_hi)
        gc1_wr(regCP_MES_CNTL, 0x04000000 if pipe == 0 else 0x0C000000)
    gc1_wr(regGRBM_GFX_CNTL, 0)
    time.sleep(0.5)
    print(f"CP_MES_CNTL = 0x{gc1_rd(regCP_MES_CNTL):08x}")

    # MES liveness probe — sample INSTR_PNTR and MINSTRET over 100 ms.
    print(f"\n== MES liveness probe ==")
    def _mes_sample(label):
        ip = gc1_rd(regCP_MES_INSTR_PNTR)
        hdr = gc1_rd(regCP_MES_HEADER_DUMP)
        lo = gc1_rd(regCP_MES_MINSTRET_LO)
        hi = gc1_rd(regCP_MES_MINSTRET_HI)
        print(f"  [{label}] INSTR_PNTR=0x{ip:08x} HEADER=0x{hdr:08x} "
              f"MINSTRET={(hi << 32) | lo}")
    _mes_sample("post-enable t=0")
    time.sleep(0.02)
    _mes_sample("t=20ms")
    time.sleep(0.05)
    _mes_sample("t=70ms")

    # ==== 5. Register MES-KIQ with RLC ====
    # ring->me=3, ring->pipe=1 (KIQ_PIPE), ring->queue=0 for uni_mes.
    kiq_me, kiq_pipe, kiq_queue = 3, 1, 0
    sched_lo = (kiq_me << 5) | (kiq_pipe << 3) | kiq_queue
    pre_sched = gc1_rd(regRLC_CP_SCHEDULERS)
    new_sched = (pre_sched & 0xFFFFFF00) | sched_lo | 0x80
    gc1_wr(regRLC_CP_SCHEDULERS, new_sched)
    print(f"\nRLC_CP_SCHEDULERS: 0x{pre_sched:08x} -> 0x{gc1_rd(regRLC_CP_SCHEDULERS):08x}")

    # ==== 6. Build MQD per mes_v12_0_mqd_init ====
    mqd = [0] * (MQD_SIZE // 4)
    mqd[0] = 0xC0310800   # header

    # compute_pipelinestat_enable at DW offset? Let me use same layout
    # as v12_compute_mqd. Actual index requires struct layout — but
    # Linux only writes these fields via direct struct access. For our
    # array-based MQD, we need to find the DW offset for each field.
    # Most are before DW 0x80 (the "HQD image" block). For a KIQ that's
    # never checkpointed, these may not matter. Skip for now.
    # compute_static_thread_mgmt_se0..3 at DW 0x17, 0x18, 0x1A, 0x1B per v12.
    for dw in (0x17, 0x18, 0x1A, 0x1B, 0x2C, 0x2D, 0x2E, 0x2F):
        mqd[dw] = 0xFFFFFFFF

    eop_base_addr_shifted = eop_mc >> 8
    # CP_HQD_EOP_CONTROL: eop_size bits 0-5, default 0x6; set to log2(MES_EOP_SIZE/4) - 1.
    eop_size_enc = ((MES_EOP_SIZE // 4).bit_length() - 2) & 0x3f
    cp_hqd_eop_control = (CP_HQD_EOP_CONTROL_DEFAULT & ~0x3f) | eop_size_enc

    mqd[0xA5] = eop_base_addr_shifted & 0xFFFFFFFF            # cp_hqd_eop_base_addr_lo
    mqd[0xA6] = (eop_base_addr_shifted >> 32) & 0xFFFFFFFF    # cp_hqd_eop_base_addr_hi
    mqd[0xA7] = cp_hqd_eop_control                             # cp_hqd_eop_control

    # CP_MQD_BASE_ADDR: Linux masks & 0xfffffffc (4-byte aligned).
    mqd[0x80] = mqd_mc & 0xFFFFFFFC
    mqd[0x81] = (mqd_mc >> 32) & 0xFFFFFFFF
    # CP_MQD_CONTROL with VMID=0 (default already has VMID=0).
    mqd[0xA2] = CP_MQD_CONTROL_DEFAULT

    # CP_HQD_PQ_BASE: ring_mc >> 8.
    pq_base_shifted = ring_mc >> 8
    mqd[0x88] = pq_base_shifted & 0xFFFFFFFF
    mqd[0x89] = (pq_base_shifted >> 32) & 0xFFFFFFFF

    # CP_HQD_PQ_RPTR_REPORT_ADDR: rptr_mc & 0xfffffffc
    mqd[0x8B] = rptr_mc & 0xFFFFFFFC
    mqd[0x8C] = (rptr_mc >> 32) & 0xFFFF

    # CP_HQD_PQ_WPTR_POLL_ADDR: wptr_mc & 0xfffffff8
    mqd[0x8D] = wptr_mc & 0xFFFFFFF8
    mqd[0x8E] = (wptr_mc >> 32) & 0xFFFF

    # CP_HQD_PQ_CONTROL: queue_size, rptr_block_size, unord_dispatch,
    # tunnel_dispatch=0, priv_state=1, kmd_queue=1, no_update_rptr=1.
    ring_dw = RING_SIZE // 4
    queue_size_val = (ring_dw.bit_length() - 2) & 0x3f
    # AMDGPU_GPU_PAGE_SIZE = 4096, so log2(4096/4) - 1 = 10 - 1 = 9. Linux
    # then << 8 (but that's the SHIFT for RPTR_BLOCK_SIZE in the CNTL reg).
    # The default RPTR_BLOCK_SIZE field is already 5. The formula:
    #   tmp = REG_SET_FIELD(tmp, CP_HQD_PQ_CONTROL, RPTR_BLOCK_SIZE,
    #                       ((order_base_2(4096/4) - 1) << 8))
    # — but REG_SET_FIELD shifts the value into position (bits 8-13). Passing
    # `val << 8` puts bit 11 set in the field, which is wrong (8-13 is mask).
    # Looks like a Linux bug? Or the default's rptr_block_size=5 is kept.
    # Let me just use default (5).
    rptr_block_size = 5
    cp_hqd_pq_control = (
        (queue_size_val & 0x3f) |
        (rptr_block_size << 8) |
        (1 << 0x1b) |   # NO_UPDATE_RPTR
        (1 << 0x1c) |   # UNORD_DISPATCH
        (1 << 0x1e) |   # PRIV_STATE
        (1 << 0x1f)     # KMD_QUEUE
    )
    # Preserve default's MIN_AVAIL_SIZE=3 (bits 20-21) and PQ_EMPTY=1 (bit 15).
    cp_hqd_pq_control |= (0x300000 | 0x8000)
    mqd[0x91] = cp_hqd_pq_control

    # CP_HQD_IB_CONTROL default.
    mqd[0x95] = 0x00300000

    # CP_HQD_PQ_DOORBELL_CONTROL: use_doorbell=False for now (poll mode).
    mqd[0x8F] = 0

    mqd[0x83] = 0                           # cp_hqd_vmid
    mqd[0x82] = 1                           # cp_hqd_active
    mqd[0x84] = (CP_HQD_PERSISTENT_STATE_DEFAULT & ~(0x3ff << 8)) | (0x55 << 8)  # preload_size=0x55
    mqd[0xB5] = 0                            # cp_hqd_aql_control (non-AQL)
    mqd[0xB6] = 0                            # wptr_lo
    mqd[0xB7] = 0                            # wptr_hi
    # reserved_184 = BIT(15) — "DB_UPDATED_MSG_EN" hidden field
    mqd[0xB8] = (1 << 15)

    print(f"\n== MQD key fields ==")
    print(f"  cp_mqd_base_addr_lo        = 0x{mqd[0x80]:08x}")
    print(f"  cp_hqd_pq_base_lo          = 0x{mqd[0x88]:08x}")
    print(f"  cp_hqd_pq_rptr_report_lo   = 0x{mqd[0x8B]:08x}")
    print(f"  cp_hqd_pq_control          = 0x{mqd[0x91]:08x}")
    print(f"  cp_hqd_persistent_state    = 0x{mqd[0x84]:08x}")

    # Write MQD to VRAM.
    for i, v in enumerate(mqd):
        vram_wr(MQD_VRAM_OFF + i * 4, v)

    # ==== 7. Program HQD per mes_v12_0_queue_init_register ====
    print(f"\n== program HQD via me=3 pipe={kiq_pipe} queue=0 (MES-KIQ) ==")
    gc1_wr(regGRBM_GFX_CNTL, (queue_vmid := 0) | (kiq_me << 2) | kiq_pipe)

    # Step 1: CP_HQD_VMID.VMID = 0 (RMW).
    v = gc1_rd(regCP_HQD_VMID)
    gc1_wr(regCP_HQD_VMID, v & ~0xf)   # VMID bits 0-3
    # Step 2: DOORBELL_EN = 0 (RMW).
    v = gc1_rd(regCP_HQD_PQ_DOORBELL_CONTROL)
    gc1_wr(regCP_HQD_PQ_DOORBELL_CONTROL, v & ~0x40000000)  # DOORBELL_EN bit 30
    # Step 3: CP_MQD_BASE_ADDR / HI
    gc1_wr(regCP_MQD_BASE_ADDR, mqd[0x80])
    gc1_wr(regCP_MQD_BASE_ADDR_HI, mqd[0x81])
    # Step 4: CP_MQD_CONTROL = 0 (not `data`!)
    gc1_wr(regCP_MQD_CONTROL, 0)
    # Step 5: CP_HQD_PQ_BASE / HI
    gc1_wr(regCP_HQD_PQ_BASE, mqd[0x88])
    gc1_wr(regCP_HQD_PQ_BASE_HI, mqd[0x89])
    # Step 6: CP_HQD_PQ_RPTR_REPORT_ADDR / HI
    gc1_wr(regCP_HQD_PQ_RPTR_REPORT_ADDR, mqd[0x8B])
    gc1_wr(regCP_HQD_PQ_RPTR_REPORT_ADDR_HI, mqd[0x8C])
    # Step 7: CP_HQD_PQ_CONTROL
    gc1_wr(regCP_HQD_PQ_CONTROL, mqd[0x91])
    # Step 8: CP_HQD_PQ_WPTR_POLL_ADDR / HI
    gc1_wr(regCP_HQD_PQ_WPTR_POLL_ADDR, mqd[0x8D])
    gc1_wr(regCP_HQD_PQ_WPTR_POLL_ADDR_HI, mqd[0x8E])
    # Step 9: CP_HQD_PQ_DOORBELL_CONTROL (final)
    gc1_wr(regCP_HQD_PQ_DOORBELL_CONTROL, mqd[0x8F])
    # Step 10: CP_HQD_PERSISTENT_STATE
    gc1_wr(regCP_HQD_PERSISTENT_STATE, mqd[0x84])
    # Step 11: CP_HQD_ACTIVE = 1
    gc1_wr(regCP_HQD_ACTIVE, mqd[0x82])
    time.sleep(0.01)

    print(f"  GRBM_STATUS (pre submit) = 0x{gc0_rd(regGRBM_STATUS):08x}")
    print(f"  CP_STAT     (pre submit) = 0x{gc0_rd(regCP_STAT):08x}")

    # ==== 8. Submit a PM4 NOP packet ====
    # PM4 type-3: header = 0xC0001000 (type=3, count=0 meaning 1 DW total, opcode=0x10=NOP).
    # Actually count=0 means 2 DW packet (1 header + 1 body DW). For a pure NOP,
    # we want count=0 (minimum) with the body being any DW.
    # Let me just use 0xC0001000 + one zero body DW = 2 DW total.
    pm4_nop = 0xC0001000
    vram_wr(RING_VRAM_OFF, pm4_nop)
    vram_wr(RING_VRAM_OFF + 4, 0)

    # Advance wptr by 2 DWORDs.
    gc1_wr(regCP_HQD_PQ_WPTR_LO, 2)

    # ==== 9. Watch for side effects ====
    print(f"\n== poll for queue activity (5s) ==")
    deadline = time.time() + 5
    last = None
    while time.time() < deadline:
        rptr_report = vram_rd(RPTR_VRAM_OFF)
        rptr_reg = gc1_rd(regCP_HQD_PQ_RPTR)
        wptr_reg = gc1_rd(regCP_HQD_PQ_WPTR_LO)
        active = gc1_rd(regCP_HQD_ACTIVE)
        cp_stat = gc0_rd(regCP_STAT)
        grbm = gc0_rd(regGRBM_STATUS)
        snap = (rptr_report, rptr_reg, wptr_reg, active, cp_stat, grbm)
        if snap != last:
            t = time.time() - deadline + 5
            print(f"  t={t:5.2f}s rptr_report=0x{rptr_report:08x} rptr_reg=0x{rptr_reg:x} "
                  f"wptr_reg=0x{wptr_reg:x} active=0x{active:x} CP_STAT=0x{cp_stat:08x} "
                  f"GRBM=0x{grbm:08x}")
            last = snap
        if rptr_report > 0:
            print(f"  QUEUE PROCESSING ✓")
            break
        time.sleep(0.05)

    gc1_wr(regGRBM_GFX_CNTL, 0)


if __name__ == "__main__":
    main()
