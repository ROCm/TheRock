"""Phase 9c (attempt 8): full doorbell plumbing for MES-KIQ activation.

Prior run (try_phase9_mes_kiq.py): MES is alive (HEADER_DUMP
incrementing) but HQD not activated and no rptr advance. The blocker
was programming the HQD/MQD register block through GC BASE_IDX=1; Linux's
generated offsets put CP_MQD_BASE_ADDR..CP_HQD_PQ_WPTR_HI at BASE_IDX=0.

Linux uses doorbell for MES/compute rings unconditionally. Our
wptr-via-register path might not trigger MES's queue scan. Adding:

  1. `regRCC_DEV0_EPF2_STRAP2.strap_no_soft_reset_dev0_f2 = 0`
  2. `regRCC_DEV0_EPF0_RCC_DOORBELL_APER_EN = 1`
  3. `regGDC_S2A0_S2A_DOORBELL_ENTRY_0_CTRL` — port 0 config
     (enable=1, awid=3, awaddr_31_28=3)
  4. `regGDC_S2A0_S2A_DOORBELL_ENTRY_3_CTRL` — port 3 config
     (enable=1, awid=6, awaddr_31_28=3)
  5. `regCP_MEC_DOORBELL_RANGE_LOWER/UPPER` — CP accepts doorbells
     in [0, 0x450].
  6. MQD `cp_hqd_pq_doorbell_control` with DOORBELL_EN=1,
     DOORBELL_OFFSET = MES_KIQ ring index * 2 = 0xC * 2 = 0x18.
  7. After HQD program, write new wptr to BAR2 + 0x18 * 4 = BAR2+0x60.

Set PHASE9_ATTACH_ONLY=1 when MEC/MES are already alive and only the
MES-KIQ HQD/doorbell path should be reprogrammed. This avoids replaying
PSP firmware-load commands against a partially initialized GPU.

Register locations:
  nbio_6_3_1 BASE_IDX=2 (NBIO base 0xD20):
    regRCC_DEV0_EPF0_RCC_DOORBELL_APER_EN    = 0x00c0
    regGDC_S2A0_S2A_DOORBELL_ENTRY_0_CTRL    = 0x01cb
    regGDC_S2A0_S2A_DOORBELL_ENTRY_3_CTRL    = 0x01ce
  nbio_6_3_1 BASE_IDX=5 (=0x4040000):
    regRCC_DEV0_EPF2_STRAP2                  = 0xd102

S2A_DOORBELL_ENTRY_N_CTRL bitfields:
  ENABLE      bit 0
  AWID        bits 1-5
  FENCE_EN    bit 6
  RANGE_OFFSET bits 7-16
  RANGE_SIZE  bits 17-24
  AWADDR_31_28 bits 28-31
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
NBIO_B2 = 0xD20       # NBIO BASE_IDX=2
NBIO_B5 = 0x4040000   # NBIO BASE_IDX=5

GFX_FW_TYPE_CP_MES           = 33
GFX_FW_TYPE_MES_STACK        = 34
GFX_FW_TYPE_CP_MES_KIQ       = 81
GFX_FW_TYPE_MES_KIQ_STACK    = 82

# NBIO regs
regRCC_DEV0_EPF0_RCC_DOORBELL_APER_EN = 0x00c0   # BASE_IDX=2
regGDC_S2A0_S2A_DOORBELL_ENTRY_0_CTRL = 0x01cb   # BASE_IDX=2
regGDC_S2A0_S2A_DOORBELL_ENTRY_3_CTRL = 0x01ce   # BASE_IDX=2
regRCC_DEV0_EPF2_STRAP2               = 0xd102   # BASE_IDX=5

# GC regs (BASE_IDX=1)
regGRBM_GFX_CNTL                 = 0x0900
regRLC_CP_SCHEDULERS             = 0x098a
regCP_MEC_RS64_PRGRM_CNTR_START  = 0x2900
regCP_MEC_RS64_PRGRM_CNTR_START_HI = 0x2938
regCP_MEC_RS64_CNTL              = 0x2904
regCP_MEC_RS64_INSTR_PNTR        = 0x2908
regCP_MES_PRGRM_CNTR_START       = 0x2800
regCP_MES_PRGRM_CNTR_START_HI    = 0x289d
regCP_MES_CNTL                   = 0x2807
regCP_MES_HEADER_DUMP            = 0x280d
regCP_MES_INSTR_PNTR             = 0x2813
# GC regs (BASE_IDX=0)
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
regCP_HQD_DEQUEUE_REQUEST        = 0x1fc1
regCP_HQD_PQ_WPTR_LO             = 0x1fdf
regCP_HQD_PQ_WPTR_HI             = 0x1fe0
regCP_MQD_CONTROL                = 0x1fcb
regCP_HQD_EOP_BASE_ADDR          = 0x1fce
regCP_HQD_EOP_BASE_ADDR_HI       = 0x1fcf
regCP_HQD_EOP_CONTROL            = 0x1fd0
regGRBM_STATUS                   = 0x0da4
regCP_STAT                       = 0x0f40
regCP_MEC_DOORBELL_RANGE_LOWER   = 0x1dfc
regCP_MEC_DOORBELL_RANGE_UPPER   = 0x1dfd

CP_HQD_PQ_CONTROL_DEFAULT        = 0x00308509
CP_HQD_PERSISTENT_STATE_DEFAULT  = 0x0be05501

MQD_VRAM_OFF   = 0x1800000
RING_VRAM_OFF  = 0x1802000
EOP_VRAM_OFF   = 0x1810000
RPTR_VRAM_OFF  = 0x1820000
WPTR_VRAM_OFF  = 0x1821000
FENCE_VRAM_OFF = 0x1830000
SCH_CTX_VRAM_OFF = 0x1831000
QUERY_STATUS_VRAM_OFF = 0x1832000
RESOURCE1_VRAM_OFF = 0x1840000
SCHED_MQD_VRAM_OFF = 0x1850000
SCHED_RING_VRAM_OFF = 0x1852000
SCHED_EOP_VRAM_OFF = 0x1860000
SCHED_RPTR_VRAM_OFF = 0x1870000
SCHED_WPTR_VRAM_OFF = 0x1871000
SCHED_SCH_CTX_VRAM_OFF = 0x1880000
SCHED_QUERY_STATUS_VRAM_OFF = 0x1881000
SCHED_RESOURCE1_VRAM_OFF = 0x1890000

MQD_SIZE  = 0x1000
RING_SIZE = 0x1000
MES_EOP_SIZE = 0x1000

AMDGPU_NAVI10_DOORBELL_MES_RING0 = 0x00B
API_FRAME_SIZE_IN_DWORDS = 64
MES_API_TYPE_SCHEDULER = 1
MES_SCH_API_SET_HW_RSRC = 0
MES_SCH_API_ADD_QUEUE = 2
MES_SCH_API_QUERY_SCHEDULER_STATUS = 11
MES_SCH_API_SET_HW_RSRC_1 = 19
MES_QUEUE_TYPE_SCHQ = 3
GC_BASES = (0x1260, 0xA000, 0x1C000, 0x2402C00, 0, 0, 0, 0)
MMHUB_BASES = (0x1A000, 0x2408800, 0, 0, 0, 0, 0, 0)
OSSSYS_BASES = (0x10A0, 0x240A000, 0, 0, 0, 0, 0, 0)
MES_SCHQ_VMID_MASK = 0xFF00
MES_COMPUTE_HQD_MASKS = (0x0C, 0x0C, 0, 0, 0, 0, 0, 0)
MES_GFX_HQD_MASKS = (0xFE, 0)
MES_SDMA_HQD_MASKS = (0xFC, 0xFC)
MES_AGGREGATED_DOORBELLS = (0x800, 0x802, 0x804, 0x806, 0x808)

# union MESAPI_SET_HW_RESOURCES dword positions under #pragma pack(push, 8).
SET_HW_DW_VMID_MASK_MMHUB = 1
SET_HW_DW_VMID_MASK_GFXHUB = 2
SET_HW_DW_COMPUTE_HQD_MASK = 5
SET_HW_DW_GFX_HQD_MASK = 13
SET_HW_DW_SDMA_HQD_MASK = 15
SET_HW_DW_AGGREGATED_DOORBELLS = 17
SET_HW_DW_G_SCH_CTX = 22
SET_HW_DW_QUERY_STATUS = 24
SET_HW_DW_GC_BASE = 26
SET_HW_DW_MMHUB_BASE = 34
SET_HW_DW_OSSSYS_BASE = 42
SET_HW_DW_API_STATUS = 50
SET_HW_DW_FLAGS = 54
SET_HW_DW_OVERSUBSCRIPTION_TIMER = 55

# union MESAPI_SET_HW_RESOURCES_1 dword positions.
SET_HW1_DW_API_STATUS = 2
SET_HW1_DW_MES_KIQ_UNMAP_TIMEOUT = 13
SET_HW1_DW_CLEANER_SHADER_FENCE = 16

# union MESAPI__ADD_QUEUE dword positions.
ADD_QUEUE_DW_DOORBELL_OFFSET = 18
ADD_QUEUE_DW_MQD_ADDR = 20
ADD_QUEUE_DW_WPTR_ADDR = 22
ADD_QUEUE_DW_QUEUE_TYPE = 28
ADD_QUEUE_DW_FLAGS = 37
ADD_QUEUE_DW_API_STATUS = 38
ADD_QUEUE_DW_PIPE_ID = 50
ADD_QUEUE_DW_QUEUE_ID = 51


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
    drv = _DriverShim(c)
    info = c.get_info()
    print(f"device=0x{info.device_id:04x} rev=0x{info.revision_id:02x}")
    attach_only = os.environ.get("PHASE9_ATTACH_ONLY") == "1"

    def rd(base, off): return c.mmio_read32(_MMIO_BAR, (base + off) * 4)
    def wr(base, off, v): c.mmio_write32(_MMIO_BAR, (base + off) * 4, v & 0xFFFFFFFF)
    def gc1_rd(o): return rd(GC_B1, o)
    def gc1_wr(o, v): wr(GC_B1, o, v)
    def gc0_rd(o): return rd(GC_B0, o)
    def gc0_wr(o, v): wr(GC_B0, o, v)
    def hqd_rd(o): return gc0_rd(o)
    def hqd_wr(o, v): gc0_wr(o, v)

    # Bring up GFX unless attaching to an already live MEC/MES instance.
    if attach_only:
        print("PHASE9_ATTACH_ONLY=1: skipping gfx_bring_up() and PSP firmware loads")
    else:
        r = gfx_bring_up(c, drv, firmware_dir=FIRMWARE_DIR)
        if not (r.bootload_status & 0x80000000):
            print("BOOTLOAD_COMPLETE not set — aborting.")
            sys.exit(1)

    fb_base = (c.mmio_read32(_MMIO_BAR, (0x1A000 + 0x0554) * 4) & 0xFFFFFF) << 24
    bar0_cpu, _ = c.map_bar(0)

    # Map BAR2 (doorbell aperture).
    try:
        bar2_cpu, bar2_size = c.map_bar(2)
        print(f"BAR2 (doorbell): addr=0x{bar2_cpu:x} size={bar2_size // 1024}KB")
    except Exception as e:
        print(f"BAR2 map failed: {e}")
        sys.exit(1)

    def bar2_wr64(byte_off, val):
        (ctypes.c_uint64 * 1).from_address(bar2_cpu + byte_off)[0] = val

    mqd_mc  = fb_base + MQD_VRAM_OFF
    ring_mc = fb_base + RING_VRAM_OFF
    eop_mc  = fb_base + EOP_VRAM_OFF
    rptr_mc = fb_base + RPTR_VRAM_OFF
    wptr_mc = fb_base + WPTR_VRAM_OFF
    sched_mqd_mc = fb_base + SCHED_MQD_VRAM_OFF
    sched_ring_mc = fb_base + SCHED_RING_VRAM_OFF
    sched_eop_mc = fb_base + SCHED_EOP_VRAM_OFF
    sched_rptr_mc = fb_base + SCHED_RPTR_VRAM_OFF
    sched_wptr_mc = fb_base + SCHED_WPTR_VRAM_OFF

    def vram_wr(off, val):
        (ctypes.c_uint32 * 1).from_address(bar0_cpu + off)[0] = val & 0xFFFFFFFF

    def vram_rd(off):
        return (ctypes.c_uint32 * 1).from_address(bar0_cpu + off)[0]

    def vram_rd64(off):
        return (ctypes.c_uint64 * 1).from_address(bar0_cpu + off)[0]

    def vram_wr64(off, val):
        (ctypes.c_uint64 * 1).from_address(bar0_cpu + off)[0] = val & 0xFFFFFFFFFFFFFFFF

    def select_hqd(me, pipe, queue=0, vmid=0):
        gc1_wr(regGRBM_GFX_CNTL,
               ((pipe & 0x3) << 0) | ((me & 0x3) << 2) |
               ((vmid & 0xf) << 4) | ((queue & 0x7) << 8))

    def mes_header(opcode):
        return (
            MES_API_TYPE_SCHEDULER |
            (opcode << 4) |
            (API_FRAME_SIZE_IN_DWORDS << 12)
        )

    def put64(pkt, dw, value):
        pkt[dw] = value & 0xFFFFFFFF
        pkt[dw + 1] = (value >> 32) & 0xFFFFFFFF

    def submit_api_and_poll(label, ring_off, wptr_off, doorbell_idx,
                            hqd_select, start_wptr, pkt, fence_value,
                            timeout=5):
        end_wptr = start_wptr + len(pkt)
        ring_dw = RING_SIZE // 4
        vram_wr64(FENCE_VRAM_OFF, 0)

        print(f"\n== {label} at ring DW 0x{start_wptr:x}, end=0x{end_wptr:x}, "
              f"status_fence=0x{fb_base + FENCE_VRAM_OFF:x} ==")
        for i, word in enumerate(pkt):
            vram_wr(ring_off + (((start_wptr + i) % ring_dw) * 4), word)

        vram_wr64(wptr_off, end_wptr)
        bar2_wr64(doorbell_idx * 4, end_wptr)

        deadline = time.time() + timeout
        last = None
        while time.time() < deadline:
            fence = vram_rd64(FENCE_VRAM_OFF)
            select_hqd(*hqd_select)
            rptr_reg = hqd_rd(regCP_HQD_PQ_RPTR)
            wptr_reg = hqd_rd(regCP_HQD_PQ_WPTR_LO)
            active = hqd_rd(regCP_HQD_ACTIVE)
            snap = (fence, rptr_reg, wptr_reg, active)
            if snap != last:
                t = time.time() - deadline + timeout
                print(f"  t={t:5.2f}s fence=0x{fence:016x} rptr_reg=0x{rptr_reg:x} "
                      f"wptr_reg=0x{wptr_reg:x} active=0x{active:x}")
                last = snap
            if fence == fence_value:
                print(f"  {label} fence signaled ✓")
                return True, end_wptr
            time.sleep(0.05)

        print(f"  {label} did not signal status fence before timeout")
        return False, end_wptr

    def build_sched_mqd():
        sched_doorbell = AMDGPU_NAVI10_DOORBELL_MES_RING0 << 1
        sched_mqd = [0] * (MQD_SIZE // 4)
        sched_mqd[0] = 0xC0310800
        sched_mqd[1] = 1
        for dw in (0x17, 0x18, 0x1A, 0x1B):
            sched_mqd[dw] = 0xFFFFFFFF
        sched_mqd[0x2C] = 7

        eop_base_addr_shifted = sched_eop_mc >> 8
        sched_mqd[0xA5] = eop_base_addr_shifted & 0xFFFFFFFF
        sched_mqd[0xA6] = (eop_base_addr_shifted >> 32) & 0xFFFFFFFF
        eop_size_enc = ((MES_EOP_SIZE // 4).bit_length() - 2) & 0x3f
        sched_mqd[0xA7] = (0x6 & ~0x3f) | eop_size_enc

        sched_mqd[0x80] = sched_mqd_mc & 0xFFFFFFFC
        sched_mqd[0x81] = (sched_mqd_mc >> 32) & 0xFFFFFFFF
        sched_mqd[0xA2] = 0x100

        pq_base_shifted = sched_ring_mc >> 8
        sched_mqd[0x88] = pq_base_shifted & 0xFFFFFFFF
        sched_mqd[0x89] = (pq_base_shifted >> 32) & 0xFFFFFFFF
        sched_mqd[0x8B] = sched_rptr_mc & 0xFFFFFFFC
        sched_mqd[0x8C] = (sched_rptr_mc >> 32) & 0xFFFF
        sched_mqd[0x8D] = sched_wptr_mc & 0xFFFFFFF8
        sched_mqd[0x8E] = (sched_wptr_mc >> 32) & 0xFFFF

        ring_dw = RING_SIZE // 4
        queue_size_val = (ring_dw.bit_length() - 2) & 0x3f
        sched_mqd[0x91] = (
            queue_size_val |
            (5 << 8) |
            (1 << 0x1b) |
            (1 << 0x1c) |
            (1 << 0x1e) |
            (1 << 0x1f) |
            0x300000 |
            0x8000
        )
        sched_mqd[0x8F] = ((sched_doorbell & 0x3FFFFFF) << 2) | (1 << 30)
        sched_mqd[0x95] = 0x00300000
        sched_mqd[0x82] = 1
        sched_mqd[0x84] = (CP_HQD_PERSISTENT_STATE_DEFAULT & ~(0x3ff << 8)) | (0x55 << 8)
        sched_mqd[0xB8] = (1 << 15)
        return sched_mqd

    # Zero VRAM regions.
    for base, size in [(MQD_VRAM_OFF, MQD_SIZE), (RING_VRAM_OFF, RING_SIZE),
                       (EOP_VRAM_OFF, 0x1000), (RPTR_VRAM_OFF, 0x20),
                       (WPTR_VRAM_OFF, 0x20), (FENCE_VRAM_OFF, 0x20),
                       (SCH_CTX_VRAM_OFF, 0x1000), (QUERY_STATUS_VRAM_OFF, 0x1000),
                       (RESOURCE1_VRAM_OFF, 0x1000),
                       (SCHED_MQD_VRAM_OFF, MQD_SIZE), (SCHED_RING_VRAM_OFF, RING_SIZE),
                       (SCHED_EOP_VRAM_OFF, 0x1000), (SCHED_RPTR_VRAM_OFF, 0x20),
                       (SCHED_WPTR_VRAM_OFF, 0x20), (SCHED_SCH_CTX_VRAM_OFF, 0x1000),
                       (SCHED_QUERY_STATUS_VRAM_OFF, 0x1000),
                       (SCHED_RESOURCE1_VRAM_OFF, 0x1000)]:
        for i in range(0, size, 4):
            vram_wr(base + i, 0)

    # ==== NBIO: doorbell aperture enable ====
    print(f"\n== NBIO doorbell aperture ==")
    # STRAP2 is at NBIF BASE_IDX=5, which is outside the BAR5 range accepted
    # by the current DEXT MMIO escape. Linux clears bit 7 here, but it is a
    # persisting-state quirk fix, not required for the doorbell aperture test.
    if os.environ.get("TOUCH_NBIO_STRAP2") == "1":
        pre_strap = rd(NBIO_B5, regRCC_DEV0_EPF2_STRAP2)
        wr(NBIO_B5, regRCC_DEV0_EPF2_STRAP2, pre_strap & ~0x80)
        post_strap = rd(NBIO_B5, regRCC_DEV0_EPF2_STRAP2)
        print(f"  RCC_DEV0_EPF2_STRAP2: pre=0x{pre_strap:08x} post=0x{post_strap:08x}")
    else:
        print("  RCC_DEV0_EPF2_STRAP2 skipped (set TOUCH_NBIO_STRAP2=1 to try it)")

    pre_aper = rd(NBIO_B2, regRCC_DEV0_EPF0_RCC_DOORBELL_APER_EN)
    wr(NBIO_B2, regRCC_DEV0_EPF0_RCC_DOORBELL_APER_EN, 1)
    post_aper = rd(NBIO_B2, regRCC_DEV0_EPF0_RCC_DOORBELL_APER_EN)
    print(f"  APER_EN: pre=0x{pre_aper:08x} post=0x{post_aper:08x}")

    # S2A_DOORBELL_ENTRY_0_CTRL bit layout:
    #   ENABLE(0) | AWID(1:5) | FENCE_EN(6) | RANGE_OFFSET(7:16)
    #   | RANGE_SIZE(17:24) | 64BIT_DIS(25) | NEED_DEDUCT(26) | DROP_EN(27)
    #   | AWADDR_31_28(28:31)
    # Port 0: enable=1 awid=3 awaddr_31_28=3 -> 0x30000007
    port0_val = (1 << 0) | (3 << 1) | (3 << 28)
    wr(NBIO_B2, regGDC_S2A0_S2A_DOORBELL_ENTRY_0_CTRL, port0_val)
    print(f"  S2A_DOORBELL_ENTRY_0_CTRL = 0x{port0_val:08x}")

    # Port 3: enable=1 awid=6 awaddr_31_28=3 -> 0x3000000D
    port3_val = (1 << 0) | (6 << 1) | (3 << 28)
    wr(NBIO_B2, regGDC_S2A0_S2A_DOORBELL_ENTRY_3_CTRL, port3_val)
    print(f"  S2A_DOORBELL_ENTRY_3_CTRL = 0x{port3_val:08x}")

    # CP_MEC_DOORBELL_RANGE — covers KIQ doorbell 0..USERQUEUE_END.
    gc0_wr(regCP_MEC_DOORBELL_RANGE_LOWER, 0x0)
    gc0_wr(regCP_MEC_DOORBELL_RANGE_UPPER, (0x8A * 2) << 2)
    print(f"  CP_MEC_DOORBELL_RANGE [LOWER=0, UPPER=0x{(0x8A * 2) << 2:x}]")

    # ==== Load MES + enable MEC + enable MES ====
    if not attach_only:
        uni_blob = open(os.path.join(FIRMWARE_DIR, "gc_12_0_1_uni_mes.bin"), "rb").read()
        h = _parse_mes(uni_blob)
        ucode = uni_blob[h["ucode_offset"]:h["ucode_offset"] + h["ucode_size"]]
        data  = uni_blob[h["data_offset"]:h["data_offset"] + h["data_size"]]

        ctx = alloc_cmd_ctx(drv)
        for label, fw_type, payload in [
            ("CP_MES", GFX_FW_TYPE_CP_MES, ucode),
            ("MES_STACK", GFX_FW_TYPE_MES_STACK, data),
            ("CP_MES_KIQ", GFX_FW_TYPE_CP_MES_KIQ, ucode),
            ("MES_KIQ_STACK", GFX_FW_TYPE_MES_KIQ_STACK, data),
        ]:
            _load_one(c, drv, MP0_BASE_DW, r.ring, ctx, payload, fw_type, label, strict=True)

        mec_blob = open(os.path.join(FIRMWARE_DIR, "gc_12_0_1_mec.bin"), "rb").read()
        us = _parse_rs64(mec_blob)
        for pipe in range(4):
            gc1_wr(regGRBM_GFX_CNTL, (1 << 2) | pipe)
            gc1_wr(regCP_MEC_RS64_PRGRM_CNTR_START, (us >> 2) & 0xFFFFFFFF)
            gc1_wr(regCP_MEC_RS64_PRGRM_CNTR_START_HI, (us >> 34) & 0xFFFFFFFF)
        gc1_wr(regGRBM_GFX_CNTL, 0)
        cntl = gc1_rd(regCP_MEC_RS64_CNTL)
        gc1_wr(regCP_MEC_RS64_CNTL, cntl | 0x000F0010)
        cntl = gc1_rd(regCP_MEC_RS64_CNTL)
        gc1_wr(regCP_MEC_RS64_CNTL, cntl & ~0x000F0010)
        cntl = gc1_rd(regCP_MEC_RS64_CNTL)
        gc1_wr(regCP_MEC_RS64_CNTL,
               (cntl & ~(0x40000000 | 0x00000010 | 0x000F0000)) | 0x3C000000)
        time.sleep(0.05)

        uc = h["uc_start_addr"] >> 2
        pre = gc1_rd(regCP_MES_CNTL)
        gc1_wr(regCP_MES_CNTL, (pre & ~0x0C000000) | 0x40000000 | 0x00000010 | 0x00030000)
        time.sleep(0.01)
        for pipe in range(2):
            gc1_wr(regGRBM_GFX_CNTL, (3 << 2) | pipe)
            cntl = gc1_rd(regCP_MES_CNTL)
            gc1_wr(regCP_MES_CNTL, cntl | (1 << (16 + pipe)))
            gc1_wr(regCP_MES_PRGRM_CNTR_START, uc & 0xFFFFFFFF)
            gc1_wr(regCP_MES_PRGRM_CNTR_START_HI, (uc >> 32) & 0xFFFFFFFF)
            gc1_wr(regCP_MES_CNTL, 0x04000000 if pipe == 0 else 0x0C000000)
        gc1_wr(regGRBM_GFX_CNTL, 0)
        time.sleep(0.5)

    print(f"\nMEC/MES alive check: CP_MEC_INSTR=0x{gc1_rd(regCP_MEC_RS64_INSTR_PNTR):x} "
          f"MES_INSTR=0x{gc1_rd(regCP_MES_INSTR_PNTR):x} MES_HEADER=0x{gc1_rd(regCP_MES_HEADER_DUMP):x}")

    # ==== Register MES-KIQ with RLC ====
    kiq_me, kiq_pipe, kiq_queue = 3, 1, 0
    sched_lo = (kiq_me << 5) | (kiq_pipe << 3) | kiq_queue
    pre_sched = gc1_rd(regRLC_CP_SCHEDULERS)
    gc1_wr(regRLC_CP_SCHEDULERS, (pre_sched & 0xFFFFFF00) | sched_lo | 0x80)
    print(f"RLC_CP_SCHEDULERS: 0x{pre_sched:08x} -> 0x{gc1_rd(regRLC_CP_SCHEDULERS):08x}")

    # ==== Build MQD ====
    mes_kiq_doorbell = (AMDGPU_NAVI10_DOORBELL_MES_RING0 + 1) << 1  # 0x18
    mqd = [0] * (MQD_SIZE // 4)
    mqd[0] = 0xC0310800
    for dw in (0x17, 0x18, 0x1A, 0x1B, 0x2C, 0x2D, 0x2E, 0x2F):
        mqd[dw] = 0xFFFFFFFF

    eop_base_addr_shifted = eop_mc >> 8
    mqd[0xA5] = eop_base_addr_shifted & 0xFFFFFFFF
    mqd[0xA6] = (eop_base_addr_shifted >> 32) & 0xFFFFFFFF
    eop_size_enc = ((MES_EOP_SIZE // 4).bit_length() - 2) & 0x3f
    mqd[0xA7] = (0x6 & ~0x3f) | eop_size_enc   # default 0x6 field 0

    mqd[0x80] = mqd_mc & 0xFFFFFFFC
    mqd[0x81] = (mqd_mc >> 32) & 0xFFFFFFFF
    mqd[0xA2] = 0x100   # CP_MQD_CONTROL default (VMID=0)

    pq_base_shifted = ring_mc >> 8
    mqd[0x88] = pq_base_shifted & 0xFFFFFFFF
    mqd[0x89] = (pq_base_shifted >> 32) & 0xFFFFFFFF
    mqd[0x8B] = rptr_mc & 0xFFFFFFFC
    mqd[0x8C] = (rptr_mc >> 32) & 0xFFFF
    mqd[0x8D] = wptr_mc & 0xFFFFFFF8
    mqd[0x8E] = (wptr_mc >> 32) & 0xFFFF

    ring_dw = RING_SIZE // 4
    queue_size_val = (ring_dw.bit_length() - 2) & 0x3f
    mqd[0x91] = (
        queue_size_val |
        (5 << 8) |          # RPTR_BLOCK_SIZE
        (1 << 0x1b) |       # NO_UPDATE_RPTR
        (1 << 0x1c) |       # UNORD_DISPATCH
        (1 << 0x1e) |       # PRIV_STATE
        (1 << 0x1f) |       # KMD_QUEUE
        0x300000 |          # MIN_AVAIL_SIZE
        0x8000              # PQ_EMPTY
    )

    mqd[0x95] = 0x00300000  # CP_HQD_IB_CONTROL default

    # DOORBELL_CONTROL: DOORBELL_OFFSET bits 2-27, DOORBELL_EN bit 30
    mqd[0x8F] = ((mes_kiq_doorbell & 0x3FFFFFF) << 2) | (1 << 30)

    mqd[0x83] = 0
    mqd[0x82] = 1
    mqd[0x84] = (CP_HQD_PERSISTENT_STATE_DEFAULT & ~(0x3ff << 8)) | (0x55 << 8)
    mqd[0xB5] = 0
    mqd[0xB6] = 0
    mqd[0xB7] = 0
    mqd[0xB8] = (1 << 15)   # reserved_184 = DB_UPDATED_MSG_EN

    print(f"\nMQD key fields:")
    print(f"  cp_hqd_pq_doorbell_control = 0x{mqd[0x8F]:08x} (offset=0x{mes_kiq_doorbell:x})")
    print(f"  cp_hqd_pq_control          = 0x{mqd[0x91]:08x}")

    for i, v in enumerate(mqd):
        vram_wr(MQD_VRAM_OFF + i * 4, v)

    # ==== Program HQD via me=3, pipe=1 ====
    gc1_wr(regGRBM_GFX_CNTL, (kiq_me << 2) | kiq_pipe)
    pre_active = hqd_rd(regCP_HQD_ACTIVE)
    if pre_active and os.environ.get("PHASE9_DEQUEUE_EXISTING") == "1":
        hqd_wr(regCP_HQD_DEQUEUE_REQUEST, 1)
        deadline = time.time() + 1
        while time.time() < deadline and hqd_rd(regCP_HQD_ACTIVE):
            time.sleep(0.001)
        if hqd_rd(regCP_HQD_ACTIVE):
            hqd_wr(regCP_HQD_ACTIVE, 0)
            time.sleep(0.01)
        hqd_wr(regCP_HQD_DEQUEUE_REQUEST, 0)
    post_dequeue_active = hqd_rd(regCP_HQD_ACTIVE)
    hqd_wr(regCP_HQD_PQ_RPTR, 0)
    hqd_wr(regCP_HQD_PQ_WPTR_LO, 0)
    hqd_wr(regCP_HQD_PQ_WPTR_HI, 0)
    v = hqd_rd(regCP_HQD_VMID)
    hqd_wr(regCP_HQD_VMID, v & ~0xf)
    v = hqd_rd(regCP_HQD_PQ_DOORBELL_CONTROL)
    hqd_wr(regCP_HQD_PQ_DOORBELL_CONTROL, v & ~0x40000000)
    hqd_wr(regCP_MQD_BASE_ADDR, mqd[0x80])
    hqd_wr(regCP_MQD_BASE_ADDR_HI, mqd[0x81])
    hqd_wr(regCP_MQD_CONTROL, 0)
    hqd_wr(regCP_HQD_EOP_BASE_ADDR, mqd[0xA5])
    hqd_wr(regCP_HQD_EOP_BASE_ADDR_HI, mqd[0xA6])
    hqd_wr(regCP_HQD_EOP_CONTROL, mqd[0xA7])
    hqd_wr(regCP_HQD_PQ_BASE, mqd[0x88])
    hqd_wr(regCP_HQD_PQ_BASE_HI, mqd[0x89])
    hqd_wr(regCP_HQD_PQ_RPTR_REPORT_ADDR, mqd[0x8B])
    hqd_wr(regCP_HQD_PQ_RPTR_REPORT_ADDR_HI, mqd[0x8C])
    hqd_wr(regCP_HQD_PQ_CONTROL, mqd[0x91])
    hqd_wr(regCP_HQD_PQ_WPTR_POLL_ADDR, mqd[0x8D])
    hqd_wr(regCP_HQD_PQ_WPTR_POLL_ADDR_HI, mqd[0x8E])
    hqd_wr(regCP_HQD_PQ_DOORBELL_CONTROL, mqd[0x8F])
    hqd_wr(regCP_HQD_PERSISTENT_STATE, mqd[0x84])
    hqd_wr(regCP_HQD_ACTIVE, mqd[0x82])
    time.sleep(0.01)

    print(f"\nHQD readback after program:")
    print(f"  PRE_ACTIVE=0x{pre_active:08x} POST_DEQUEUE_ACTIVE=0x{post_dequeue_active:08x} "
          f"ACTIVE=0x{hqd_rd(regCP_HQD_ACTIVE):08x} "
          f"VMID=0x{hqd_rd(regCP_HQD_VMID):08x} "
          f"MQD=0x{hqd_rd(regCP_MQD_BASE_ADDR_HI):08x}:{hqd_rd(regCP_MQD_BASE_ADDR):08x}")
    print(f"  PQ_BASE=0x{hqd_rd(regCP_HQD_PQ_BASE_HI):08x}:{hqd_rd(regCP_HQD_PQ_BASE):08x} "
          f"PQ_CONTROL=0x{hqd_rd(regCP_HQD_PQ_CONTROL):08x} "
          f"DOORBELL_CONTROL=0x{hqd_rd(regCP_HQD_PQ_DOORBELL_CONTROL):08x}")
    print(f"  RPTR=0x{hqd_rd(regCP_HQD_PQ_RPTR):08x} "
          f"WPTR=0x{hqd_rd(regCP_HQD_PQ_WPTR_LO):08x}:{hqd_rd(regCP_HQD_PQ_WPTR_HI):08x}")

    # BAR2 is 4 byte aligned doorbells; MES KIQ doorbell slot = mes_kiq_doorbell
    # dwords from BAR2 base. Each doorbell is 64-bit (wptr_lo + wptr_hi).
    # Linux writes only the 64-bit wptr value.
    doorbell_byte_off = mes_kiq_doorbell * 4
    next_wptr = hqd_rd(regCP_HQD_PQ_WPTR_LO)
    if next_wptr:
        print(f"  KIQ HQD already active; appending at current WPTR=0x{next_wptr:x}")
    submit_nop = os.environ.get("PHASE9_SKIP_NOP") != "1"
    processed = False
    set_hw_done = False
    set_hw1_done = False

    if submit_nop:
        # ==== Submit PM4 NOP via ring + doorbell ====
        pm4_nop = 0xC0001000
        ring_dw = RING_SIZE // 4
        nop_wptr = next_wptr
        vram_wr(RING_VRAM_OFF + ((nop_wptr % ring_dw) * 4), pm4_nop)
        vram_wr(RING_VRAM_OFF + (((nop_wptr + 1) % ring_dw) * 4), 0)
        next_wptr = nop_wptr + 2

        print(f"\n== ring doorbell at BAR2 + 0x{doorbell_byte_off:x}, wptr={next_wptr} ==")
        vram_wr64(WPTR_VRAM_OFF, next_wptr)
        bar2_wr64(doorbell_byte_off, next_wptr)

        # ==== Observe ====
        print(f"\n== poll for queue consumption (5s) ==")
        deadline = time.time() + 5
        last = None
        while time.time() < deadline:
            rptr_report = vram_rd(RPTR_VRAM_OFF)
            rptr_reg = hqd_rd(regCP_HQD_PQ_RPTR)
            wptr_reg = hqd_rd(regCP_HQD_PQ_WPTR_LO)
            active = hqd_rd(regCP_HQD_ACTIVE)
            cp_stat = gc0_rd(regCP_STAT)
            grbm = gc0_rd(regGRBM_STATUS)
            mes_hdr = gc1_rd(regCP_MES_HEADER_DUMP)
            snap = (rptr_report, rptr_reg, wptr_reg, active, cp_stat, grbm)
            if snap != last:
                t = time.time() - deadline + 5
                print(f"  t={t:5.2f}s rptr_report=0x{rptr_report:x} rptr_reg=0x{rptr_reg:x} "
                      f"wptr_reg=0x{wptr_reg:x} active=0x{active:x} CP_STAT=0x{cp_stat:x} "
                      f"GRBM=0x{grbm:x} MES_HDR=0x{mes_hdr:x}")
                last = snap
            if rptr_reg == next_wptr or rptr_report == next_wptr:
                processed = True
                print(f"  QUEUE CONSUMED PM4 ✓")
                break
            time.sleep(0.05)

        if not processed:
            print("  queue did not consume PM4 before timeout")
    else:
        processed = hqd_rd(regCP_HQD_ACTIVE) != 0
        print(f"\n== PM4 NOP skipped; starting MES API stream at ring DW 0x{next_wptr:x} ==")

    if os.environ.get("PHASE9_SEND_SET_HW_RSRC") == "1":
        if not processed:
            print("\n== MES SET_HW_RSRC skipped: queue is not active ==")
        else:
            fence_value = 1
            fence_mc = fb_base + FENCE_VRAM_OFF
            sch_ctx_mc = fb_base + SCH_CTX_VRAM_OFF
            query_status_mc = fb_base + QUERY_STATUS_VRAM_OFF
            vram_wr64(FENCE_VRAM_OFF, 0)

            set_hw_wptr = next_wptr
            set_hw_end_wptr = set_hw_wptr + API_FRAME_SIZE_IN_DWORDS
            pkt = [0] * API_FRAME_SIZE_IN_DWORDS
            pkt[0] = (
                MES_API_TYPE_SCHEDULER |
                (MES_SCH_API_SET_HW_RSRC << 4) |
                (API_FRAME_SIZE_IN_DWORDS << 12)
            )

            put64(pkt, SET_HW_DW_G_SCH_CTX, sch_ctx_mc)
            put64(pkt, SET_HW_DW_QUERY_STATUS, query_status_mc)
            for i, base in enumerate(GC_BASES):
                pkt[SET_HW_DW_GC_BASE + i] = base
            for i, base in enumerate(MMHUB_BASES):
                pkt[SET_HW_DW_MMHUB_BASE + i] = base
            for i, base in enumerate(OSSSYS_BASES):
                pkt[SET_HW_DW_OSSSYS_BASE + i] = base
            put64(pkt, SET_HW_DW_API_STATUS, fence_mc)
            put64(pkt, SET_HW_DW_API_STATUS + 2, fence_value)
            pkt[SET_HW_DW_FLAGS] = (
                (1 << 0) |   # disable_reset
                (1 << 1) |   # use_different_vmid_compute
                (1 << 2) |   # disable_mes_log
                (1 << 6) |   # enable_level_process_quantum_check
                (1 << 10) |  # enable_reg_active_poll
                (1 << 19)    # unmapped_doorbell_handling = 1
            )
            pkt[SET_HW_DW_OVERSUBSCRIPTION_TIMER] = 0

            print(f"\n== MES SET_HW_RSRC at ring DW {set_hw_wptr}, status_fence=0x{fence_mc:x} ==")
            print(f"  sch_ctx=0x{sch_ctx_mc:x} query_status=0x{query_status_mc:x} flags=0x{pkt[SET_HW_DW_FLAGS]:x}")
            ring_dw = RING_SIZE // 4
            for i, word in enumerate(pkt):
                vram_wr(RING_VRAM_OFF + (((set_hw_wptr + i) % ring_dw) * 4), word)

            vram_wr64(WPTR_VRAM_OFF, set_hw_end_wptr)
            bar2_wr64(doorbell_byte_off, set_hw_end_wptr)

            deadline = time.time() + 5
            last = None
            set_hw_done = False
            while time.time() < deadline:
                fence = vram_rd64(FENCE_VRAM_OFF)
                rptr_reg = hqd_rd(regCP_HQD_PQ_RPTR)
                wptr_reg = hqd_rd(regCP_HQD_PQ_WPTR_LO)
                active = hqd_rd(regCP_HQD_ACTIVE)
                snap = (fence, rptr_reg, wptr_reg, active)
                if snap != last:
                    t = time.time() - deadline + 5
                    print(f"  t={t:5.2f}s fence=0x{fence:016x} rptr_reg=0x{rptr_reg:x} "
                          f"wptr_reg=0x{wptr_reg:x} active=0x{active:x}")
                    last = snap
                if fence == fence_value:
                    set_hw_done = True
                    print("  MES SET_HW_RSRC status fence signaled ✓")
                    break
                time.sleep(0.05)

            if not set_hw_done:
                print("  MES SET_HW_RSRC did not signal status fence before timeout")
            next_wptr = set_hw_end_wptr

            if set_hw_done and os.environ.get("PHASE9_SEND_SET_HW_RSRC_1") == "1":
                fence_value = 2
                resource1_mc = fb_base + RESOURCE1_VRAM_OFF
                vram_wr64(FENCE_VRAM_OFF, 0)

                set_hw1_wptr = next_wptr
                set_hw1_end_wptr = set_hw1_wptr + API_FRAME_SIZE_IN_DWORDS
                pkt = [0] * API_FRAME_SIZE_IN_DWORDS
                pkt[0] = (
                    MES_API_TYPE_SCHEDULER |
                    (MES_SCH_API_SET_HW_RSRC_1 << 4) |
                    (API_FRAME_SIZE_IN_DWORDS << 12)
                )

                put64(pkt, SET_HW1_DW_API_STATUS, fence_mc)
                put64(pkt, SET_HW1_DW_API_STATUS + 2, fence_value)
                pkt[SET_HW1_DW_MES_KIQ_UNMAP_TIMEOUT] = 0xA
                put64(pkt, SET_HW1_DW_CLEANER_SHADER_FENCE, resource1_mc)

                print(f"\n== MES SET_HW_RSRC_1 at ring DW {set_hw1_wptr}, status_fence=0x{fence_mc:x} ==")
                print(f"  cleaner_shader_fence=0x{resource1_mc:x}")
                for i, word in enumerate(pkt):
                    vram_wr(RING_VRAM_OFF + (((set_hw1_wptr + i) % ring_dw) * 4), word)

                vram_wr64(WPTR_VRAM_OFF, set_hw1_end_wptr)
                bar2_wr64(doorbell_byte_off, set_hw1_end_wptr)

                deadline = time.time() + 5
                last = None
                set_hw1_done = False
                while time.time() < deadline:
                    fence = vram_rd64(FENCE_VRAM_OFF)
                    rptr_reg = hqd_rd(regCP_HQD_PQ_RPTR)
                    wptr_reg = hqd_rd(regCP_HQD_PQ_WPTR_LO)
                    active = hqd_rd(regCP_HQD_ACTIVE)
                    snap = (fence, rptr_reg, wptr_reg, active)
                    if snap != last:
                        t = time.time() - deadline + 5
                        print(f"  t={t:5.2f}s fence=0x{fence:016x} rptr_reg=0x{rptr_reg:x} "
                              f"wptr_reg=0x{wptr_reg:x} active=0x{active:x}")
                        last = snap
                    if fence == fence_value:
                        set_hw1_done = True
                        print("  MES SET_HW_RSRC_1 status fence signaled ✓")
                        break
                    time.sleep(0.05)

                if not set_hw1_done:
                    print("  MES SET_HW_RSRC_1 did not signal status fence before timeout")
                next_wptr = set_hw1_end_wptr

    if os.environ.get("PHASE9_MAP_SCHED") == "1":
        assume_sched_mapped = os.environ.get("PHASE9_ASSUME_SCHED_MAPPED") == "1"
        sched_query_only = os.environ.get("PHASE9_SCHED_QUERY_ONLY") == "1"
        if not (set_hw_done and set_hw1_done) and not assume_sched_mapped:
            print("\n== MES scheduler map skipped: requires successful "
                  "PHASE9_SEND_SET_HW_RSRC=1 and PHASE9_SEND_SET_HW_RSRC_1=1 ==")
        else:
            sched_me, sched_pipe, sched_queue = 3, 0, 0
            sched_doorbell = AMDGPU_NAVI10_DOORBELL_MES_RING0 << 1
            fence_mc = fb_base + FENCE_VRAM_OFF
            if assume_sched_mapped:
                select_hqd(sched_me, sched_pipe, sched_queue)
                add_done = hqd_rd(regCP_HQD_ACTIVE) != 0
                print("\n== PHASE9_ASSUME_SCHED_MAPPED=1: using existing scheduler HQD ==")
            else:
                sched_mqd = build_sched_mqd()
                for i, v in enumerate(sched_mqd):
                    vram_wr(SCHED_MQD_VRAM_OFF + i * 4, v)

                print(f"\n== scheduler MQD key fields ==")
                print(f"  cp_hqd_pq_doorbell_control = 0x{sched_mqd[0x8F]:08x} "
                      f"(offset=0x{sched_doorbell:x})")
                print(f"  cp_hqd_pq_control          = 0x{sched_mqd[0x91]:08x}")

                add_queue_fence = 3
                pkt = [0] * API_FRAME_SIZE_IN_DWORDS
                pkt[0] = mes_header(MES_SCH_API_ADD_QUEUE)
                pkt[ADD_QUEUE_DW_DOORBELL_OFFSET] = sched_doorbell
                put64(pkt, ADD_QUEUE_DW_MQD_ADDR, sched_mqd_mc)
                put64(pkt, ADD_QUEUE_DW_WPTR_ADDR, sched_wptr_mc)
                pkt[ADD_QUEUE_DW_QUEUE_TYPE] = MES_QUEUE_TYPE_SCHQ
                pkt[ADD_QUEUE_DW_FLAGS] = 1 << 13  # map_legacy_kq
                put64(pkt, ADD_QUEUE_DW_API_STATUS, fence_mc)
                put64(pkt, ADD_QUEUE_DW_API_STATUS + 2, add_queue_fence)
                pkt[ADD_QUEUE_DW_PIPE_ID] = sched_pipe
                pkt[ADD_QUEUE_DW_QUEUE_ID] = sched_queue

                print(f"  ADD_QUEUE sched mqd=0x{sched_mqd_mc:x} "
                      f"wptr_addr=0x{sched_wptr_mc:x} doorbell=0x{sched_doorbell:x}")
                add_done, next_wptr = submit_api_and_poll(
                    "MES ADD_QUEUE sched",
                    RING_VRAM_OFF,
                    WPTR_VRAM_OFF,
                    mes_kiq_doorbell,
                    (kiq_me, kiq_pipe, kiq_queue, 0),
                    next_wptr,
                    pkt,
                    add_queue_fence,
                )

            if add_done:
                select_hqd(sched_me, sched_pipe, sched_queue)
                print(f"\nScheduler HQD readback after ADD_QUEUE:")
                print(f"  ACTIVE=0x{hqd_rd(regCP_HQD_ACTIVE):08x} "
                      f"RPTR=0x{hqd_rd(regCP_HQD_PQ_RPTR):08x} "
                      f"WPTR=0x{hqd_rd(regCP_HQD_PQ_WPTR_LO):08x} "
                      f"PQ_BASE=0x{hqd_rd(regCP_HQD_PQ_BASE_HI):08x}:"
                      f"{hqd_rd(regCP_HQD_PQ_BASE):08x} "
                      f"PQ_CONTROL=0x{hqd_rd(regCP_HQD_PQ_CONTROL):08x} "
                      f"DOORBELL_CONTROL=0x{hqd_rd(regCP_HQD_PQ_DOORBELL_CONTROL):08x} "
                      f"MQD=0x{hqd_rd(regCP_MQD_BASE_ADDR_HI):08x}:"
                      f"{hqd_rd(regCP_MQD_BASE_ADDR):08x}")

                sched_next_wptr = hqd_rd(regCP_HQD_PQ_WPTR_LO) if assume_sched_mapped else 0
                sched_set_hw_fence = 4
                sched_sch_ctx_mc = fb_base + SCHED_SCH_CTX_VRAM_OFF
                sched_query_status_mc = fb_base + SCHED_QUERY_STATUS_VRAM_OFF
                sched_set_hw_done = sched_query_only
                if sched_query_only:
                    print(f"  PHASE9_SCHED_QUERY_ONLY=1: starting scheduler query at "
                          f"WPTR=0x{sched_next_wptr:x}")
                else:
                    pkt = [0] * API_FRAME_SIZE_IN_DWORDS
                    pkt[0] = mes_header(MES_SCH_API_SET_HW_RSRC)
                    pkt[SET_HW_DW_VMID_MASK_MMHUB] = MES_SCHQ_VMID_MASK
                    pkt[SET_HW_DW_VMID_MASK_GFXHUB] = MES_SCHQ_VMID_MASK
                    for i, mask in enumerate(MES_COMPUTE_HQD_MASKS):
                        pkt[SET_HW_DW_COMPUTE_HQD_MASK + i] = mask
                    for i, mask in enumerate(MES_GFX_HQD_MASKS):
                        pkt[SET_HW_DW_GFX_HQD_MASK + i] = mask
                    for i, mask in enumerate(MES_SDMA_HQD_MASKS):
                        pkt[SET_HW_DW_SDMA_HQD_MASK + i] = mask
                    for i, doorbell in enumerate(MES_AGGREGATED_DOORBELLS):
                        pkt[SET_HW_DW_AGGREGATED_DOORBELLS + i] = doorbell
                    put64(pkt, SET_HW_DW_G_SCH_CTX, sched_sch_ctx_mc)
                    put64(pkt, SET_HW_DW_QUERY_STATUS, sched_query_status_mc)
                    for i, base in enumerate(GC_BASES):
                        pkt[SET_HW_DW_GC_BASE + i] = base
                    for i, base in enumerate(MMHUB_BASES):
                        pkt[SET_HW_DW_MMHUB_BASE + i] = base
                    for i, base in enumerate(OSSSYS_BASES):
                        pkt[SET_HW_DW_OSSSYS_BASE + i] = base
                    put64(pkt, SET_HW_DW_API_STATUS, fence_mc)
                    put64(pkt, SET_HW_DW_API_STATUS + 2, sched_set_hw_fence)
                    pkt[SET_HW_DW_FLAGS] = (
                        (1 << 0) |
                        (1 << 1) |
                        (1 << 2) |
                        (1 << 6) |
                        (1 << 10) |
                        (1 << 19)
                    )
                    pkt[SET_HW_DW_OVERSUBSCRIPTION_TIMER] = 0
                    print(f"  SCHED SET_HW_RSRC sch_ctx=0x{sched_sch_ctx_mc:x} "
                          f"query=0x{sched_query_status_mc:x}")
                    sched_set_hw_done, sched_next_wptr = submit_api_and_poll(
                        "SCHED SET_HW_RSRC",
                        SCHED_RING_VRAM_OFF,
                        SCHED_WPTR_VRAM_OFF,
                        sched_doorbell,
                        (sched_me, sched_pipe, sched_queue, 0),
                        sched_next_wptr,
                        pkt,
                        sched_set_hw_fence,
                    )

                if sched_set_hw_done and not sched_query_only:
                    sched_set_hw1_fence = 5
                    sched_resource1_mc = fb_base + SCHED_RESOURCE1_VRAM_OFF
                    pkt = [0] * API_FRAME_SIZE_IN_DWORDS
                    pkt[0] = mes_header(MES_SCH_API_SET_HW_RSRC_1)
                    put64(pkt, SET_HW1_DW_API_STATUS, fence_mc)
                    put64(pkt, SET_HW1_DW_API_STATUS + 2, sched_set_hw1_fence)
                    pkt[SET_HW1_DW_MES_KIQ_UNMAP_TIMEOUT] = 0xA
                    put64(pkt, SET_HW1_DW_CLEANER_SHADER_FENCE, sched_resource1_mc)
                    print(f"  SCHED SET_HW_RSRC_1 resource=0x{sched_resource1_mc:x}")
                    sched_set_hw1_done, sched_next_wptr = submit_api_and_poll(
                        "SCHED SET_HW_RSRC_1",
                        SCHED_RING_VRAM_OFF,
                        SCHED_WPTR_VRAM_OFF,
                        sched_doorbell,
                        (sched_me, sched_pipe, sched_queue, 0),
                        sched_next_wptr,
                        pkt,
                        sched_set_hw1_fence,
                    )
                else:
                    sched_set_hw1_done = sched_query_only

                if sched_set_hw1_done:
                    sched_query_fence = 6
                    pkt = [0] * API_FRAME_SIZE_IN_DWORDS
                    pkt[0] = mes_header(MES_SCH_API_QUERY_SCHEDULER_STATUS)
                    put64(pkt, 2, fence_mc)
                    put64(pkt, 4, sched_query_fence)
                    sched_query_done, sched_next_wptr = submit_api_and_poll(
                        "SCHED QUERY_SCHEDULER_STATUS",
                        SCHED_RING_VRAM_OFF,
                        SCHED_WPTR_VRAM_OFF,
                        sched_doorbell,
                        (sched_me, sched_pipe, sched_queue, 0),
                        sched_next_wptr,
                        pkt,
                        sched_query_fence,
                    )
                    if not sched_query_done:
                        print("  SCHED QUERY_SCHEDULER_STATUS failed after scheduler resources")

    if os.environ.get("PHASE9_SEND_MES_QUERY") == "1":
        if not processed:
            print("\n== MES QUERY_SCHEDULER_STATUS skipped: queue is not active ==")
        else:
            select_hqd(kiq_me, kiq_pipe, kiq_queue)
            fence_value = 0x1234567800000001
            fence_dma = None
            if os.environ.get("PHASE9_QUERY_FENCE") == "dma":
                fence_dma = c.alloc_dma(0x1000)
                fence_mc = fence_dma.segments[0][0]
                ctypes.c_uint64.from_address(fence_dma.cpu_addr).value = 0
                def read_query_fence():
                    return ctypes.c_uint64.from_address(fence_dma.cpu_addr).value
                print(f"  using DMA fence bus=0x{fence_mc:x} cpu=0x{fence_dma.cpu_addr:x}")
            else:
                fence_mc = fb_base + FENCE_VRAM_OFF
                vram_wr64(FENCE_VRAM_OFF, 0)
                def read_query_fence():
                    return vram_rd64(FENCE_VRAM_OFF)

            query_wptr = next_wptr
            query_end_wptr = query_wptr + API_FRAME_SIZE_IN_DWORDS
            pkt = [0] * API_FRAME_SIZE_IN_DWORDS
            pkt[0] = (
                MES_API_TYPE_SCHEDULER |
                (MES_SCH_API_QUERY_SCHEDULER_STATUS << 4) |
                (API_FRAME_SIZE_IN_DWORDS << 12)
            )
            # dword 1 is the QUERY_MES subopcode; Linux leaves it at 0
            # for mes_v12_0_query_sched_status().
            pkt[2] = fence_mc & 0xFFFFFFFF
            pkt[3] = (fence_mc >> 32) & 0xFFFFFFFF
            pkt[4] = fence_value & 0xFFFFFFFF
            pkt[5] = (fence_value >> 32) & 0xFFFFFFFF

            print(f"\n== MES QUERY_SCHEDULER_STATUS at ring DW {query_wptr}, fence=0x{fence_mc:x} ==")
            ring_dw = RING_SIZE // 4
            for i, word in enumerate(pkt):
                vram_wr(RING_VRAM_OFF + (((query_wptr + i) % ring_dw) * 4), word)

            vram_wr64(WPTR_VRAM_OFF, query_end_wptr)
            bar2_wr64(doorbell_byte_off, query_end_wptr)

            deadline = time.time() + 5
            last = None
            query_done = False
            while time.time() < deadline:
                fence = read_query_fence()
                rptr_reg = hqd_rd(regCP_HQD_PQ_RPTR)
                wptr_reg = hqd_rd(regCP_HQD_PQ_WPTR_LO)
                active = hqd_rd(regCP_HQD_ACTIVE)
                snap = (fence, rptr_reg, wptr_reg, active)
                if snap != last:
                    t = time.time() - deadline + 5
                    print(f"  t={t:5.2f}s fence=0x{fence:016x} rptr_reg=0x{rptr_reg:x} "
                          f"wptr_reg=0x{wptr_reg:x} active=0x{active:x}")
                    last = snap
                if fence == fence_value:
                    query_done = True
                    print("  MES QUERY_SCHEDULER_STATUS fence signaled ✓")
                    break
                time.sleep(0.05)

            if not query_done:
                print("  MES QUERY_SCHEDULER_STATUS did not signal fence before timeout")
            if fence_dma is not None:
                c.free_dma(fence_dma.buffer_id)

    gc1_wr(regGRBM_GFX_CNTL, 0)


if __name__ == "__main__":
    main()
