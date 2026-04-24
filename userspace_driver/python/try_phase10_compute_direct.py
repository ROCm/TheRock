"""Phase 10: direct compute HQD smoke test on gfx1201 macOS.

This intentionally bypasses MES queue creation and programs one inactive
compute HQD directly, using fixed VRAM offsets for the MQD/ring/EOP/WPTR
buffers. The DEXT DMA path currently reports non-unique bus addresses, so
VRAM BAR0-backed buffers are less ambiguous for the first compute smoke.

Expected success signal: CP_HQD_PQ_RPTR advances to the submitted PM4 NOP
write pointer after a BAR2 doorbell write.
"""
from __future__ import annotations

import ctypes
import os
import struct
import sys
import time

from amd_gpu_driver.commands.pm4 import INT_SEL_NONE, PM4PacketBuilder
from amd_gpu_driver.backends.macos.iokit_client import IOKitClient

_MMIO_BAR = 5
GC_B0 = 0x1260
GC_B1 = 0xA000

regGRBM_GFX_CNTL = 0x0900
regCP_MQD_BASE_ADDR = 0x1fa9
regCP_MQD_BASE_ADDR_HI = 0x1faa
regCP_HQD_ACTIVE = 0x1fab
regCP_HQD_VMID = 0x1fac
regCP_HQD_PERSISTENT_STATE = 0x1fad
regCP_HQD_PQ_BASE = 0x1fb1
regCP_HQD_PQ_BASE_HI = 0x1fb2
regCP_HQD_PQ_RPTR = 0x1fb3
regCP_HQD_PQ_RPTR_REPORT_ADDR = 0x1fb4
regCP_HQD_PQ_RPTR_REPORT_ADDR_HI = 0x1fb5
regCP_HQD_PQ_WPTR_POLL_ADDR = 0x1fb6
regCP_HQD_PQ_WPTR_POLL_ADDR_HI = 0x1fb7
regCP_HQD_PQ_DOORBELL_CONTROL = 0x1fb8
regCP_HQD_PQ_CONTROL = 0x1fba
regCP_HQD_DEQUEUE_REQUEST = 0x1fc1
regCP_MQD_CONTROL = 0x1fcb
regCP_HQD_EOP_BASE_ADDR = 0x1fce
regCP_HQD_EOP_BASE_ADDR_HI = 0x1fcf
regCP_HQD_EOP_CONTROL = 0x1fd0
regCP_HQD_PQ_WPTR_LO = 0x1fdf
regCP_HQD_PQ_WPTR_HI = 0x1fe0

regCP_STAT = 0x0f40
regGRBM_STATUS = 0x0da4

CP_HQD_PERSISTENT_STATE_DEFAULT = 0x0be05501

MQD_OFF = 0x1900000
RING_OFF = 0x1902000
EOP_OFF = 0x1910000
RPTR_OFF = 0x1920000
WPTR_OFF = 0x1921000
FENCE_OFF = 0x1930000

MQD_SIZE = 0x1000
RING_SIZE = 0x1000
EOP_SIZE = 0x1000

# Keep this inside the CP_MEC_DOORBELL_RANGE programmed by phase 9, and away
# from MES scheduler/KIQ doorbells 0x16/0x18.
COMPUTE_DOORBELL = 0x20
PACKET3_WRITE_DATA = 0x37
WRITE_DATA_DST_SEL_MEM_ASYNC = 5
WRITE_DATA_WR_CONFIRM = 1 << 20


def main() -> None:
    c = IOKitClient()
    c.open()
    info = c.get_info()
    print(f"device=0x{info.device_id:04x} rev=0x{info.revision_id:02x}")

    def rd(base, off):
        return c.mmio_read32(_MMIO_BAR, (base + off) * 4)

    def wr(base, off, value):
        c.mmio_write32(_MMIO_BAR, (base + off) * 4, value & 0xFFFFFFFF)

    def gc0_rd(off):
        return rd(GC_B0, off)

    def gc0_wr(off, value):
        wr(GC_B0, off, value)

    def gc1_wr(off, value):
        wr(GC_B1, off, value)

    def select_hqd(me, pipe, queue=0, vmid=0):
        gc1_wr(
            regGRBM_GFX_CNTL,
            ((pipe & 0x3) << 0)
            | ((me & 0x3) << 2)
            | ((vmid & 0xf) << 4)
            | ((queue & 0x7) << 8),
        )

    fb_base = (c.mmio_read32(_MMIO_BAR, (0x1A000 + 0x0554) * 4) & 0xFFFFFF) << 24
    bar0_cpu, _ = c.map_bar(0)
    bar2_cpu, bar2_size = c.map_bar(2)
    print(f"fb_base=0x{fb_base:x} BAR2 size={bar2_size // 1024}KB")

    def vram_wr32(off, value):
        (ctypes.c_uint32 * 1).from_address(bar0_cpu + off)[0] = value & 0xFFFFFFFF

    def vram_wr64(off, value):
        (ctypes.c_uint64 * 1).from_address(bar0_cpu + off)[0] = value & 0xFFFFFFFFFFFFFFFF

    def vram_rd64(off):
        return (ctypes.c_uint64 * 1).from_address(bar0_cpu + off)[0]

    def doorbell_wr64(index, value):
        (ctypes.c_uint64 * 1).from_address(bar2_cpu + index * 4)[0] = (
            value & 0xFFFFFFFFFFFFFFFF
        )

    for base, size in (
        (MQD_OFF, MQD_SIZE),
        (RING_OFF, RING_SIZE),
        (EOP_OFF, EOP_SIZE),
        (RPTR_OFF, 0x20),
        (WPTR_OFF, 0x20),
        (FENCE_OFF, 0x20),
    ):
        for i in range(0, size, 4):
            vram_wr32(base + i, 0)

    me = int(os.environ.get("PHASE10_COMPUTE_ME", "1"), 0)
    pipe = int(os.environ.get("PHASE10_COMPUTE_PIPE", "0"), 0)
    queue = int(os.environ.get("PHASE10_COMPUTE_QUEUE", "0"), 0)
    select_hqd(me, pipe, queue)

    pre_active = gc0_rd(regCP_HQD_ACTIVE)
    print(f"target compute HQD me={me} pipe={pipe} queue={queue} PRE_ACTIVE=0x{pre_active:x}")
    if pre_active and os.environ.get("PHASE10_FORCE_ACTIVE") != "1":
        print("target HQD is already active; set PHASE10_FORCE_ACTIVE=1 to overwrite")
        sys.exit(2)

    if pre_active:
        gc0_wr(regCP_HQD_DEQUEUE_REQUEST, 1)
        deadline = time.time() + 1
        while time.time() < deadline and gc0_rd(regCP_HQD_ACTIVE):
            time.sleep(0.001)
        gc0_wr(regCP_HQD_DEQUEUE_REQUEST, 0)

    mqd_mc = fb_base + MQD_OFF
    ring_mc = fb_base + RING_OFF
    eop_mc = fb_base + EOP_OFF
    rptr_mc = fb_base + RPTR_OFF
    wptr_mc = fb_base + WPTR_OFF

    mqd = [0] * (MQD_SIZE // 4)
    mqd[0] = 0xC0310800
    mqd[1] = 1
    for dw in (0x17, 0x18, 0x1A, 0x1B):
        mqd[dw] = 0xFFFFFFFF
    mqd[0x2C] = 7

    eop_base_addr_shifted = eop_mc >> 8
    mqd[0xA5] = eop_base_addr_shifted & 0xFFFFFFFF
    mqd[0xA6] = (eop_base_addr_shifted >> 32) & 0xFFFFFFFF
    mqd[0xA7] = ((EOP_SIZE // 4).bit_length() - 2) & 0x3F

    mqd[0x80] = mqd_mc & 0xFFFFFFFC
    mqd[0x81] = (mqd_mc >> 32) & 0xFFFFFFFF
    mqd[0x82] = 1
    mqd[0x83] = 0
    mqd[0x84] = (CP_HQD_PERSISTENT_STATE_DEFAULT & ~(0x3FF << 8)) | (0x55 << 8)

    pq_base_shifted = ring_mc >> 8
    mqd[0x88] = pq_base_shifted & 0xFFFFFFFF
    mqd[0x89] = (pq_base_shifted >> 32) & 0xFFFFFFFF
    mqd[0x8B] = rptr_mc & 0xFFFFFFFC
    mqd[0x8C] = (rptr_mc >> 32) & 0xFFFF
    mqd[0x8D] = wptr_mc & 0xFFFFFFF8
    mqd[0x8E] = (wptr_mc >> 32) & 0xFFFF
    mqd[0x8F] = ((COMPUTE_DOORBELL & 0x3FFFFFF) << 2) | (1 << 30)

    ring_dw = RING_SIZE // 4
    queue_size_val = (ring_dw.bit_length() - 2) & 0x3F
    mqd[0x91] = (
        queue_size_val
        | (5 << 8)
        | (1 << 0x1B)
        | (1 << 0x1C)
        | (1 << 0x1E)
        | (1 << 0x1F)
        | 0x300000
        | 0x8000
    )
    mqd[0x95] = 0x00300000
    mqd[0xA2] = 0x100
    mqd[0xB8] = 1 << 15

    for i, value in enumerate(mqd):
        vram_wr32(MQD_OFF + i * 4, value)

    gc0_wr(regCP_HQD_ACTIVE, 0)
    gc0_wr(regCP_HQD_PQ_RPTR, 0)
    gc0_wr(regCP_HQD_PQ_WPTR_LO, 0)
    gc0_wr(regCP_HQD_PQ_WPTR_HI, 0)
    gc0_wr(regCP_HQD_VMID, gc0_rd(regCP_HQD_VMID) & ~0xF)
    gc0_wr(regCP_HQD_PQ_DOORBELL_CONTROL, gc0_rd(regCP_HQD_PQ_DOORBELL_CONTROL) & ~0x40000000)
    gc0_wr(regCP_MQD_BASE_ADDR, mqd[0x80])
    gc0_wr(regCP_MQD_BASE_ADDR_HI, mqd[0x81])
    gc0_wr(regCP_MQD_CONTROL, 0)
    gc0_wr(regCP_HQD_EOP_BASE_ADDR, mqd[0xA5])
    gc0_wr(regCP_HQD_EOP_BASE_ADDR_HI, mqd[0xA6])
    gc0_wr(regCP_HQD_EOP_CONTROL, mqd[0xA7])
    gc0_wr(regCP_HQD_PQ_BASE, mqd[0x88])
    gc0_wr(regCP_HQD_PQ_BASE_HI, mqd[0x89])
    gc0_wr(regCP_HQD_PQ_RPTR_REPORT_ADDR, mqd[0x8B])
    gc0_wr(regCP_HQD_PQ_RPTR_REPORT_ADDR_HI, mqd[0x8C])
    gc0_wr(regCP_HQD_PQ_CONTROL, mqd[0x91])
    gc0_wr(regCP_HQD_PQ_WPTR_POLL_ADDR, mqd[0x8D])
    gc0_wr(regCP_HQD_PQ_WPTR_POLL_ADDR_HI, mqd[0x8E])
    gc0_wr(regCP_HQD_PQ_DOORBELL_CONTROL, mqd[0x8F])
    gc0_wr(regCP_HQD_PERSISTENT_STATE, mqd[0x84])
    gc0_wr(regCP_HQD_ACTIVE, 1)
    time.sleep(0.01)

    print("HQD readback after program:")
    print(
        f"  ACTIVE=0x{gc0_rd(regCP_HQD_ACTIVE):08x} "
        f"PQ_BASE=0x{gc0_rd(regCP_HQD_PQ_BASE_HI):08x}:{gc0_rd(regCP_HQD_PQ_BASE):08x} "
        f"PQ_CONTROL=0x{gc0_rd(regCP_HQD_PQ_CONTROL):08x} "
        f"DOORBELL_CONTROL=0x{gc0_rd(regCP_HQD_PQ_DOORBELL_CONTROL):08x}"
    )
    print(
        f"  RPTR=0x{gc0_rd(regCP_HQD_PQ_RPTR):08x} "
        f"WPTR=0x{gc0_rd(regCP_HQD_PQ_WPTR_LO):08x}:"
        f"{gc0_rd(regCP_HQD_PQ_WPTR_HI):08x}"
    )

    # PM4 type-3 NOP: header + one body DW.
    vram_wr32(RING_OFF, 0xC0001000)
    vram_wr32(RING_OFF + 4, 0)
    vram_wr64(WPTR_OFF, 2)
    doorbell_wr64(COMPUTE_DOORBELL, 2)

    print(f"\n== compute PM4 NOP doorbell index=0x{COMPUTE_DOORBELL:x} wptr=2 ==")
    deadline = time.time() + 5
    last = None
    while time.time() < deadline:
        rptr = gc0_rd(regCP_HQD_PQ_RPTR)
        wptr = gc0_rd(regCP_HQD_PQ_WPTR_LO)
        active = gc0_rd(regCP_HQD_ACTIVE)
        cp_stat = gc0_rd(regCP_STAT)
        grbm = gc0_rd(regGRBM_STATUS)
        snap = (rptr, wptr, active, cp_stat, grbm)
        if snap != last:
            print(
                f"  rptr=0x{rptr:x} wptr=0x{wptr:x} active=0x{active:x} "
                f"CP_STAT=0x{cp_stat:x} GRBM=0x{grbm:x}"
            )
            last = snap
        if rptr == 2:
            print("  COMPUTE QUEUE CONSUMED PM4 NOP ✓")
            break
        time.sleep(0.05)
    else:
        print("  compute queue did not consume PM4 before timeout")

    if os.environ.get("PHASE10_RELEASE_MEM") == "1":
        fence_value = int(os.environ.get("PHASE10_FENCE_VALUE", "0x12345678"), 0)
        fence_mc = fb_base + FENCE_OFF
        vram_wr64(FENCE_OFF, 0)

        builder = PM4PacketBuilder()
        builder.release_mem(
            addr=fence_mc,
            value=fence_value,
            int_sel=INT_SEL_NONE,
            cache_flush=False,
        )
        packet = builder.build()
        dwords = list(struct.unpack(f"<{len(packet) // 4}I", packet))

        start = gc0_rd(regCP_HQD_PQ_WPTR_LO)
        for i, word in enumerate(dwords):
            vram_wr32(RING_OFF + (((start + i) % ring_dw) * 4), word)
        end = start + len(dwords)
        vram_wr64(WPTR_OFF, end)
        doorbell_wr64(COMPUTE_DOORBELL, end)

        print(f"\n== compute RELEASE_MEM fence=0x{fence_mc:x} value=0x{fence_value:x} "
              f"start=0x{start:x} end=0x{end:x} ==")
        deadline = time.time() + 5
        last = None
        while time.time() < deadline:
            fence = vram_rd64(FENCE_OFF)
            rptr = gc0_rd(regCP_HQD_PQ_RPTR)
            wptr = gc0_rd(regCP_HQD_PQ_WPTR_LO)
            active = gc0_rd(regCP_HQD_ACTIVE)
            snap = (fence, rptr, wptr, active)
            if snap != last:
                print(
                    f"  fence=0x{fence:016x} rptr=0x{rptr:x} "
                    f"wptr=0x{wptr:x} active=0x{active:x}"
                )
                last = snap
            if fence == fence_value:
                print("  COMPUTE RELEASE_MEM fence signaled ✓")
                break
            time.sleep(0.05)
        else:
            print("  compute RELEASE_MEM fence did not signal before timeout")

    if os.environ.get("PHASE10_WRITE_DATA") == "1":
        write_addr = fb_base + FENCE_OFF
        values = [0xCAFE0001, 0xCAFE0002]
        vram_wr64(FENCE_OFF, 0)

        n_body = 3 + len(values)
        header = (3 << 30) | (((n_body - 1) & 0x3FFF) << 16) | (PACKET3_WRITE_DATA << 8)
        control = (WRITE_DATA_DST_SEL_MEM_ASYNC << 8) | WRITE_DATA_WR_CONFIRM
        dwords = [
            header,
            control,
            write_addr & 0xFFFFFFFF,
            (write_addr >> 32) & 0xFFFFFFFF,
            *values,
        ]

        start = gc0_rd(regCP_HQD_PQ_WPTR_LO)
        for i, word in enumerate(dwords):
            vram_wr32(RING_OFF + (((start + i) % ring_dw) * 4), word)
        end = start + len(dwords)
        vram_wr64(WPTR_OFF, end)
        doorbell_wr64(COMPUTE_DOORBELL, end)

        print(f"\n== compute WRITE_DATA addr=0x{write_addr:x} start=0x{start:x} end=0x{end:x} ==")
        deadline = time.time() + 5
        last = None
        while time.time() < deadline:
            written = vram_rd64(FENCE_OFF)
            rptr = gc0_rd(regCP_HQD_PQ_RPTR)
            wptr = gc0_rd(regCP_HQD_PQ_WPTR_LO)
            active = gc0_rd(regCP_HQD_ACTIVE)
            snap = (written, rptr, wptr, active)
            if snap != last:
                print(
                    f"  written=0x{written:016x} rptr=0x{rptr:x} "
                    f"wptr=0x{wptr:x} active=0x{active:x}"
                )
                last = snap
            if written == ((values[1] << 32) | values[0]):
                print("  COMPUTE WRITE_DATA wrote VRAM ✓")
                break
            time.sleep(0.05)
        else:
            print("  compute WRITE_DATA did not update VRAM before timeout")

    gc1_wr(regGRBM_GFX_CNTL, 0)


if __name__ == "__main__":
    main()
