"""macOS queue manager — compute and SDMA queue creation.

Creates GPU hardware queues by:
  1. Allocating ring buffer, EOP buffer, and context save area in GTT
  2. Constructing a Micro Queue Descriptor (MQD) in system memory
  3. Programming the MQD address into CP/SDMA registers via MMIO
  4. Mapping the doorbell BAR for command submission

Queue submission:
  1. Write PM4/SDMA packets into the ring buffer
  2. Update the write pointer
  3. Write the doorbell register to notify the GPU

This follows the same pattern as the amdgpu_lite and Windows backends:
the kernel driver only provides DMA allocation and BAR mapping, while
all queue setup logic is in Python.
"""

from __future__ import annotations

import ctypes
import os
import time
from dataclasses import dataclass

from amd_gpu_driver.backends.base import (
    MemoryHandle,
    MemoryLocation,
    QueueHandle,
    QueueType,
)
from amd_gpu_driver.backends.macos.iokit_client import IOKitClient
from amd_gpu_driver.backends.macos.memory import MacOSMemoryManager

# Queue configuration constants
COMPUTE_RING_SIZE = 256 * 1024   # 256 KB (must be power of 2)
SDMA_RING_SIZE = 256 * 1024      # 256 KB
EOP_BUFFER_SIZE = 4096           # 4 KB
CTX_SAVE_SIZE = 0                # Context save disabled initially

# Doorbell constants for RDNA4
DOORBELL_BAR_INDEX = 2           # BAR2 = doorbell aperture on RDNA
DOORBELL_STRIDE = 8              # 8 bytes per doorbell (64-bit)

# gfx1201 / RDNA4 register access. BAR5 is the non-prefetchable MMIO BAR;
# BAR0 is the VRAM aperture on the tested RX 9070 XT eGPU.
_MMIO_BAR = 5
_VRAM_BAR = 0
GC_B0 = 0x1260
GC_B1 = 0xA000
NBIO_B2 = 0xD20

regRCC_DEV0_EPF0_RCC_DOORBELL_APER_EN = 0x00c0
regGDC_S2A0_S2A_DOORBELL_ENTRY_0_CTRL = 0x01cb
regGDC_S2A0_S2A_DOORBELL_ENTRY_3_CTRL = 0x01ce

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
regCP_MEC_DOORBELL_RANGE_LOWER = 0x1dfc
regCP_MEC_DOORBELL_RANGE_UPPER = 0x1dfd

CP_HQD_PERSISTENT_STATE_DEFAULT = 0x0be05501
MQD_SIZE = 0x1000
DIRECT_COMPUTE_RING_SIZE = 0x1000
DIRECT_COMPUTE_EOP_SIZE = 0x1000

# Reserve a high scratch range matching the proven phase-10 script. Each queue
# gets a private 256 KiB window so ring/MQD/EOP/WPTR buffers do not overlap.
DIRECT_COMPUTE_BASE_OFF = 0x1900000
DIRECT_COMPUTE_STRIDE = 0x40000
DIRECT_COMPUTE_MQD_REL = 0x00000
DIRECT_COMPUTE_RING_REL = 0x02000
DIRECT_COMPUTE_EOP_REL = 0x10000
DIRECT_COMPUTE_RPTR_REL = 0x20000
DIRECT_COMPUTE_WPTR_REL = 0x21000
DIRECT_COMPUTE_DOORBELL = 0x20


@dataclass
class _DirectComputeQueue:
    """Bookkeeping for the current VRAM-backed macOS compute queue path."""

    me: int
    pipe: int
    queue: int
    doorbell_index: int
    mqd_off: int
    ring_off: int
    eop_off: int
    rptr_off: int
    wptr_off: int
    mqd_mc: int
    ring_mc: int
    eop_mc: int
    rptr_mc: int
    wptr_mc: int
    wptr: int = 0


class MacOSQueueManager:
    """Manages GPU compute and SDMA queues on macOS.

    After bringup configures the GPU's MES (Micro Engine Scheduler) or
    CP (Command Processor), this manager allocates queue resources and
    programs the hardware to accept commands.
    """

    def __init__(
        self,
        client: IOKitClient,
        memory: MacOSMemoryManager,
    ) -> None:
        self._client = client
        self._memory = memory
        self._queues: dict[int, QueueHandle] = {}
        self._next_queue_id = 1

        # Doorbell state (populated when BAR2 is mapped)
        self._doorbell_addr: int = 0
        self._doorbell_size: int = 0
        self._next_doorbell_offset: int = 0
        self._vram_addr: int = 0
        self._vram_size: int = 0
        self._direct_compute: dict[int, _DirectComputeQueue] = {}

    def set_doorbell_bar(self, addr: int, size: int) -> None:
        """Set the doorbell BAR mapping (called during bringup)."""
        self._doorbell_addr = addr
        self._doorbell_size = size

    def create_compute_queue(self) -> QueueHandle:
        """Create a VRAM-backed direct compute queue.

        This is the first macOS hardware path that is known to work on the
        RX 9070 XT eGPU. It deliberately uses VRAM BAR-backed MQD/ring
        buffers because the current DEXT DMA allocator reports non-unique
        bus addresses for separate GTT allocations.
        """
        queue_id = self._next_queue_id
        self._next_queue_id += 1

        self._ensure_bar_mappings()
        self._ensure_doorbell_aperture()

        q_index = queue_id - 1
        me = 1
        pipe = q_index // 4
        queue = q_index % 4
        if pipe >= 2:
            raise RuntimeError("direct macOS compute queue prototype supports 8 queues")

        base_off = DIRECT_COMPUTE_BASE_OFF + q_index * DIRECT_COMPUTE_STRIDE
        meta = _DirectComputeQueue(
            me=me,
            pipe=pipe,
            queue=queue,
            doorbell_index=DIRECT_COMPUTE_DOORBELL + q_index * 2,
            mqd_off=base_off + DIRECT_COMPUTE_MQD_REL,
            ring_off=base_off + DIRECT_COMPUTE_RING_REL,
            eop_off=base_off + DIRECT_COMPUTE_EOP_REL,
            rptr_off=base_off + DIRECT_COMPUTE_RPTR_REL,
            wptr_off=base_off + DIRECT_COMPUTE_WPTR_REL,
            mqd_mc=0,
            ring_mc=0,
            eop_mc=0,
            rptr_mc=0,
            wptr_mc=0,
        )

        fb_base = self._framebuffer_base()
        meta.mqd_mc = fb_base + meta.mqd_off
        meta.ring_mc = fb_base + meta.ring_off
        meta.eop_mc = fb_base + meta.eop_off
        meta.rptr_mc = fb_base + meta.rptr_off
        meta.wptr_mc = fb_base + meta.wptr_off

        self._zero_vram(meta.mqd_off, MQD_SIZE)
        self._zero_vram(meta.ring_off, DIRECT_COMPUTE_RING_SIZE)
        self._zero_vram(meta.eop_off, DIRECT_COMPUTE_EOP_SIZE)
        self._zero_vram(meta.rptr_off, 0x20)
        self._zero_vram(meta.wptr_off, 0x20)

        self._activate_direct_compute_queue(meta)

        ring_buf = MemoryHandle(
            kfd_handle=-queue_id,
            gpu_addr=meta.ring_mc,
            cpu_addr=self._vram_addr + meta.ring_off,
            size=DIRECT_COMPUTE_RING_SIZE,
            location=MemoryLocation.VRAM,
            owner_gpu_id=0,
            mapped_gpu_ids=[0],
        )
        eop_buf = MemoryHandle(
            kfd_handle=-(queue_id + 0x10000),
            gpu_addr=meta.eop_mc,
            cpu_addr=self._vram_addr + meta.eop_off,
            size=DIRECT_COMPUTE_EOP_SIZE,
            location=MemoryLocation.VRAM,
            owner_gpu_id=0,
            mapped_gpu_ids=[0],
        )

        handle = QueueHandle(
            queue_id=queue_id,
            queue_type=QueueType.COMPUTE,
            ring_buffer=ring_buf,
            ring_size=DIRECT_COMPUTE_RING_SIZE,
            write_ptr_addr=self._vram_addr + meta.wptr_off,
            read_ptr_addr=self._vram_addr + meta.rptr_off,
            doorbell_offset=meta.doorbell_index * 4,
            doorbell_addr=self._doorbell_addr + meta.doorbell_index * 4,
            eop_buffer=eop_buf,
            ctx_save_restore=None,
        )

        self._queues[queue_id] = handle
        self._direct_compute[queue_id] = meta

        return handle

    def create_sdma_queue(self) -> QueueHandle:
        """Create an SDMA (DMA copy) queue."""
        queue_id = self._next_queue_id
        self._next_queue_id += 1

        ring_buf = self._memory.alloc(
            SDMA_RING_SIZE,
            MemoryLocation.GTT,
            uncached=True,
        )

        eop_buf = self._memory.alloc(
            EOP_BUFFER_SIZE,
            MemoryLocation.GTT,
            uncached=True,
        )

        ctypes.memset(ring_buf.cpu_addr, 0, SDMA_RING_SIZE)
        ctypes.memset(eop_buf.cpu_addr, 0, EOP_BUFFER_SIZE)

        doorbell_offset = self._alloc_doorbell()
        doorbell_addr = self._doorbell_addr + doorbell_offset

        handle = QueueHandle(
            queue_id=queue_id,
            queue_type=QueueType.SDMA,
            ring_buffer=ring_buf,
            ring_size=SDMA_RING_SIZE,
            write_ptr_addr=ring_buf.cpu_addr,
            read_ptr_addr=ring_buf.cpu_addr + 8,
            doorbell_offset=doorbell_offset,
            doorbell_addr=doorbell_addr,
            eop_buffer=eop_buf,
            ctx_save_restore=None,
        )

        self._queues[queue_id] = handle
        return handle

    def destroy_queue(self, handle: QueueHandle) -> None:
        """Destroy a hardware queue and free its resources."""
        self._queues.pop(handle.queue_id, None)
        self._direct_compute.pop(handle.queue_id, None)

        if handle.ring_buffer:
            self._memory.free(handle.ring_buffer)
        if handle.eop_buffer:
            self._memory.free(handle.eop_buffer)
        if handle.ctx_save_restore:
            self._memory.free(handle.ctx_save_restore)

    def submit_packets(self, queue: QueueHandle, packets: bytes) -> None:
        """Submit command packets to a queue's ring buffer.

        1. Write packets into ring buffer at current write pointer
        2. Advance write pointer (wrapping at ring_size)
        3. Write doorbell to notify GPU
        """
        if queue.queue_id in self._direct_compute:
            self._submit_direct_compute(queue, packets)
            return

        if queue.ring_buffer is None:
            raise RuntimeError("Queue has no ring buffer")

        ring_addr = queue.ring_buffer.cpu_addr
        ring_size = queue.ring_size

        # Read current write pointer (64-bit, in DWORDs)
        wptr = ctypes.c_uint64.from_address(queue.write_ptr_addr).value

        # Packet offset in bytes (write pointer is in DWORDs for PM4)
        if queue.queue_type == QueueType.COMPUTE:
            byte_offset = (wptr * 4) % ring_size
        else:
            byte_offset = wptr % ring_size

        # Write packets into ring buffer
        # Handle wrap-around
        remaining = ring_size - byte_offset
        if len(packets) <= remaining:
            ctypes.memmove(ring_addr + byte_offset, packets, len(packets))
        else:
            # Split across ring boundary
            ctypes.memmove(ring_addr + byte_offset, packets[:remaining], remaining)
            ctypes.memmove(ring_addr, packets[remaining:], len(packets) - remaining)

        # Advance write pointer
        if queue.queue_type == QueueType.COMPUTE:
            new_wptr = wptr + (len(packets) // 4)  # PM4: count in DWORDs
        else:
            new_wptr = wptr + len(packets)

        ctypes.c_uint64.from_address(queue.write_ptr_addr).value = new_wptr

        # Ring doorbell to notify GPU
        if queue.doorbell_addr:
            # Write lower 32 bits of write pointer to doorbell
            ctypes.c_uint32.from_address(queue.doorbell_addr).value = (
                new_wptr & 0xFFFFFFFF
            )

    def _submit_direct_compute(self, queue: QueueHandle, packets: bytes) -> None:
        """Submit PM4 packets to the VRAM-backed direct compute queue."""
        if queue.ring_buffer is None:
            raise RuntimeError("Queue has no ring buffer")
        if len(packets) % 4:
            raise ValueError("compute PM4 packet stream must be DWORD-aligned")
        meta = self._direct_compute.get(queue.queue_id)
        if meta is None:
            raise RuntimeError("missing direct compute queue metadata")

        ring_addr = queue.ring_buffer.cpu_addr
        ring_size = queue.ring_size
        # BAR0 CPU reads can return stale data on this eGPU mapping. Keep the
        # write pointer in host bookkeeping and only use BAR0 as GPU-visible
        # writeback storage.
        wptr = meta.wptr
        ring_dw = ring_size // 4
        start_dw = wptr % ring_dw

        # BAR0 VRAM tolerates DWORD stores reliably; larger libc memmove
        # copies can use access patterns that bus-error on this eGPU BAR.
        for i in range(0, len(packets), 4):
            word = int.from_bytes(packets[i:i + 4], "little")
            ring_off = ((start_dw + (i // 4)) % ring_dw) * 4
            (ctypes.c_uint32 * 1).from_address(ring_addr + ring_off)[0] = word

        new_wptr = wptr + len(packets) // 4
        ctypes.c_uint64.from_address(queue.write_ptr_addr).value = new_wptr

        # Doorbell/WPTR polling is not sufficient with the current macOS BAR
        # mappings. The proven path explicitly advances the selected HQD WPTR.
        self._select_hqd(meta.me, meta.pipe, meta.queue)
        active = self._gc0_rd(regCP_HQD_ACTIVE)
        if active == 0:
            self._gc1_wr(regGRBM_GFX_CNTL, 0)
            raise RuntimeError("direct macOS compute HQD is inactive at submit")
        self._gc0_wr(regCP_HQD_PQ_WPTR_LO, new_wptr & 0xFFFFFFFF)
        self._gc0_wr(regCP_HQD_PQ_WPTR_HI, (new_wptr >> 32) & 0xFFFFFFFF)
        self._gc1_wr(regGRBM_GFX_CNTL, 0)

        # Compute/MES doorbells on gfx12 are 64-bit wptr values at BAR2 +
        # doorbell_dword_index * 4.
        ctypes.c_uint64.from_address(queue.doorbell_addr).value = new_wptr
        meta.wptr = new_wptr

    def _ensure_bar_mappings(self) -> None:
        """Map VRAM BAR0 and doorbell BAR2 if bringup did not pre-map them."""
        if self._vram_addr == 0:
            self._vram_addr, self._vram_size = self._client.map_bar(_VRAM_BAR)
        if self._doorbell_addr == 0:
            self._doorbell_addr, self._doorbell_size = self._client.map_bar(DOORBELL_BAR_INDEX)

    def _framebuffer_base(self) -> int:
        """Return the MC base for BAR0 VRAM offsets."""
        return (
            self._client.mmio_read32(_MMIO_BAR, (0x1A000 + 0x0554) * 4)
            & 0xFFFFFF
        ) << 24

    def _rd(self, base: int, off: int) -> int:
        return self._client.mmio_read32(_MMIO_BAR, (base + off) * 4)

    def _wr(self, base: int, off: int, value: int) -> None:
        self._client.mmio_write32(_MMIO_BAR, (base + off) * 4, value & 0xFFFFFFFF)

    def _gc0_rd(self, off: int) -> int:
        return self._rd(GC_B0, off)

    def _gc0_wr(self, off: int, value: int) -> None:
        self._wr(GC_B0, off, value)

    def _gc1_wr(self, off: int, value: int) -> None:
        self._wr(GC_B1, off, value)

    def _select_hqd(self, me: int, pipe: int, queue: int, vmid: int = 0) -> None:
        self._gc1_wr(
            regGRBM_GFX_CNTL,
            ((pipe & 0x3) << 0)
            | ((me & 0x3) << 2)
            | ((vmid & 0xF) << 4)
            | ((queue & 0x7) << 8),
        )

    def _vram_wr32(self, off: int, value: int) -> None:
        (ctypes.c_uint32 * 1).from_address(self._vram_addr + off)[0] = (
            value & 0xFFFFFFFF
        )

    def _zero_vram(self, off: int, size: int) -> None:
        for i in range(0, size, 4):
            self._vram_wr32(off + i, 0)

    def _ensure_doorbell_aperture(self) -> None:
        """Program the minimal NBIO/CP doorbell aperture used by gfx12 queues."""
        self._wr(NBIO_B2, regRCC_DEV0_EPF0_RCC_DOORBELL_APER_EN, 1)

        # Port 0: enable=1 awid=3 awaddr_31_28=3.
        self._wr(
            NBIO_B2,
            regGDC_S2A0_S2A_DOORBELL_ENTRY_0_CTRL,
            (1 << 0) | (3 << 1) | (3 << 28),
        )
        # Port 3: enable=1 awid=6 awaddr_31_28=3.
        self._wr(
            NBIO_B2,
            regGDC_S2A0_S2A_DOORBELL_ENTRY_3_CTRL,
            (1 << 0) | (6 << 1) | (3 << 28),
        )

        self._gc0_wr(regCP_MEC_DOORBELL_RANGE_LOWER, 0)
        self._gc0_wr(regCP_MEC_DOORBELL_RANGE_UPPER, (0x8A * 2) << 2)

    def _activate_direct_compute_queue(self, meta: _DirectComputeQueue) -> None:
        """Build a v12 compute MQD in VRAM and program its HQD registers."""
        self._select_hqd(meta.me, meta.pipe, meta.queue)
        pre_active = self._gc0_rd(regCP_HQD_ACTIVE)
        force = os.environ.get("AMD_GPU_MACOS_FORCE_DIRECT_COMPUTE") == "1"
        if pre_active and not force:
            raise RuntimeError(
                "target compute HQD is already active; set "
                "AMD_GPU_MACOS_FORCE_DIRECT_COMPUTE=1 to overwrite it"
            )

        if pre_active:
            self._gc0_wr(regCP_HQD_DEQUEUE_REQUEST, 1)
            deadline = time.time() + 1
            while time.time() < deadline and self._gc0_rd(regCP_HQD_ACTIVE):
                time.sleep(0.001)
            self._gc0_wr(regCP_HQD_DEQUEUE_REQUEST, 0)

        mqd = [0] * (MQD_SIZE // 4)
        mqd[0] = 0xC0310800
        mqd[1] = 1
        for dw in (0x17, 0x18, 0x1A, 0x1B):
            mqd[dw] = 0xFFFFFFFF
        mqd[0x2C] = 7

        eop_base_shifted = meta.eop_mc >> 8
        mqd[0xA5] = eop_base_shifted & 0xFFFFFFFF
        mqd[0xA6] = (eop_base_shifted >> 32) & 0xFFFFFFFF
        mqd[0xA7] = ((DIRECT_COMPUTE_EOP_SIZE // 4).bit_length() - 2) & 0x3F

        mqd[0x80] = meta.mqd_mc & 0xFFFFFFFC
        mqd[0x81] = (meta.mqd_mc >> 32) & 0xFFFFFFFF
        mqd[0x82] = 1
        mqd[0x83] = 0
        mqd[0x84] = (CP_HQD_PERSISTENT_STATE_DEFAULT & ~(0x3FF << 8)) | (0x55 << 8)

        pq_base_shifted = meta.ring_mc >> 8
        mqd[0x88] = pq_base_shifted & 0xFFFFFFFF
        mqd[0x89] = (pq_base_shifted >> 32) & 0xFFFFFFFF
        mqd[0x8B] = meta.rptr_mc & 0xFFFFFFFC
        mqd[0x8C] = (meta.rptr_mc >> 32) & 0xFFFF
        mqd[0x8D] = meta.wptr_mc & 0xFFFFFFF8
        mqd[0x8E] = (meta.wptr_mc >> 32) & 0xFFFF
        mqd[0x8F] = ((meta.doorbell_index & 0x3FFFFFF) << 2) | (1 << 30)

        ring_dw = DIRECT_COMPUTE_RING_SIZE // 4
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
            self._vram_wr32(meta.mqd_off + i * 4, value)

        self._gc0_wr(regCP_HQD_ACTIVE, 0)
        self._gc0_wr(regCP_HQD_PQ_RPTR, 0)
        self._gc0_wr(regCP_HQD_PQ_WPTR_LO, 0)
        self._gc0_wr(regCP_HQD_PQ_WPTR_HI, 0)
        self._gc0_wr(regCP_HQD_VMID, self._gc0_rd(regCP_HQD_VMID) & ~0xF)
        self._gc0_wr(
            regCP_HQD_PQ_DOORBELL_CONTROL,
            self._gc0_rd(regCP_HQD_PQ_DOORBELL_CONTROL) & ~0x40000000,
        )
        self._gc0_wr(regCP_MQD_BASE_ADDR, mqd[0x80])
        self._gc0_wr(regCP_MQD_BASE_ADDR_HI, mqd[0x81])
        self._gc0_wr(regCP_MQD_CONTROL, 0)
        self._gc0_wr(regCP_HQD_EOP_BASE_ADDR, mqd[0xA5])
        self._gc0_wr(regCP_HQD_EOP_BASE_ADDR_HI, mqd[0xA6])
        self._gc0_wr(regCP_HQD_EOP_CONTROL, mqd[0xA7])
        self._gc0_wr(regCP_HQD_PQ_BASE, mqd[0x88])
        self._gc0_wr(regCP_HQD_PQ_BASE_HI, mqd[0x89])
        self._gc0_wr(regCP_HQD_PQ_RPTR_REPORT_ADDR, mqd[0x8B])
        self._gc0_wr(regCP_HQD_PQ_RPTR_REPORT_ADDR_HI, mqd[0x8C])
        self._gc0_wr(regCP_HQD_PQ_CONTROL, mqd[0x91])
        self._gc0_wr(regCP_HQD_PQ_WPTR_POLL_ADDR, mqd[0x8D])
        self._gc0_wr(regCP_HQD_PQ_WPTR_POLL_ADDR_HI, mqd[0x8E])
        self._gc0_wr(regCP_HQD_PQ_DOORBELL_CONTROL, mqd[0x8F])
        self._gc0_wr(regCP_HQD_PERSISTENT_STATE, mqd[0x84])
        self._gc0_wr(regCP_HQD_ACTIVE, 1)
        time.sleep(0.010)
        post_active = self._gc0_rd(regCP_HQD_ACTIVE)
        self._gc1_wr(regGRBM_GFX_CNTL, 0)
        if post_active == 0:
            raise RuntimeError(
                "direct macOS compute HQD did not become active after programming"
            )

    def _alloc_doorbell(self) -> int:
        """Allocate a doorbell slot. Returns BAR-relative offset."""
        if self._doorbell_addr == 0:
            raise RuntimeError(
                "Doorbell BAR not mapped — call set_doorbell_bar() during bringup"
            )

        offset = self._next_doorbell_offset
        self._next_doorbell_offset += DOORBELL_STRIDE

        if offset >= self._doorbell_size:
            raise RuntimeError("Doorbell slots exhausted")

        return offset
