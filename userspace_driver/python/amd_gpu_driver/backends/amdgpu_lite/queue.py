"""Compute and SDMA queue management for amdgpu_lite.

Unlike KFD where the kernel creates and manages queues via MES/HWS,
amdgpu_lite queues are constructed entirely in userspace:
  1. Allocate ring buffer, EOP buffer, write/read pointers in GTT
  2. Build the MQD (Memory Queue Descriptor) in userspace
  3. Map the doorbell BAR for CPU writes
  4. Submit PM4 packets directly to the ring buffer
  5. Ring the doorbell to notify the GPU

The MQD construction and queue initialization use the same register
programming as the Python firmware loading code (MES/MEC bring-up).
"""

from __future__ import annotations

import ctypes
import struct

from amd_gpu_driver.backends.base import MemoryHandle, MemoryLocation, QueueHandle, QueueType
from amd_gpu_driver.backends.amdgpu_lite.memory import LiteMemoryManager
from amd_gpu_driver.errors import QueueError

DEFAULT_RING_SIZE = 256 * 1024  # 256KB
DEFAULT_EOP_SIZE = 4096


class LiteQueueManager:
    """Manages compute and SDMA queues via amdgpu_lite.

    Queue creation is done entirely in userspace — the kernel module only
    provides memory allocation and BAR access. MQD construction, doorbell
    assignment, and ring management happen here.
    """

    def __init__(
        self,
        fd: int,
        memory: LiteMemoryManager,
        doorbell_addr: int,
        doorbell_size: int,
    ) -> None:
        self._fd = fd
        self._memory = memory
        self._doorbell_addr = doorbell_addr
        self._doorbell_size = doorbell_size
        self._queues: list[QueueHandle] = []
        self._next_doorbell_offset = 0

    def create_compute_queue(
        self, ring_size: int = DEFAULT_RING_SIZE
    ) -> QueueHandle:
        """Create a compute queue with ring buffer and doorbell."""
        return self._create_queue(QueueType.COMPUTE, ring_size)

    def create_sdma_queue(
        self, ring_size: int = DEFAULT_RING_SIZE
    ) -> QueueHandle:
        """Create an SDMA queue."""
        return self._create_queue(QueueType.SDMA, ring_size)

    def _create_queue(self, queue_type: QueueType, ring_size: int) -> QueueHandle:
        """Create a hardware queue.

        Allocates ring buffer, EOP buffer, and write/read pointer memory
        in GTT. Assigns a doorbell slot from the mmap'd doorbell BAR.
        """
        # Allocate ring buffer in GTT (DMA-coherent for CPU/GPU access)
        ring_buffer = self._memory.alloc(
            ring_size, MemoryLocation.GTT, map_gpu=True
        )

        # Allocate EOP buffer
        eop_buffer = self._memory.alloc(
            DEFAULT_EOP_SIZE, MemoryLocation.GTT, map_gpu=True
        )

        # Allocate write/read pointer page (8 bytes each)
        wr_ptr_mem = self._memory.alloc(
            4096, MemoryLocation.GTT, map_gpu=True
        )
        write_ptr_addr = wr_ptr_mem.cpu_addr
        read_ptr_addr = wr_ptr_mem.cpu_addr + 8

        # Zero the write/read pointers
        if wr_ptr_mem.cpu_addr:
            ctypes.memset(wr_ptr_mem.cpu_addr, 0, 16)

        # Assign doorbell offset (8 bytes per doorbell for GFX9+)
        doorbell_offset = self._next_doorbell_offset
        self._next_doorbell_offset += 8
        doorbell_addr = 0
        if self._doorbell_addr and doorbell_offset < self._doorbell_size:
            doorbell_addr = self._doorbell_addr + doorbell_offset

        handle = QueueHandle(
            queue_id=len(self._queues),
            queue_type=queue_type,
            ring_buffer=ring_buffer,
            ring_size=ring_size,
            write_ptr_addr=write_ptr_addr,
            read_ptr_addr=read_ptr_addr,
            doorbell_offset=doorbell_offset,
            doorbell_addr=doorbell_addr,
            eop_buffer=eop_buffer,
        )
        self._queues.append(handle)
        return handle

    def submit(self, queue: QueueHandle, packets: bytes) -> None:
        """Submit packets to queue ring buffer and ring doorbell.

        Compute queues use dword-based write pointers and 64-bit doorbells.
        SDMA queues use byte-based write pointers and 32-bit doorbells.
        """
        ring = queue.ring_buffer
        if ring is None or ring.cpu_addr == 0:
            raise QueueError("Queue ring buffer not mapped")

        is_sdma = queue.queue_type in (QueueType.SDMA, QueueType.SDMA_XGMI)
        packet_bytes = len(packets)

        # Read current write pointer
        wp_val = ctypes.c_uint64.from_address(queue.write_ptr_addr).value
        ring_mask = queue.ring_size - 1

        if is_sdma:
            offset = wp_val & ring_mask
        else:
            offset = (wp_val * 4) & ring_mask

        ring_base = ring.cpu_addr

        # Copy packets to ring (handle wrap-around)
        space_to_end = queue.ring_size - offset
        if packet_bytes <= space_to_end:
            ctypes.memmove(ring_base + offset, packets, packet_bytes)
        else:
            ctypes.memmove(ring_base + offset, packets[:space_to_end], space_to_end)
            remainder = packet_bytes - space_to_end
            ctypes.memmove(ring_base, packets[space_to_end:], remainder)

        # Update write pointer and ring doorbell
        if is_sdma:
            new_wp = wp_val + packet_bytes
            ctypes.c_uint64.from_address(queue.write_ptr_addr).value = new_wp
            if queue.doorbell_addr:
                ctypes.c_uint32.from_address(queue.doorbell_addr).value = (
                    new_wp & 0xFFFFFFFF
                )
        else:
            new_wp = wp_val + (packet_bytes // 4)
            ctypes.c_uint64.from_address(queue.write_ptr_addr).value = new_wp
            if queue.doorbell_addr:
                ctypes.c_uint64.from_address(queue.doorbell_addr).value = new_wp

    def destroy_queue(self, handle: QueueHandle) -> None:
        """Destroy a queue (free its memory allocations)."""
        if handle.ring_buffer:
            self._memory.free(handle.ring_buffer)
        if handle.eop_buffer:
            self._memory.free(handle.eop_buffer)
        if handle in self._queues:
            self._queues.remove(handle)

    def destroy_all(self) -> None:
        """Destroy all tracked queues."""
        for q in list(self._queues):
            self.destroy_queue(q)
