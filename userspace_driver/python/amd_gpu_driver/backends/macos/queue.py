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
import struct
import time

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

    def set_doorbell_bar(self, addr: int, size: int) -> None:
        """Set the doorbell BAR mapping (called during bringup)."""
        self._doorbell_addr = addr
        self._doorbell_size = size

    def create_compute_queue(self) -> QueueHandle:
        """Create a compute queue.

        Allocates:
          - Ring buffer (256KB GTT) for PM4 command packets
          - EOP buffer (4KB GTT) for end-of-pipe signaling
          - Doorbell slot (8 bytes in BAR2)

        Then constructs an MQD and programs it into the CP.
        """
        queue_id = self._next_queue_id
        self._next_queue_id += 1

        # Allocate ring buffer
        ring_buf = self._memory.alloc(
            COMPUTE_RING_SIZE,
            MemoryLocation.GTT,
            uncached=True,
        )

        # Allocate EOP buffer
        eop_buf = self._memory.alloc(
            EOP_BUFFER_SIZE,
            MemoryLocation.GTT,
            uncached=True,
        )

        # Zero-init ring and EOP
        ctypes.memset(ring_buf.cpu_addr, 0, COMPUTE_RING_SIZE)
        ctypes.memset(eop_buf.cpu_addr, 0, EOP_BUFFER_SIZE)

        # Allocate doorbell slot
        doorbell_offset = self._alloc_doorbell()
        doorbell_addr = self._doorbell_addr + doorbell_offset

        # Write pointer is stored at the beginning of ring buffer
        # (or in a separate GTT allocation — simplified here)
        # For MQD-based queues, the write/read pointers are in the MQD
        wptr_addr = ring_buf.cpu_addr  # First 8 bytes = write ptr
        rptr_addr = ring_buf.cpu_addr + 8  # Next 8 bytes = read ptr

        handle = QueueHandle(
            queue_id=queue_id,
            queue_type=QueueType.COMPUTE,
            ring_buffer=ring_buf,
            ring_size=COMPUTE_RING_SIZE,
            write_ptr_addr=wptr_addr,
            read_ptr_addr=rptr_addr,
            doorbell_offset=doorbell_offset,
            doorbell_addr=doorbell_addr,
            eop_buffer=eop_buf,
            ctx_save_restore=None,
        )

        self._queues[queue_id] = handle

        # TODO: Construct MQD and program into CP registers
        # This requires the GPU bringup (GMC, PSP, MES) to be complete.
        # The MQD structure is GPU-generation-specific and will reuse
        # the register definitions from the Windows backend.

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
