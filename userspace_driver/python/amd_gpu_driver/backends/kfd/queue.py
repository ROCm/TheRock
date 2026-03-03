"""KFD compute and SDMA queue creation."""

from __future__ import annotations

import ctypes
import struct

from amd_gpu_driver.backends.base import MemoryHandle, MemoryLocation, QueueHandle, QueueType
from amd_gpu_driver.backends.kfd.memory import KFDMemoryManager
from amd_gpu_driver.errors import QueueError
from amd_gpu_driver.gpu.family import GPUFamilyConfig
from amd_gpu_driver.ioctl import helpers
from amd_gpu_driver.ioctl.kfd import (
    AMDKFD_IOC_CREATE_QUEUE,
    AMDKFD_IOC_DESTROY_QUEUE,
    KFD_IOC_QUEUE_TYPE_COMPUTE,
    KFD_IOC_QUEUE_TYPE_SDMA,
    KFD_IOC_QUEUE_TYPE_SDMA_XGMI,
    KFD_MAX_QUEUE_PERCENTAGE,
    KFD_MAX_QUEUE_PRIORITY,
    kfd_ioctl_create_queue_args,
    kfd_ioctl_destroy_queue_args,
)
from amd_gpu_driver.topology import GPUNode, compute_queue_sizes

# Default ring buffer size: 256KB (power of 2)
DEFAULT_RING_SIZE = 256 * 1024

# Default EOP buffer size
DEFAULT_EOP_SIZE = 4096

# Control stack size
DEFAULT_CTL_STACK_SIZE = 0x1000


class KFDQueueManager:
    """Manages KFD queue creation and submission."""

    def __init__(
        self,
        kfd_fd: int,
        gpu_id: int,
        memory: KFDMemoryManager,
        family: GPUFamilyConfig,
        node: GPUNode | None = None,
    ) -> None:
        self._kfd_fd = kfd_fd
        self._gpu_id = gpu_id
        self._memory = memory
        self._family = family
        self._node = node
        self._queues: list[QueueHandle] = []
        self._doorbell_page_addr: int = 0
        self._doorbell_page_size: int = 0
        # Pre-compute queue sizes from topology (replicates kernel logic)
        if node is not None:
            self._queue_sizes = compute_queue_sizes(node)
        else:
            self._queue_sizes = {
                "ctl_stack_size": DEFAULT_CTL_STACK_SIZE,
                "cwsr_size": DEFAULT_CTL_STACK_SIZE,
                "eop_buffer_size": DEFAULT_EOP_SIZE,
                "debug_memory_size": 0,
            }

    def create_compute_queue(
        self,
        ring_size: int = DEFAULT_RING_SIZE,
    ) -> QueueHandle:
        """Create a compute queue."""
        return self._create_queue(QueueType.COMPUTE, ring_size)

    def create_sdma_queue(
        self,
        ring_size: int = DEFAULT_RING_SIZE,
    ) -> QueueHandle:
        """Create an SDMA queue."""
        return self._create_queue(QueueType.SDMA, ring_size)

    def create_xgmi_sdma_queue(
        self,
        ring_size: int = DEFAULT_RING_SIZE,
    ) -> QueueHandle:
        """Create an XGMI SDMA queue for cross-GPU copies."""
        return self._create_queue(QueueType.SDMA_XGMI, ring_size)

    def _create_queue(self, queue_type: QueueType, ring_size: int) -> QueueHandle:
        """Create a hardware queue via AMDKFD_IOC_CREATE_QUEUE."""
        # Get kernel-matching queue sizes
        qs = self._queue_sizes
        eop_size = qs["eop_buffer_size"]
        ctl_stack_size = qs["ctl_stack_size"]
        cwsr_size = qs["cwsr_size"]
        debug_memory_size = qs["debug_memory_size"]

        # The ioctl ctx_save_restore_size must be >= cwsr_size.
        # The kernel then computes the total buffer requirement as:
        #   total = (ctx_save_restore_size + debug_memory_size) * NUM_XCC
        # So we pass cwsr_size as ctx_save_restore_size and allocate enough
        # for the kernel's total computation.
        num_xcc = self._node.num_xcc if self._node else 1
        num_xcc = max(num_xcc, 1)
        ctx_save_ioctl_size = cwsr_size
        ctx_save_alloc_size = (cwsr_size + debug_memory_size) * num_xcc
        ctx_save_alloc_size = (ctx_save_alloc_size + 4095) & ~4095  # page-align

        # Allocate ring buffer (GTT, uncached for CPU/GPU coherence)
        ring_buffer = self._memory.alloc(
            ring_size, MemoryLocation.GTT, uncached=True
        )

        # Allocate EOP buffer
        eop_buffer = self._memory.alloc(
            max(eop_size, 4096), MemoryLocation.GTT, uncached=True
        )

        # Allocate ctx save/restore area (must be large enough for kernel's
        # total = (ctx_save_restore_size + debug_memory_size) * NUM_XCC)
        ctx_save = self._memory.alloc(
            max(ctx_save_alloc_size, 4096), MemoryLocation.GTT, uncached=True
        )

        # Allocate write/read pointer memory (8 bytes each, in GTT)
        wr_ptr_mem = self._memory.alloc(
            4096,  # Page-aligned minimum allocation
            MemoryLocation.GTT,
            uncached=True,
        )
        # We use the first 8 bytes as write_ptr and the next 8 as read_ptr
        write_ptr_addr = wr_ptr_mem.gpu_addr
        read_ptr_addr = wr_ptr_mem.gpu_addr + 8

        # Zero the write/read pointers
        if wr_ptr_mem.cpu_addr:
            ctypes.memset(wr_ptr_mem.cpu_addr, 0, 16)

        # Set up the ioctl args
        args = kfd_ioctl_create_queue_args()
        args.ring_base_address = ring_buffer.gpu_addr
        args.write_pointer_address = write_ptr_addr
        args.read_pointer_address = read_ptr_addr
        args.ring_size = ring_size
        args.gpu_id = self._gpu_id
        if queue_type == QueueType.COMPUTE:
            args.queue_type = KFD_IOC_QUEUE_TYPE_COMPUTE
        elif queue_type == QueueType.SDMA_XGMI:
            args.queue_type = KFD_IOC_QUEUE_TYPE_SDMA_XGMI
        else:
            args.queue_type = KFD_IOC_QUEUE_TYPE_SDMA
        args.queue_percentage = KFD_MAX_QUEUE_PERCENTAGE
        args.queue_priority = KFD_MAX_QUEUE_PRIORITY
        args.eop_buffer_address = eop_buffer.gpu_addr
        args.eop_buffer_size = eop_size
        args.ctx_save_restore_address = ctx_save.gpu_addr
        args.ctx_save_restore_size = ctx_save_ioctl_size
        args.ctl_stack_size = ctl_stack_size

        try:
            helpers.ioctl(
                self._kfd_fd, AMDKFD_IOC_CREATE_QUEUE, args, "CREATE_QUEUE"
            )
        except Exception as e:
            raise QueueError(f"Failed to create {queue_type.value} queue: {e}") from e

        # mmap the doorbell page from KFD
        doorbell_offset = args.doorbell_offset
        doorbell_addr = self._map_doorbell(doorbell_offset)

        handle = QueueHandle(
            queue_id=args.queue_id,
            queue_type=queue_type,
            ring_buffer=ring_buffer,
            ring_size=ring_size,
            write_ptr_addr=write_ptr_addr,
            read_ptr_addr=read_ptr_addr,
            doorbell_offset=doorbell_offset,
            doorbell_addr=doorbell_addr,
            eop_buffer=eop_buffer,
            ctx_save_restore=ctx_save,
        )
        self._queues.append(handle)
        return handle

    def _map_doorbell(self, doorbell_offset: int) -> int:
        """Map the doorbell page from KFD fd for CPU doorbell writes.

        The doorbell_offset from CREATE_QUEUE encodes:
        - Bits 63-62: KFD_MMAP_TYPE_DOORBELL (0x3)
        - Bits 61-46: GPU ID hash
        - Lower bits: doorbell_offset_in_process (on SOC15)

        We mmap the entire process doorbell slice (8KB for SOC15 with
        8-byte doorbells * 1024 max queues, 4KB for older with 4-byte
        doorbells * 1024 max queues). The mmap offset strips the
        per-queue doorbell offset. The returned address includes it.
        """
        # Doorbell process slice size depends on doorbell_size:
        # SOC15 (GFX9+): 8 bytes * 1024 = 8192 = 0x2000
        # Older: 4 bytes * 1024 = 4096 = 0x1000
        if self._family and self._family.gfx_version[0] >= 9:
            db_process_slice = 8192  # 8-byte doorbells
        else:
            db_process_slice = 4096  # 4-byte doorbells

        # If we already mapped this doorbell page, reuse it
        if self._doorbell_page_addr:
            within_slice = doorbell_offset & (db_process_slice - 1)
            return self._doorbell_page_addr + within_slice

        # Strip per-queue offset to get the mmap base offset
        mmap_offset = doorbell_offset & ~(db_process_slice - 1)
        within_slice = doorbell_offset & (db_process_slice - 1)

        addr = helpers.libc_mmap(
            None,
            db_process_slice,
            helpers.PROT_READ | helpers.PROT_WRITE,
            helpers.MAP_SHARED,
            self._kfd_fd,
            mmap_offset,
        )
        self._doorbell_page_addr = addr
        self._doorbell_page_size = db_process_slice
        return addr + within_slice

    def submit(self, queue: QueueHandle, packets: bytes) -> None:
        """Submit packets to queue ring buffer and ring doorbell.

        Compute queues use dword-based write pointers and doorbells.
        SDMA queues use byte-based write pointers and doorbells.
        """
        ring = queue.ring_buffer
        if ring is None or ring.cpu_addr == 0:
            raise QueueError("Queue ring buffer not mapped")

        is_sdma = queue.queue_type in (QueueType.SDMA, QueueType.SDMA_XGMI)
        packet_bytes = len(packets)

        # Read current write pointer
        wp_val = ctypes.c_uint64.from_address(queue.write_ptr_addr).value
        ring_mask = queue.ring_size - 1  # ring_size is power of 2

        if is_sdma:
            # SDMA: write pointer is in bytes
            offset = wp_val & ring_mask
        else:
            # Compute: write pointer is in dwords
            offset = (wp_val * 4) & ring_mask

        ring_base = ring.cpu_addr

        # Copy packets to ring (handle wrap-around)
        space_to_end = queue.ring_size - offset
        if packet_bytes <= space_to_end:
            ctypes.memmove(ring_base + offset, packets, packet_bytes)
        else:
            # Wrap around
            ctypes.memmove(ring_base + offset, packets[:space_to_end], space_to_end)
            remainder = packet_bytes - space_to_end
            ctypes.memmove(ring_base, packets[space_to_end:], remainder)

        # Update write pointer and ring doorbell
        if is_sdma:
            # SDMA: byte-based write pointer, 32-bit doorbell
            new_wp = wp_val + packet_bytes
            ctypes.c_uint64.from_address(queue.write_ptr_addr).value = new_wp
            if queue.doorbell_addr:
                ctypes.c_uint32.from_address(queue.doorbell_addr).value = new_wp & 0xFFFFFFFF
        else:
            # Compute: dword-based write pointer, 64-bit doorbell
            new_wp = wp_val + (packet_bytes // 4)
            ctypes.c_uint64.from_address(queue.write_ptr_addr).value = new_wp
            if queue.doorbell_addr:
                ctypes.c_uint64.from_address(queue.doorbell_addr).value = new_wp

    def destroy_queue(self, handle: QueueHandle) -> None:
        """Destroy a hardware queue."""
        args = kfd_ioctl_destroy_queue_args()
        args.queue_id = handle.queue_id
        try:
            helpers.ioctl(
                self._kfd_fd, AMDKFD_IOC_DESTROY_QUEUE, args, "DESTROY_QUEUE"
            )
        except Exception:
            pass
        if handle in self._queues:
            self._queues.remove(handle)

    def destroy_all(self) -> None:
        """Destroy all tracked queues."""
        for q in list(self._queues):
            self.destroy_queue(q)
