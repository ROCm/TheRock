"""macOS memory manager — DMA and VRAM allocation via DEXT.

Memory model:
  GTT (system memory):
    - Allocated via DEXT's AllocDMA (IOBufferMemoryDescriptor + IODMACommand)
    - IOMMU-translated physical addresses returned for GPU page tables
    - CPU-mapped into userspace for read/write
    - Used for: ring buffers, EOP buffers, command packets, kernel args

  VRAM:
    - Accessed via BAR mapping (MAP_BAR for BAR0/BAR1)
    - For RDNA4: BAR0 = MMIO registers, BAR1 or resizable BAR0 = VRAM
    - GPU page tables map virtual addresses to BAR-relative offsets
    - Used for: kernel code, large data buffers

  GPU Page Tables:
    - Written by Python via MMIO (same as Windows backend)
    - PTE/PDE entries reference IOMMU physical addresses (from DMA alloc)
    - Managed by the GMC init code (shared with Windows backend)
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field

from amd_gpu_driver.backends.base import MemoryHandle, MemoryLocation
from amd_gpu_driver.backends.macos.iokit_client import DMAAllocation, IOKitClient

_MMIO_BAR = 5
_VRAM_BAR = 0
_MMHUB_B0 = 0x1A000
_REG_MMC_VM_FB_LOCATION_BASE = 0x0554


@dataclass
class MacOSMemoryAllocation:
    """Extended memory allocation info for the macOS backend."""
    handle: MemoryHandle
    dma: DMAAllocation | None = None  # Set for GTT allocations
    bar_offset: int = 0               # Set for VRAM allocations (BAR-relative)


class MacOSMemoryManager:
    """Manages GPU-accessible memory on macOS.

    Wraps the DEXT's DMA allocation and BAR mapping to provide
    the MemoryHandle interface expected by DeviceBackend.
    """

    def __init__(self, client: IOKitClient, gpu_id: int) -> None:
        self._client = client
        self._gpu_id = gpu_id
        self._allocations: dict[int, MacOSMemoryAllocation] = {}
        self._next_handle_id = 1

        # VRAM state (populated during bringup)
        self._vram_bar_addr: int = 0    # CPU virtual address of VRAM BAR
        self._vram_bar_size: int = 0    # Size of mapped VRAM region
        self._vram_mc_base: int = 0      # GPU MC address for BAR offset 0
        self._vram_offset: int = 0      # Next free offset in VRAM BAR

    def set_vram_bar(self, addr: int, size: int, mc_base: int | None = None) -> None:
        """Set the VRAM BAR mapping (called during bringup)."""
        self._vram_bar_addr = addr
        self._vram_bar_size = size
        self._vram_mc_base = self._read_framebuffer_base() if mc_base is None else mc_base
        # Reserve the low scratch range used by the current bring-up probes
        # and direct compute prototype.
        self._vram_offset = 32 * 1024 * 1024

    def alloc(
        self,
        size: int,
        location: MemoryLocation,
        *,
        executable: bool = False,
        public: bool = False,
        uncached: bool = False,
    ) -> MemoryHandle:
        """Allocate GPU-accessible memory."""
        # Round up to page size
        page_size = 4096
        size = (size + page_size - 1) & ~(page_size - 1)

        handle_id = self._next_handle_id
        self._next_handle_id += 1

        if location == MemoryLocation.GTT:
            return self._alloc_gtt(handle_id, size, uncached)
        elif location == MemoryLocation.VRAM:
            return self._alloc_vram(handle_id, size)
        else:
            raise ValueError(f"Unsupported memory location: {location}")

    def _alloc_gtt(self, handle_id: int, size: int, uncached: bool) -> MemoryHandle:
        """Allocate system memory (GTT) via DEXT DMA allocation."""
        flags = 0
        if uncached:
            flags |= 0x2  # kROCmGPU_DMA_Uncached

        dma = self._client.alloc_dma(size, flags)

        handle = MemoryHandle(
            kfd_handle=handle_id,
            gpu_addr=dma.segments[0][0] if dma.segments else 0,
            cpu_addr=dma.cpu_addr,
            size=size,
            location=MemoryLocation.GTT,
            flags=flags,
            owner_gpu_id=self._gpu_id,
            mapped_gpu_ids=[self._gpu_id],
        )

        self._allocations[handle_id] = MacOSMemoryAllocation(
            handle=handle,
            dma=dma,
        )

        return handle

    def _alloc_vram(self, handle_id: int, size: int) -> MemoryHandle:
        """Allocate VRAM via BAR mapping.

        Simple bump allocator for VRAM. Real implementation would
        need a proper allocator with free list.
        """
        self._ensure_vram_bar()
        if self._vram_bar_addr == 0:
            raise RuntimeError(
                "VRAM BAR not mapped — call set_vram_bar() during bringup"
            )

        if self._vram_offset + size > self._vram_bar_size:
            raise MemoryError(
                f"VRAM exhausted: requested {size}, "
                f"available {self._vram_bar_size - self._vram_offset}"
            )

        offset = self._vram_offset
        self._vram_offset += size

        # CPU address = BAR base + offset
        cpu_addr = self._vram_bar_addr + offset

        handle = MemoryHandle(
            kfd_handle=handle_id,
            gpu_addr=self._vram_mc_base + offset,
            cpu_addr=cpu_addr,
            size=size,
            location=MemoryLocation.VRAM,
            flags=0,
            owner_gpu_id=self._gpu_id,
            mapped_gpu_ids=[self._gpu_id],
        )

        self._allocations[handle_id] = MacOSMemoryAllocation(
            handle=handle,
            bar_offset=offset,
        )

        return handle

    def _ensure_vram_bar(self) -> None:
        """Map BAR0 lazily for code/data allocations if bringup did not."""
        if self._vram_bar_addr:
            return
        addr, size = self._client.map_bar(_VRAM_BAR)
        self.set_vram_bar(addr, size)

    def _read_framebuffer_base(self) -> int:
        return (
            self._client.mmio_read32(
                _MMIO_BAR,
                (_MMHUB_B0 + _REG_MMC_VM_FB_LOCATION_BASE) * 4,
            )
            & 0xFFFFFF
        ) << 24

    def free(self, handle: MemoryHandle) -> None:
        """Free a previously allocated memory region."""
        alloc = self._allocations.pop(handle.kfd_handle, None)
        if alloc is None:
            return

        if alloc.dma is not None:
            self._client.free_dma(alloc.dma.buffer_id)
        # VRAM deallocation is a no-op for bump allocator
        # (would need a real free list for production use)

    def get_phys_addr(self, handle: MemoryHandle) -> int:
        """Get the IOMMU-translated physical address for a handle.

        For GTT: returns the DMA segment address
        For VRAM: returns the BAR-relative address (needs page table setup)
        """
        alloc = self._allocations.get(handle.kfd_handle)
        if alloc is None:
            raise ValueError(f"Unknown memory handle: {handle.kfd_handle}")

        if alloc.dma is not None and alloc.dma.segments:
            return alloc.dma.segments[0][0]
        return handle.gpu_addr

    def get_scatter_gather(self, handle: MemoryHandle) -> list[tuple[int, int]]:
        """Get scatter-gather list for a GTT allocation.

        Returns list of (phys_addr, length) tuples for GPU page table entries.
        """
        alloc = self._allocations.get(handle.kfd_handle)
        if alloc is None or alloc.dma is None:
            raise ValueError("Not a GTT allocation")
        return list(alloc.dma.segments)

    def read(self, handle: MemoryHandle, offset: int, size: int) -> bytes:
        """Read bytes from a mapped memory region."""
        addr = handle.cpu_addr + offset
        return ctypes.string_at(addr, size)

    def write(self, handle: MemoryHandle, offset: int, data: bytes) -> None:
        """Write bytes to a mapped memory region."""
        addr = handle.cpu_addr + offset
        ctypes.memmove(addr, data, len(data))
