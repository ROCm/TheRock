"""Memory management via amdgpu_lite ioctls.

Unlike KFD which uses separate alloc/map ioctls through /dev/kfd + DRM,
amdgpu_lite handles everything through a single device fd. The kernel module
manages DMA-coherent GTT allocations, VRAM bitmap allocation, and GART
page table entries directly.
"""

from __future__ import annotations

import ctypes

from dataclasses import dataclass

from amd_gpu_driver.backends.base import MemoryHandle, MemoryLocation
from amd_gpu_driver.errors import MemoryAllocationError
from amd_gpu_driver.ioctl import helpers
from amd_gpu_driver.ioctl.amdgpu_lite import (
    AMDGPU_LITE_IOC_ALLOC_GTT,
    AMDGPU_LITE_IOC_ALLOC_VRAM,
    AMDGPU_LITE_IOC_FREE_GTT,
    AMDGPU_LITE_IOC_FREE_VRAM,
    AMDGPU_LITE_IOC_MAP_GPU,
    AMDGPU_LITE_IOC_UNMAP_GPU,
    amdgpu_lite_alloc_gtt,
    amdgpu_lite_alloc_vram,
    amdgpu_lite_free_gtt,
    amdgpu_lite_free_vram,
    amdgpu_lite_map_gpu,
    amdgpu_lite_unmap_gpu,
)

PAGE_SIZE = 4096


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


@dataclass
class DMAAllocation:
    """A DMA-coherent allocation with both CPU and bus addresses.

    Used for hardware register programming where the physical bus address
    is needed (ring buffers, MQD, EOP, GART table base, etc.).
    """

    cpu_addr: int
    bus_addr: int
    handle: int
    size: int


class LiteMemoryManager:
    """Manages memory allocation via amdgpu_lite kernel module ioctls."""

    def __init__(self, fd: int) -> None:
        self._fd = fd
        self._allocations: list[MemoryHandle] = []
        self._dma_allocations: dict[int, DMAAllocation] = {}  # handle -> DMAAllocation

    def alloc(
        self,
        size: int,
        location: MemoryLocation,
        *,
        executable: bool = False,
        public: bool = False,
        uncached: bool = False,
        map_gpu: bool = True,
    ) -> MemoryHandle:
        """Allocate memory and optionally install GPU page table entries.

        For GTT: kernel does dma_alloc_coherent, returns bus_addr + mmap_offset.
        For VRAM: kernel does bitmap alloc from VRAM, returns gpu_addr + mmap_offset.
        Then MAP_GPU installs GART page table entries for GPU access.
        """
        size = _align_up(size, PAGE_SIZE)

        if location == MemoryLocation.VRAM:
            return self._alloc_vram(size, map_gpu=map_gpu)
        elif location == MemoryLocation.GTT:
            return self._alloc_gtt(size, map_gpu=map_gpu)
        else:
            raise MemoryAllocationError(
                size, location.value, "USERPTR not supported on amdgpu_lite"
            )

    def _alloc_gtt(self, size: int, *, map_gpu: bool = True) -> MemoryHandle:
        """Allocate DMA-coherent system memory via ALLOC_GTT ioctl."""
        args = amdgpu_lite_alloc_gtt()
        args.size = size

        try:
            helpers.ioctl(self._fd, AMDGPU_LITE_IOC_ALLOC_GTT, args, "ALLOC_GTT")
        except Exception:
            raise MemoryAllocationError(size, "GTT")

        # CPU-map via mmap on the device fd
        cpu_addr = helpers.libc_mmap(
            None,
            size,
            helpers.PROT_READ | helpers.PROT_WRITE,
            helpers.MAP_SHARED,
            self._fd,
            args.mmap_offset,
        )

        handle = MemoryHandle(
            kfd_handle=args.handle,
            gpu_addr=0,  # Set after MAP_GPU
            cpu_addr=cpu_addr,
            size=size,
            location=MemoryLocation.GTT,
        )

        if map_gpu:
            self._map_to_gpu(handle)

        self._allocations.append(handle)
        return handle

    def _alloc_vram(self, size: int, *, map_gpu: bool = True) -> MemoryHandle:
        """Allocate VRAM via ALLOC_VRAM ioctl.

        VRAM allocations already have a physical GPU address (their offset
        within the VRAM aperture). They don't need GART page table entries
        because the GPU accesses VRAM directly. The gpu_addr returned by
        the kernel is the VRAM-relative offset.
        """
        args = amdgpu_lite_alloc_vram()
        args.size = size
        args.flags = 0

        try:
            helpers.ioctl(self._fd, AMDGPU_LITE_IOC_ALLOC_VRAM, args, "ALLOC_VRAM")
        except Exception:
            raise MemoryAllocationError(size, "VRAM")

        # CPU-map VRAM via BAR aperture mmap
        cpu_addr = 0
        try:
            cpu_addr = helpers.libc_mmap(
                None,
                size,
                helpers.PROT_READ | helpers.PROT_WRITE,
                helpers.MAP_SHARED,
                self._fd,
                args.mmap_offset,
            )
        except OSError:
            pass  # CPU mapping not always possible for all VRAM

        handle = MemoryHandle(
            kfd_handle=args.handle,
            gpu_addr=args.gpu_addr,  # VRAM offset, directly addressable by GPU
            cpu_addr=cpu_addr,
            size=size,
            location=MemoryLocation.VRAM,
        )
        # No MAP_GPU needed — VRAM is directly addressable by the GPU
        self._allocations.append(handle)
        return handle

    def _map_to_gpu(self, handle: MemoryHandle) -> None:
        """Install GART page table entries for GPU access via MAP_GPU ioctl."""
        args = amdgpu_lite_map_gpu()
        args.handle = handle.kfd_handle
        args.gpu_va = 0  # Auto-assign
        args.size = handle.size
        args.flags = 0

        helpers.ioctl(self._fd, AMDGPU_LITE_IOC_MAP_GPU, args, "MAP_GPU")
        handle.gpu_addr = args.mapped_gpu_va

    def unmap_from_gpu(self, handle: MemoryHandle) -> None:
        """Remove GART page table entries."""
        if handle.gpu_addr == 0:
            return
        args = amdgpu_lite_unmap_gpu()
        args.gpu_va = handle.gpu_addr
        args.size = handle.size
        try:
            helpers.ioctl(self._fd, AMDGPU_LITE_IOC_UNMAP_GPU, args, "UNMAP_GPU")
        except Exception:
            pass
        handle.gpu_addr = 0

    def free(self, handle: MemoryHandle) -> None:
        """Free a memory allocation."""
        # Unmap GPU page tables
        self.unmap_from_gpu(handle)

        # Free kernel allocation
        if handle.location == MemoryLocation.GTT:
            args = amdgpu_lite_free_gtt()
            args.handle = handle.kfd_handle
            try:
                helpers.ioctl(self._fd, AMDGPU_LITE_IOC_FREE_GTT, args, "FREE_GTT")
            except Exception:
                pass
        elif handle.location == MemoryLocation.VRAM:
            args = amdgpu_lite_free_vram()
            args.handle = handle.kfd_handle
            try:
                helpers.ioctl(self._fd, AMDGPU_LITE_IOC_FREE_VRAM, args, "FREE_VRAM")
            except Exception:
                pass

        # Unmap CPU mapping
        if handle.cpu_addr and handle.size:
            try:
                helpers.libc_munmap(handle.cpu_addr, handle.size)
            except Exception:
                pass

        if handle in self._allocations:
            self._allocations.remove(handle)

    def alloc_dma(self, size: int) -> DMAAllocation:
        """Allocate DMA-coherent memory WITHOUT GART mapping.

        Returns both cpu_addr and bus_addr. The bus_addr is the physical
        DMA address needed for programming hardware registers (ring bases,
        MQD address, EOP buffer, GART table base, etc.).

        Unlike alloc(GTT, map_gpu=True), this does NOT install GART PTEs.
        The hardware accesses these buffers via their physical bus address,
        not through GPU virtual address translation.
        """
        size = _align_up(size, PAGE_SIZE)
        args = amdgpu_lite_alloc_gtt()
        args.size = size

        try:
            helpers.ioctl(self._fd, AMDGPU_LITE_IOC_ALLOC_GTT, args, "ALLOC_GTT")
        except Exception:
            raise MemoryAllocationError(size, "DMA")

        cpu_addr = helpers.libc_mmap(
            None,
            size,
            helpers.PROT_READ | helpers.PROT_WRITE,
            helpers.MAP_SHARED,
            self._fd,
            args.mmap_offset,
        )

        alloc = DMAAllocation(
            cpu_addr=cpu_addr,
            bus_addr=args.bus_addr,
            handle=args.handle,
            size=size,
        )
        self._dma_allocations[args.handle] = alloc
        return alloc

    def free_dma(self, handle: int) -> None:
        """Free a DMA allocation by handle."""
        alloc = self._dma_allocations.pop(handle, None)
        if alloc is None:
            return

        free_args = amdgpu_lite_free_gtt()
        free_args.handle = handle
        try:
            helpers.ioctl(self._fd, AMDGPU_LITE_IOC_FREE_GTT, free_args, "FREE_GTT")
        except Exception:
            pass

        if alloc.cpu_addr and alloc.size:
            try:
                helpers.libc_munmap(alloc.cpu_addr, alloc.size)
            except Exception:
                pass

    def free_all(self) -> None:
        """Free all tracked allocations."""
        for handle in list(self._allocations):
            self.free(handle)
        for handle in list(self._dma_allocations.keys()):
            self.free_dma(handle)
