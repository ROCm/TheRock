"""Memory management via amdgpu_lite ioctls.

Unlike KFD which uses separate alloc/map ioctls through /dev/kfd + DRM,
amdgpu_lite handles everything through a single device fd. The kernel module
manages DMA-coherent GTT allocations, VRAM bitmap allocation, and GART
page table entries directly.
"""

from __future__ import annotations

import ctypes

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


class LiteMemoryManager:
    """Manages memory allocation via amdgpu_lite kernel module ioctls."""

    def __init__(self, fd: int) -> None:
        self._fd = fd
        self._allocations: list[MemoryHandle] = []

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

    def free_all(self) -> None:
        """Free all tracked allocations."""
        for handle in list(self._allocations):
            self.free(handle)
