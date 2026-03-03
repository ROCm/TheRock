"""KFD memory allocation via ioctls."""

from __future__ import annotations

import ctypes
import os

from amd_gpu_driver.backends.base import MemoryHandle, MemoryLocation
from amd_gpu_driver.errors import MemoryAllocationError
from amd_gpu_driver.ioctl import helpers
from amd_gpu_driver.ioctl.kfd import (
    AMDKFD_IOC_ALLOC_MEMORY_OF_GPU,
    AMDKFD_IOC_FREE_MEMORY_OF_GPU,
    AMDKFD_IOC_MAP_MEMORY_TO_GPU,
    AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU,
    KFD_IOC_ALLOC_MEM_FLAGS_COHERENT,
    KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE,
    KFD_IOC_ALLOC_MEM_FLAGS_GTT,
    KFD_IOC_ALLOC_MEM_FLAGS_MMIO_REMAP,
    KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE,
    KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC,
    KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED,
    KFD_IOC_ALLOC_MEM_FLAGS_USERPTR,
    KFD_IOC_ALLOC_MEM_FLAGS_VRAM,
    KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE,
    kfd_ioctl_alloc_memory_of_gpu_args,
    kfd_ioctl_free_memory_of_gpu_args,
    kfd_ioctl_map_memory_to_gpu_args,
    kfd_ioctl_unmap_memory_from_gpu_args,
)

PAGE_SIZE = 4096


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


class KFDMemoryManager:
    """Manages GPU memory allocation via KFD ioctls."""

    def __init__(self, kfd_fd: int, drm_fd: int, gpu_id: int) -> None:
        self._kfd_fd = kfd_fd
        self._drm_fd = drm_fd
        self._gpu_id = gpu_id
        self._allocations: list[MemoryHandle] = []

    def alloc(
        self,
        size: int,
        location: MemoryLocation,
        *,
        executable: bool = False,
        public: bool = False,
        uncached: bool = False,
    ) -> MemoryHandle:
        """Allocate GPU memory following the KFD pattern:
        1. mmap(PROT_NONE) to reserve VA
        2. ALLOC_MEMORY_OF_GPU ioctl
        3. mmap(MAP_FIXED) the DRM fd for CPU access
        4. MAP_MEMORY_TO_GPU for GPU page table entries
        """
        size = _align_up(size, PAGE_SIZE)

        # Build flags
        flags = KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE | KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE

        if location == MemoryLocation.VRAM:
            flags |= KFD_IOC_ALLOC_MEM_FLAGS_VRAM | KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE
            if public:
                flags |= KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC
        elif location == MemoryLocation.GTT:
            flags |= KFD_IOC_ALLOC_MEM_FLAGS_GTT | KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE
            if uncached:
                flags |= KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED | KFD_IOC_ALLOC_MEM_FLAGS_COHERENT
        elif location == MemoryLocation.USERPTR:
            flags |= (
                KFD_IOC_ALLOC_MEM_FLAGS_USERPTR
                | KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE
                | KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC
            )

        if executable:
            flags |= KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE

        # Step 1: Reserve VA space with anonymous mmap
        va = helpers.libc_mmap(
            None,
            size,
            helpers.PROT_NONE,
            helpers.MAP_PRIVATE | helpers.MAP_ANONYMOUS | helpers.MAP_NORESERVE,
            -1,
            0,
        )

        # Step 2: KFD alloc
        alloc_args = kfd_ioctl_alloc_memory_of_gpu_args()
        alloc_args.va_addr = va
        alloc_args.size = size
        alloc_args.gpu_id = self._gpu_id
        alloc_args.flags = flags

        try:
            helpers.ioctl(
                self._kfd_fd,
                AMDKFD_IOC_ALLOC_MEMORY_OF_GPU,
                alloc_args,
                "ALLOC_MEMORY_OF_GPU",
            )
        except Exception:
            helpers.libc_munmap(va, size)
            raise MemoryAllocationError(size, location.value)

        kfd_handle = alloc_args.handle
        mmap_offset = alloc_args.mmap_offset

        # Step 3: CPU mapping via DRM fd (for VRAM and GTT)
        cpu_addr = 0
        if location in (MemoryLocation.VRAM, MemoryLocation.GTT):
            try:
                cpu_addr = helpers.libc_mmap(
                    va,
                    size,
                    helpers.PROT_READ | helpers.PROT_WRITE,
                    helpers.MAP_SHARED | helpers.MAP_FIXED,
                    self._drm_fd,
                    mmap_offset,
                )
            except OSError:
                # CPU mapping failed, not fatal for all use cases
                cpu_addr = 0

        handle = MemoryHandle(
            kfd_handle=kfd_handle,
            gpu_addr=va,
            cpu_addr=cpu_addr if cpu_addr else va,
            size=size,
            location=location,
            flags=flags,
        )

        # Step 4: Map to GPU
        self.map_to_gpu(handle)
        handle.owner_gpu_id = self._gpu_id

        self._allocations.append(handle)
        return handle

    def alloc_mmio_remap(self, page_size: int = PAGE_SIZE) -> MemoryHandle:
        """Allocate MMIO remap page (for HDP flush).

        Unlike VRAM/GTT allocations, MMIO remap pages are mmap'd against
        the KFD fd (not the DRM fd).
        """
        va = helpers.libc_mmap(
            None,
            page_size,
            helpers.PROT_NONE,
            helpers.MAP_PRIVATE | helpers.MAP_ANONYMOUS | helpers.MAP_NORESERVE,
            -1,
            0,
        )

        flags = KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE | KFD_IOC_ALLOC_MEM_FLAGS_MMIO_REMAP

        alloc_args = kfd_ioctl_alloc_memory_of_gpu_args()
        alloc_args.va_addr = va
        alloc_args.size = page_size
        alloc_args.gpu_id = self._gpu_id
        alloc_args.flags = flags

        helpers.ioctl(
            self._kfd_fd,
            AMDKFD_IOC_ALLOC_MEMORY_OF_GPU,
            alloc_args,
            "ALLOC_MEMORY_OF_GPU(MMIO)",
        )

        # MMIO remap pages use the KFD fd for mmap, not the DRM fd
        cpu_addr = helpers.libc_mmap(
            va,
            page_size,
            helpers.PROT_READ | helpers.PROT_WRITE,
            helpers.MAP_SHARED | helpers.MAP_FIXED,
            self._kfd_fd,
            alloc_args.mmap_offset,
        )

        handle = MemoryHandle(
            kfd_handle=alloc_args.handle,
            gpu_addr=va,
            cpu_addr=cpu_addr,
            size=page_size,
            location=MemoryLocation.VRAM,
            flags=flags,
        )
        self._allocations.append(handle)
        return handle

    def map_to_gpu(
        self, handle: MemoryHandle, gpu_ids: list[int] | None = None
    ) -> None:
        """Install GPU page table entries for this allocation.

        Args:
            handle: Memory allocation to map.
            gpu_ids: List of GPU IDs to map to. If None, maps to the
                     owning GPU only (preserves single-GPU behavior).
        """
        if gpu_ids is None:
            gpu_ids = [self._gpu_id]
        n = len(gpu_ids)
        gpu_id_array = (ctypes.c_uint32 * n)(*gpu_ids)

        map_args = kfd_ioctl_map_memory_to_gpu_args()
        map_args.handle = handle.kfd_handle
        map_args.device_ids_array_ptr = ctypes.addressof(gpu_id_array)
        map_args.n_devices = n
        map_args.n_success = 0

        helpers.ioctl(
            self._kfd_fd,
            AMDKFD_IOC_MAP_MEMORY_TO_GPU,
            map_args,
            "MAP_MEMORY_TO_GPU",
        )
        handle.mapped_gpu_ids = list(gpu_ids)

    def unmap_from_gpu(
        self, handle: MemoryHandle, gpu_ids: list[int] | None = None
    ) -> None:
        """Remove GPU page table entries.

        Args:
            handle: Memory allocation to unmap.
            gpu_ids: List of GPU IDs to unmap from. If None, unmaps from
                     all GPUs in handle.mapped_gpu_ids, or the owning GPU.
        """
        if gpu_ids is None:
            gpu_ids = handle.mapped_gpu_ids if handle.mapped_gpu_ids else [self._gpu_id]
        n = len(gpu_ids)
        gpu_id_array = (ctypes.c_uint32 * n)(*gpu_ids)

        unmap_args = kfd_ioctl_unmap_memory_from_gpu_args()
        unmap_args.handle = handle.kfd_handle
        unmap_args.device_ids_array_ptr = ctypes.addressof(gpu_id_array)
        unmap_args.n_devices = n
        unmap_args.n_success = 0

        helpers.ioctl(
            self._kfd_fd,
            AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU,
            unmap_args,
            "UNMAP_MEMORY_FROM_GPU",
        )
        for gid in gpu_ids:
            if gid in handle.mapped_gpu_ids:
                handle.mapped_gpu_ids.remove(gid)

    def free(self, handle: MemoryHandle) -> None:
        """Free a GPU memory allocation."""
        # Unmap from GPU first
        try:
            self.unmap_from_gpu(handle)
        except Exception:
            pass

        # Free via KFD
        free_args = kfd_ioctl_free_memory_of_gpu_args()
        free_args.handle = handle.kfd_handle
        try:
            helpers.ioctl(
                self._kfd_fd,
                AMDKFD_IOC_FREE_MEMORY_OF_GPU,
                free_args,
                "FREE_MEMORY_OF_GPU",
            )
        except Exception:
            pass

        # Unmap VA
        if handle.gpu_addr and handle.size:
            try:
                helpers.libc_munmap(handle.gpu_addr, handle.size)
            except Exception:
                pass

        if handle in self._allocations:
            self._allocations.remove(handle)

    def free_all(self) -> None:
        """Free all tracked allocations."""
        for handle in list(self._allocations):
            self.free(handle)
