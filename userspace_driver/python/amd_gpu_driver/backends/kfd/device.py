"""KFD device backend implementation."""

from __future__ import annotations

import ctypes
import os
from typing import Any

from amd_gpu_driver.backends.base import (
    DeviceBackend,
    MemoryHandle,
    MemoryLocation,
    QueueHandle,
    QueueType,
    SignalHandle,
)
from amd_gpu_driver.backends.kfd.events import KFDEventManager
from amd_gpu_driver.backends.kfd.memory import KFDMemoryManager
from amd_gpu_driver.backends.kfd.queue import KFDQueueManager
from amd_gpu_driver.errors import DeviceNotFoundError, IoctlError
from amd_gpu_driver.gpu import get_gpu_family
from amd_gpu_driver.gpu.family import GPUFamilyConfig
from amd_gpu_driver.ioctl import helpers
from amd_gpu_driver.ioctl.kfd import (
    AMDKFD_IOC_ACQUIRE_VM,
    AMDKFD_IOC_GET_PROCESS_APERTURES_NEW,
    AMDKFD_IOC_GET_VERSION,
    KFD_IOC_ALLOC_MEM_FLAGS_MMIO_REMAP,
    KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE,
    kfd_ioctl_acquire_vm_args,
    kfd_ioctl_get_process_apertures_new_args,
    kfd_ioctl_get_version_args,
    kfd_process_device_apertures,
)
from amd_gpu_driver.topology import GPUNode, discover_gpu_nodes

KFD_DEVICE_PATH = "/dev/kfd"

# Process-global KFD state: KFD fd + acquired GPUs are per-process singletons.
# ACQUIRE_VM can only be called once per GPU per process.
_global_kfd_fd: int = -1
_acquired_gpus: dict[int, int] = {}  # gpu_id -> drm_fd


def _get_kfd_fd() -> int:
    """Get or open the process-global KFD file descriptor."""
    global _global_kfd_fd
    if _global_kfd_fd < 0:
        _global_kfd_fd = os.open(KFD_DEVICE_PATH, os.O_RDWR | os.O_CLOEXEC)
    return _global_kfd_fd


class KFDDevice(DeviceBackend):
    """KFD backend: talks to /dev/kfd + /dev/dri/renderD* via ioctls."""

    def __init__(self) -> None:
        self._kfd_fd: int = -1
        self._drm_fd: int = -1
        self._node: GPUNode | None = None
        self._family: GPUFamilyConfig | None = None
        self._gpu_id_value: int = 0
        self._gpuvm_base: int = 0
        self._gpuvm_limit: int = 0
        self._lds_base: int = 0
        self._scratch_base: int = 0
        self._doorbell_mmap_addr: int = 0
        self._doorbell_mmap_size: int = 0
        self._mmio_addr: int = 0
        self._opened = False
        self._owns_drm_fd = False
        self._memory: KFDMemoryManager | None = None
        self._queues: KFDQueueManager | None = None
        self._events: KFDEventManager | None = None

    def open(self, device_index: int = 0) -> None:
        """Open KFD device, acquire VM, set up apertures."""
        # 1. Discover GPU nodes from topology
        nodes = discover_gpu_nodes()
        if device_index >= len(nodes):
            raise DeviceNotFoundError(device_index)
        self._node = nodes[device_index]

        # 2. Look up GPU family config
        self._family = get_gpu_family(self._node.gfx_target_version)
        if self._family is None:
            from amd_gpu_driver.errors import UnsupportedGPUError
            raise UnsupportedGPUError(self._node.gfx_target_version)

        # 3. Get the process-global KFD fd
        self._kfd_fd = _get_kfd_fd()

        # 4. Get process apertures to obtain gpu_id
        self._query_apertures_and_gpu_id()

        # 5. Open DRM render node and acquire VM (once per GPU per process)
        if self._gpu_id_value in _acquired_gpus:
            # Already acquired - reuse the DRM fd
            self._drm_fd = _acquired_gpus[self._gpu_id_value]
            self._owns_drm_fd = False
        else:
            drm_path = self._node.drm_render_path
            self._drm_fd = os.open(drm_path, os.O_RDWR | os.O_CLOEXEC)
            self._owns_drm_fd = True

            # 6. Acquire VM: bind KFD to DRM (one-time per GPU per process)
            acquire = kfd_ioctl_acquire_vm_args()
            acquire.drm_fd = self._drm_fd
            acquire.gpu_id = self._gpu_id_value
            try:
                helpers.ioctl(self._kfd_fd, AMDKFD_IOC_ACQUIRE_VM, acquire, "ACQUIRE_VM")
            except Exception:
                os.close(self._drm_fd)
                self._drm_fd = -1
                self._owns_drm_fd = False
                raise
            _acquired_gpus[self._gpu_id_value] = self._drm_fd

        # 7. Initialize subsystems
        self._memory = KFDMemoryManager(
            kfd_fd=self._kfd_fd,
            drm_fd=self._drm_fd,
            gpu_id=self._gpu_id_value,
        )
        self._events = KFDEventManager(
            kfd_fd=self._kfd_fd,
            gpu_id=self._gpu_id_value,
            node_id=self._node.node_id,
        )
        self._queues = KFDQueueManager(
            kfd_fd=self._kfd_fd,
            gpu_id=self._gpu_id_value,
            memory=self._memory,
            family=self._family,
            node=self._node,
        )

        # 8. Allocate MMIO remap page for HDP flush
        self._alloc_mmio_remap()

        self._opened = True

    def _query_apertures_and_gpu_id(self) -> None:
        """Query process apertures to get gpu_id and GPUVM address range.

        The gpu_id is not exposed in sysfs topology. It must be obtained from
        the KFD GET_PROCESS_APERTURES_NEW ioctl. We use the device_index to
        select the right aperture entry (aperture entries correspond 1:1 to
        GPU topology nodes in order).
        """
        # First call to get count
        args = kfd_ioctl_get_process_apertures_new_args()
        args.kfd_process_device_apertures_ptr = 0
        args.num_of_nodes = 0
        helpers.ioctl(
            self._kfd_fd,
            AMDKFD_IOC_GET_PROCESS_APERTURES_NEW,
            args,
            "GET_PROCESS_APERTURES_NEW",
        )

        num_nodes = args.num_of_nodes
        if num_nodes == 0:
            return

        # Allocate array and query again
        ApertureArray = kfd_process_device_apertures * num_nodes
        apertures = ApertureArray()

        args2 = kfd_ioctl_get_process_apertures_new_args()
        args2.kfd_process_device_apertures_ptr = ctypes.addressof(apertures)
        args2.num_of_nodes = num_nodes
        helpers.ioctl(
            self._kfd_fd,
            AMDKFD_IOC_GET_PROCESS_APERTURES_NEW,
            args2,
            "GET_PROCESS_APERTURES_NEW",
        )

        # Build a list of GPU apertures (those with non-zero gpu_id)
        gpu_apertures = []
        for i in range(args2.num_of_nodes):
            ap = apertures[i]
            if ap.gpu_id != 0:
                gpu_apertures.append(ap)

        # Find our GPU node's aperture. Topology GPU nodes and aperture
        # entries with non-zero gpu_id are in the same order.
        assert self._node is not None
        target_render_minor = self._node.drm_render_minor

        # Try matching by drm_render_minor using topology
        all_gpu_nodes = discover_gpu_nodes()
        for idx, node in enumerate(all_gpu_nodes):
            if node.node_id == self._node.node_id and idx < len(gpu_apertures):
                ap = gpu_apertures[idx]
                self._gpu_id_value = ap.gpu_id
                self._gpuvm_base = ap.gpuvm_base
                self._gpuvm_limit = ap.gpuvm_limit
                self._lds_base = ap.lds_base
                self._scratch_base = ap.scratch_base
                return

        # Fallback: use the device_index directly
        if gpu_apertures:
            # Find the index of our node in the GPU node list
            for idx, node in enumerate(all_gpu_nodes):
                if node.node_id == self._node.node_id:
                    if idx < len(gpu_apertures):
                        ap = gpu_apertures[idx]
                        self._gpu_id_value = ap.gpu_id
                        self._gpuvm_base = ap.gpuvm_base
                        self._gpuvm_limit = ap.gpuvm_limit
                        self._lds_base = ap.lds_base
                        self._scratch_base = ap.scratch_base
                        return

    def _alloc_mmio_remap(self) -> None:
        """Allocate MMIO remap page for HDP flush register access."""
        assert self._memory is not None
        mmio_handle = self._memory.alloc_mmio_remap(page_size=4096)
        self._mmio_addr = mmio_handle.cpu_addr

    def close(self) -> None:
        """Release all resources.

        Note: the KFD fd and DRM fd are process-global singletons and are NOT
        closed here. They persist for the process lifetime because ACQUIRE_VM
        cannot be re-issued after the fd is closed.
        """
        if self._queues is not None:
            try:
                self._queues.destroy_all()
            except Exception:
                pass
            self._queues = None
        if self._events is not None:
            try:
                self._events.destroy_all()
            except Exception:
                pass
            self._events = None
        if self._memory is not None:
            try:
                self._memory.free_all()
            except Exception:
                pass
            self._memory = None
        # KFD fd and DRM fd are process-global; do not close them
        self._drm_fd = -1
        self._kfd_fd = -1
        self._opened = False

    # --- Memory ---

    def alloc_memory(
        self,
        size: int,
        location: MemoryLocation,
        *,
        executable: bool = False,
        public: bool = False,
        uncached: bool = False,
    ) -> MemoryHandle:
        assert self._memory is not None
        return self._memory.alloc(
            size, location, executable=executable, public=public, uncached=uncached
        )

    def free_memory(self, handle: MemoryHandle) -> None:
        assert self._memory is not None
        self._memory.free(handle)

    def map_memory(self, handle: MemoryHandle) -> None:
        assert self._memory is not None
        self._memory.map_to_gpu(handle)

    def map_memory_to_peers(
        self, handle: MemoryHandle, peer_gpu_ids: list[int]
    ) -> None:
        """Map memory into page tables of peer GPUs for P2P access."""
        assert self._memory is not None
        # Include already-mapped GPUs to avoid EINVAL from the kernel
        # when remapping a buffer that already has page table entries.
        already_mapped = set(handle.mapped_gpu_ids)
        all_gpu_ids = list(
            {self._gpu_id_value} | already_mapped | set(peer_gpu_ids)
        )
        self._memory.map_to_gpu(handle, gpu_ids=all_gpu_ids)

    # --- Queues ---

    def create_compute_queue(self) -> QueueHandle:
        assert self._queues is not None
        return self._queues.create_compute_queue()

    def create_sdma_queue(self) -> QueueHandle:
        assert self._queues is not None
        return self._queues.create_sdma_queue()

    def create_xgmi_sdma_queue(self) -> QueueHandle:
        """Create an XGMI SDMA queue for cross-GPU copies."""
        assert self._queues is not None
        return self._queues.create_xgmi_sdma_queue()

    def destroy_queue(self, handle: QueueHandle) -> None:
        assert self._queues is not None
        self._queues.destroy_queue(handle)

    def submit_packets(self, queue: QueueHandle, packets: bytes) -> None:
        assert self._queues is not None
        self._queues.submit(queue, packets)

    # --- Signals ---

    def create_signal(self) -> SignalHandle:
        assert self._events is not None
        return self._events.create_signal()

    def destroy_signal(self, handle: SignalHandle) -> None:
        assert self._events is not None
        self._events.destroy(handle)

    def wait_signal(self, handle: SignalHandle, timeout_ms: int = 5000) -> None:
        assert self._events is not None
        self._events.wait(handle, timeout_ms)

    # --- Properties ---

    @property
    def gpu_id(self) -> int:
        return self._gpu_id_value

    @property
    def gfx_target_version(self) -> int:
        return self._node.gfx_target_version if self._node else 0

    @property
    def vram_size(self) -> int:
        return self._node.vram_size if self._node else 0

    @property
    def name(self) -> str:
        if self._family:
            return f"{self._family.architecture} ({self._family.name})"
        return "Unknown AMD GPU"

    @property
    def family(self) -> GPUFamilyConfig | None:
        return self._family

    @property
    def node(self) -> GPUNode | None:
        return self._node

    @property
    def kfd_fd(self) -> int:
        return self._kfd_fd

    @property
    def drm_fd(self) -> int:
        return self._drm_fd

    @property
    def mmio_addr(self) -> int:
        return self._mmio_addr
