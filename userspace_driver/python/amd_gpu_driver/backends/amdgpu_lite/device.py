"""amdgpu_lite device backend implementation.

This backend talks to /dev/amdgpu_lite0 via our lightweight kernel module
instead of /dev/kfd. The kernel module handles only PCI BAR mapping,
DMA memory allocation, VRAM allocation, GART page tables, and MSI-X
interrupt forwarding. Everything else (IP discovery, firmware loading,
GMC init, MQD construction, queue management) happens in userspace.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

from amd_gpu_driver.backends.base import (
    DeviceBackend,
    MemoryHandle,
    MemoryLocation,
    QueueHandle,
    SignalHandle,
)
from amd_gpu_driver.backends.amdgpu_lite.events import LiteEventManager
from amd_gpu_driver.backends.amdgpu_lite.memory import LiteMemoryManager
from amd_gpu_driver.backends.amdgpu_lite.queue import LiteQueueManager
from amd_gpu_driver.errors import DeviceNotFoundError
from amd_gpu_driver.ioctl import helpers
from amd_gpu_driver.ioctl.amdgpu_lite import (
    AMDGPU_LITE_IOC_GET_INFO,
    AMDGPU_LITE_IOC_MAP_BAR,
    amdgpu_lite_get_info,
    amdgpu_lite_map_bar,
)

DEVICE_PATH_PATTERN = "/dev/amdgpu_lite*"


@dataclass
class BarMapping:
    """A CPU-mapped BAR region."""

    bar_index: int
    addr: int  # CPU virtual address from mmap
    size: int
    phys_addr: int


class AmdgpuLiteDevice(DeviceBackend):
    """amdgpu_lite backend: talks to /dev/amdgpu_lite0 via our kernel module."""

    def __init__(self) -> None:
        self._fd: int = -1
        self._info: amdgpu_lite_get_info | None = None
        self._opened = False
        self._bar_mappings: dict[int, BarMapping] = {}  # bar_index -> BarMapping
        self._memory: LiteMemoryManager | None = None
        self._queues: LiteQueueManager | None = None
        self._events: LiteEventManager | None = None

    def open(self, device_index: int = 0) -> None:
        """Open the amdgpu_lite device and set up subsystems.

        1. Open /dev/amdgpu_liteN
        2. GET_INFO to discover BARs, VRAM, GART
        3. Map MMIO BAR (register access)
        4. Map doorbell BAR (queue doorbells)
        5. Initialize memory, event, and queue managers
        """
        # Find device nodes
        devices = sorted(glob.glob(DEVICE_PATH_PATTERN))
        if device_index >= len(devices):
            raise DeviceNotFoundError(device_index)

        dev_path = devices[device_index]
        self._fd = os.open(dev_path, os.O_RDWR)

        # Query device info
        self._info = amdgpu_lite_get_info()
        helpers.ioctl(self._fd, AMDGPU_LITE_IOC_GET_INFO, self._info, "GET_INFO")

        # Map MMIO BAR (for register reads/writes from userspace)
        mmio_idx = self._info.mmio_bar_index
        if mmio_idx < self._info.num_bars:
            self._map_bar(mmio_idx)

        # Map doorbell BAR (for queue doorbell writes)
        db_idx = self._info.doorbell_bar_index
        if db_idx < self._info.num_bars and db_idx != mmio_idx:
            self._map_bar(db_idx)

        # Initialize memory manager
        self._memory = LiteMemoryManager(self._fd)

        # Initialize event manager
        self._events = LiteEventManager(self._fd)

        # Initialize queue manager with doorbell mapping
        db_mapping = self._bar_mappings.get(db_idx)
        db_addr = db_mapping.addr if db_mapping else 0
        db_size = db_mapping.size if db_mapping else 0
        self._queues = LiteQueueManager(
            fd=self._fd,
            memory=self._memory,
            doorbell_addr=db_addr,
            doorbell_size=db_size,
        )

        self._opened = True

    def _map_bar(self, bar_index: int) -> BarMapping:
        """Map a PCI BAR into userspace via MAP_BAR ioctl + mmap."""
        if bar_index in self._bar_mappings:
            return self._bar_mappings[bar_index]

        assert self._info is not None
        bar_info = self._info.bars[bar_index]
        bar_size = bar_info.size

        # Get mmap offset from kernel
        args = amdgpu_lite_map_bar()
        args.bar_index = bar_index
        args.offset = 0
        args.size = 0  # 0 = entire BAR

        helpers.ioctl(self._fd, AMDGPU_LITE_IOC_MAP_BAR, args, "MAP_BAR")

        # mmap the BAR
        addr = helpers.libc_mmap(
            None,
            bar_size,
            helpers.PROT_READ | helpers.PROT_WRITE,
            helpers.MAP_SHARED,
            self._fd,
            args.mmap_offset,
        )

        mapping = BarMapping(
            bar_index=bar_index,
            addr=addr,
            size=bar_size,
            phys_addr=bar_info.phys_addr,
        )
        self._bar_mappings[bar_index] = mapping
        return mapping

    def close(self) -> None:
        """Release all resources."""
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

        # Unmap BARs
        for mapping in self._bar_mappings.values():
            try:
                helpers.libc_munmap(mapping.addr, mapping.size)
            except Exception:
                pass
        self._bar_mappings.clear()

        # Close device fd
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1

        self._opened = False

    # --- BAR access (for register reads/writes from higher-level code) ---

    @property
    def mmio_addr(self) -> int:
        """CPU address of the MMIO BAR (for register reads/writes)."""
        if self._info is None:
            return 0
        mapping = self._bar_mappings.get(self._info.mmio_bar_index)
        return mapping.addr if mapping else 0

    @property
    def doorbell_addr(self) -> int:
        """CPU address of the doorbell BAR."""
        if self._info is None:
            return 0
        mapping = self._bar_mappings.get(self._info.doorbell_bar_index)
        return mapping.addr if mapping else 0

    @property
    def vram_bar_addr(self) -> int:
        """CPU address of the VRAM BAR (if mapped)."""
        if self._info is None:
            return 0
        mapping = self._bar_mappings.get(self._info.vram_bar_index)
        return mapping.addr if mapping else 0

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
        """Map memory into GPU page tables (GART)."""
        assert self._memory is not None
        self._memory._map_to_gpu(handle)

    # --- Queues ---

    def create_compute_queue(self) -> QueueHandle:
        assert self._queues is not None
        return self._queues.create_compute_queue()

    def create_sdma_queue(self) -> QueueHandle:
        assert self._queues is not None
        return self._queues.create_sdma_queue()

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
        """Device ID (PCI device_id serves as identifier)."""
        if self._info is None:
            return 0
        return self._info.device_id

    @property
    def gfx_target_version(self) -> int:
        """GFX target version.

        For RX 9070 XT (device 0x7551), this is GFX12.0.1 = 120001.
        Determined from device_id since we do IP discovery in userspace.
        """
        if self._info is None:
            return 0
        # Known device IDs -> gfx_target_version
        device_map = {
            0x7551: 120001,  # RX 9070 XT (GFX12.0.1)
            0x7550: 120001,  # RX 9070 (GFX12.0.1)
        }
        return device_map.get(self._info.device_id, 0)

    @property
    def vram_size(self) -> int:
        if self._info is None:
            return 0
        return self._info.vram_size

    @property
    def name(self) -> str:
        if self._info is None:
            return "Unknown"
        return f"AMD GPU {self._info.device_id:#06x} (amdgpu_lite)"

    @property
    def device_fd(self) -> int:
        """Raw device file descriptor for advanced usage."""
        return self._fd

    @property
    def info(self) -> amdgpu_lite_get_info | None:
        """Raw device info struct."""
        return self._info
