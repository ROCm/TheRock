"""Abstract base class for device backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MemoryLocation(Enum):
    VRAM = "vram"
    GTT = "gtt"
    USERPTR = "userptr"


class QueueType(Enum):
    COMPUTE = "compute"
    SDMA = "sdma"


@dataclass
class MemoryHandle:
    """Represents an allocated GPU memory region."""

    kfd_handle: int = 0
    gpu_addr: int = 0
    cpu_addr: int = 0  # CPU-mapped address (0 if not mapped)
    size: int = 0
    location: MemoryLocation = MemoryLocation.VRAM
    flags: int = 0


@dataclass
class QueueHandle:
    """Represents a created hardware queue."""

    queue_id: int = 0
    queue_type: QueueType = QueueType.COMPUTE
    ring_buffer: MemoryHandle | None = None
    ring_size: int = 0
    write_ptr_addr: int = 0
    read_ptr_addr: int = 0
    doorbell_offset: int = 0
    doorbell_addr: int = 0  # CPU-mapped doorbell address
    eop_buffer: MemoryHandle | None = None
    ctx_save_restore: MemoryHandle | None = None


@dataclass
class SignalHandle:
    """Represents a signal event for synchronization."""

    event_id: int = 0
    signal_addr: int = 0
    event_slot_index: int = 0
    event_page_offset: int = 0


class DeviceBackend(ABC):
    """Abstract base defining the device backend interface."""

    @abstractmethod
    def open(self, device_index: int) -> None:
        """Open the device at the given index."""

    @abstractmethod
    def close(self) -> None:
        """Release all resources and close the device."""

    @abstractmethod
    def alloc_memory(
        self,
        size: int,
        location: MemoryLocation,
        *,
        executable: bool = False,
        public: bool = False,
        uncached: bool = False,
    ) -> MemoryHandle:
        """Allocate GPU-accessible memory."""

    @abstractmethod
    def free_memory(self, handle: MemoryHandle) -> None:
        """Free a previously allocated memory region."""

    @abstractmethod
    def map_memory(self, handle: MemoryHandle) -> None:
        """Map memory into GPU page tables."""

    @abstractmethod
    def create_compute_queue(self) -> QueueHandle:
        """Create a compute queue."""

    @abstractmethod
    def create_sdma_queue(self) -> QueueHandle:
        """Create an SDMA (DMA copy) queue."""

    @abstractmethod
    def destroy_queue(self, handle: QueueHandle) -> None:
        """Destroy a hardware queue."""

    @abstractmethod
    def submit_packets(self, queue: QueueHandle, packets: bytes) -> None:
        """Submit command packets to a queue's ring buffer."""

    @abstractmethod
    def create_signal(self) -> SignalHandle:
        """Create a signal event for GPU-CPU synchronization."""

    @abstractmethod
    def destroy_signal(self, handle: SignalHandle) -> None:
        """Destroy a signal event."""

    @abstractmethod
    def wait_signal(self, handle: SignalHandle, timeout_ms: int = 5000) -> None:
        """Wait for a signal event to be triggered."""

    @property
    @abstractmethod
    def gpu_id(self) -> int:
        """KFD GPU ID."""

    @property
    @abstractmethod
    def gfx_target_version(self) -> int:
        """GFX target version integer."""

    @property
    @abstractmethod
    def vram_size(self) -> int:
        """Total VRAM in bytes."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable device name."""
