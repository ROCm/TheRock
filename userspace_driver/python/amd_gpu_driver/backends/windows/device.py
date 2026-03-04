"""Windows MCDM device backend implementation.

Implements DeviceBackend for the amdgpu_mcdm.sys kernel driver,
communicating via D3DKMTEscape.
"""

from __future__ import annotations

from amd_gpu_driver.backends.base import (
    DeviceBackend,
    MemoryHandle,
    MemoryLocation,
    QueueHandle,
    QueueType,
    SignalHandle,
)
from amd_gpu_driver.backends.windows.discovery import (
    DiscoveredDevice,
    open_device,
)
from amd_gpu_driver.backends.windows.driver_interface import (
    DeviceInfo,
    DriverInterface,
)
from amd_gpu_driver.errors import DeviceNotFoundError


class WindowsDevice(DeviceBackend):
    """Windows MCDM backend: talks to amdgpu_mcdm.sys via D3DKMTEscape.

    GPU logic (IP init, firmware loading, ring setup, command submission)
    will be implemented in Python using register read/write escape commands.
    This skeleton provides the DeviceBackend interface and basic device
    management. Queue, memory, and signal operations are stubbed for now
    and will be implemented as the kernel driver gains capabilities.
    """

    def __init__(self) -> None:
        self._iface: DriverInterface | None = None
        self._device_info: DeviceInfo | None = None
        self._discovered: DiscoveredDevice | None = None
        self._device_index: int = 0
        self._opened = False

    def open(self, device_index: int = 0) -> None:
        """Open the AMD GPU MCDM device at the given index."""
        self._device_index = device_index
        self._iface, self._discovered = open_device(device_index)
        self._device_info = self._discovered.info
        self._opened = True

    def close(self) -> None:
        """Release all resources and close the device."""
        if self._iface is not None:
            self._iface.close()
            self._iface = None
        self._opened = False

    @property
    def driver(self) -> DriverInterface:
        """Access the underlying driver interface for escape commands."""
        if self._iface is None:
            raise RuntimeError("Device not open")
        return self._iface

    # ---- Register access (passthrough to escape commands) ----

    def read_reg32(self, offset: int, bar_index: int = 0) -> int:
        """Read a 32-bit MMIO register via kernel driver."""
        return self.driver.read_reg32(offset, bar_index)

    def write_reg32(self, offset: int, value: int, bar_index: int = 0) -> None:
        """Write a 32-bit MMIO register via kernel driver."""
        self.driver.write_reg32(offset, value, bar_index)

    def read_reg_indirect(self, address: int) -> int:
        """Read a register via SMN indirect access (NBIO index/data).

        Most AMD GPU registers are accessed indirectly:
          1. Write target address to NBIO index register
          2. Read result from NBIO data register

        The index/data register offsets depend on the NBIO version.
        For NBIO v7.11 (RDNA4): index=0x60, data=0x64.
        """
        # NBIO v7.11 SMN index/data registers
        NBIO_INDEX = 0x60
        NBIO_DATA = 0x64

        self.write_reg32(NBIO_INDEX, address)
        return self.read_reg32(NBIO_DATA)

    def write_reg_indirect(self, address: int, value: int) -> None:
        """Write a register via SMN indirect access."""
        NBIO_INDEX = 0x60
        NBIO_DATA = 0x64

        self.write_reg32(NBIO_INDEX, address)
        self.write_reg32(NBIO_DATA, value)

    # ---- DeviceBackend abstract method implementations ----

    def alloc_memory(
        self,
        size: int,
        location: MemoryLocation,
        *,
        executable: bool = False,
        public: bool = False,
        uncached: bool = False,
    ) -> MemoryHandle:
        """Allocate GPU-accessible memory.

        GTT: Uses ESCAPE_ALLOC_DMA for contiguous system memory.
        VRAM: Uses ESCAPE_MAP_VRAM for BAR2-mapped VRAM.

        Not yet implemented — requires kernel driver v0.3+.
        """
        raise NotImplementedError(
            "Memory allocation not yet implemented — "
            "requires kernel driver v0.3+ (ALLOC_DMA / MAP_VRAM)"
        )

    def free_memory(self, handle: MemoryHandle) -> None:
        """Free a previously allocated memory region."""
        raise NotImplementedError("Memory free not yet implemented")

    def map_memory(self, handle: MemoryHandle) -> None:
        """Map memory into GPU page tables.

        On Windows, this will be done by writing PTEs via MMIO
        (the kernel driver doesn't manage GPU page tables).
        """
        raise NotImplementedError(
            "GPU page table management not yet implemented"
        )

    def create_compute_queue(self) -> QueueHandle:
        """Create a compute queue.

        Requires MES firmware loaded and doorbell BAR mapped.
        Not yet implemented — requires kernel driver v0.4+ and
        Phase 3 GPU bring-up.
        """
        raise NotImplementedError(
            "Compute queue creation not yet implemented — "
            "requires GPU bring-up (Phase 3)"
        )

    def create_sdma_queue(self) -> QueueHandle:
        """Create an SDMA queue."""
        raise NotImplementedError(
            "SDMA queue creation not yet implemented — "
            "requires GPU bring-up (Phase 3)"
        )

    def destroy_queue(self, handle: QueueHandle) -> None:
        """Destroy a hardware queue."""
        raise NotImplementedError("Queue destruction not yet implemented")

    def submit_packets(self, queue: QueueHandle, packets: bytes) -> None:
        """Submit command packets to a queue's ring buffer."""
        raise NotImplementedError("Packet submission not yet implemented")

    def create_signal(self) -> SignalHandle:
        """Create a signal event for GPU-CPU synchronization."""
        raise NotImplementedError(
            "Signal events not yet implemented — "
            "requires kernel driver v0.4+ (REGISTER_EVENT)"
        )

    def destroy_signal(self, handle: SignalHandle) -> None:
        """Destroy a signal event."""
        raise NotImplementedError("Signal destruction not yet implemented")

    def wait_signal(self, handle: SignalHandle, timeout_ms: int = 5000) -> None:
        """Wait for a signal event to be triggered."""
        raise NotImplementedError("Signal wait not yet implemented")

    # ---- Properties ----

    @property
    def gpu_id(self) -> int:
        """Synthetic GPU ID (no KFD on Windows)."""
        return self._device_index

    @property
    def gfx_target_version(self) -> int:
        """GFX target version integer.

        Determined from PCI device ID. For RX 9070 XT (0x7551),
        this is gfx1201 = 120100.
        """
        if self._device_info is None:
            return 0
        # Map known device IDs to GFX target versions
        device_to_gfx = {
            0x7551: 120100,  # RX 9070 XT → gfx1201
            0x7550: 120100,  # RX 9070 → gfx1201
        }
        return device_to_gfx.get(self._device_info.device_id, 0)

    @property
    def vram_size(self) -> int:
        """Total VRAM in bytes."""
        if self._device_info is None:
            return 0
        return self._device_info.vram_size

    @property
    def name(self) -> str:
        """Human-readable device name."""
        if self._discovered is not None:
            return self._discovered.device_name
        return "Unknown AMD GPU"

    def __repr__(self) -> str:
        if self._device_info:
            return (
                f"WindowsDevice(name={self.name!r}, "
                f"device_id=0x{self._device_info.device_id:04X}, "
                f"vram={self.vram_size // (1024**3)}GB)"
            )
        return "WindowsDevice(not opened)"
