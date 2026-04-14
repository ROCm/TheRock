"""macOS DriverKit device backend implementation.

Implements DeviceBackend for the ROCmGPU.dext DriverKit extension,
communicating via IOKit user client calls. This is the macOS equivalent
of the Windows MCDM backend.

Architecture:
  MacOSDevice (this class)
    -> IOKitClient (iokit_client.py)     : DEXT communication
    -> MacOSMemoryManager (memory.py)    : DMA + VRAM allocation
    -> MacOSQueueManager (queue.py)      : Compute/SDMA queues
    -> MacOSEventManager (events.py)     : Signal/event synchronization

GPU initialization (IP discovery, GMC, PSP, rings) reuses the
register-definition modules from the Windows backend, since they
program GPU registers via MMIO which is backend-agnostic.
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
from amd_gpu_driver.backends.macos.discovery import (
    DiscoveredDevice,
    open_device,
)
from amd_gpu_driver.backends.macos.events import MacOSEventManager
from amd_gpu_driver.backends.macos.iokit_client import IOKitClient
from amd_gpu_driver.backends.macos.memory import MacOSMemoryManager
from amd_gpu_driver.backends.macos.queue import MacOSQueueManager


# Known AMD device IDs -> GFX target versions
_DEVICE_TO_GFX: dict[int, int] = {
    # RDNA4
    0x7551: 120100,  # RX 9070 XT -> gfx1201
    0x7550: 120100,  # RX 9070 -> gfx1201
    # RDNA3
    0x744C: 110000,  # RX 7900 XTX -> gfx1100
    0x7448: 110000,  # RX 7900 XT -> gfx1100
    0x7480: 110100,  # RX 7800 XT -> gfx1101
    0x7483: 110100,  # RX 7700 XT -> gfx1101
    0x7460: 110200,  # RX 7600 -> gfx1102
}


class MacOSDevice(DeviceBackend):
    """macOS DriverKit backend: talks to ROCmGPU.dext via IOKit.

    GPU logic (IP discovery, firmware loading, ring setup, command
    submission) runs in Python using MMIO register access provided
    by the DEXT. The DEXT only handles PCIe plumbing (BAR mapping,
    DMA allocation, config space, interrupts).
    """

    def __init__(self) -> None:
        self._client: IOKitClient | None = None
        self._discovered: DiscoveredDevice | None = None
        self._memory: MacOSMemoryManager | None = None
        self._queues: MacOSQueueManager | None = None
        self._events: MacOSEventManager | None = None
        self._device_index: int = 0
        self._opened = False

    def open(self, device_index: int = 0) -> None:
        """Open the AMD GPU via the ROCmGPU DEXT."""
        self._device_index = device_index
        self._client, self._discovered = open_device(device_index)

        # Initialize subsystems
        self._memory = MacOSMemoryManager(
            self._client, gpu_id=device_index)
        self._queues = MacOSQueueManager(
            self._client, self._memory)
        self._events = MacOSEventManager(
            self._client, self._memory)

        self._opened = True

        # Cold-boot the GPU (IP discovery, GMC, PSP, rings)
        # This is deferred until explicitly called because it requires
        # firmware files and may take several seconds.
        # Call bringup() after open() when ready.

    def bringup(self, firmware_path: str | None = None) -> None:
        """Initialize the GPU hardware (cold boot from reset).

        This performs:
          1. Function-Level Reset
          2. BAR mapping (MMIO, doorbell, optionally VRAM)
          3. IP Discovery
          4. NBIO init
          5. GMC init (memory controller, page tables)
          6. PSP init (firmware loading)
          7. IH init (interrupt handler ring)
          8. Ring init (compute/SDMA queue hardware)

        Must be called after open() and before any GPU operations.
        """
        from amd_gpu_driver.backends.macos.bringup import gpu_bringup
        gpu_bringup(self, firmware_path=firmware_path)

    def close(self) -> None:
        """Release all resources and close the device."""
        if self._events:
            self._events.cleanup()
            self._events = None

        if self._client is not None:
            self._client.close()
            self._client = None

        self._memory = None
        self._queues = None
        self._opened = False

    @property
    def client(self) -> IOKitClient:
        """Access the underlying IOKit client for direct DEXT calls."""
        if self._client is None:
            raise RuntimeError("Device not open")
        return self._client

    # ---- Register access (passthrough to DEXT) ----

    def read_reg32(self, offset: int, bar_index: int = 0) -> int:
        """Read a 32-bit MMIO register via DEXT."""
        return self.client.mmio_read32(bar_index, offset)

    def write_reg32(self, offset: int, value: int, bar_index: int = 0) -> None:
        """Write a 32-bit MMIO register via DEXT."""
        self.client.mmio_write32(bar_index, offset, value)

    def read_reg_indirect(self, address: int) -> int:
        """Read a register via SMN indirect access (NBIO index/data).

        Uses NBIO v7.11 index/data registers (RDNA4).
        Same interface as WindowsDevice.read_reg_indirect().
        """
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

        GTT: Uses DEXT DMA allocation (IOBufferMemoryDescriptor).
        VRAM: Uses BAR mapping with bump allocator.
        """
        if self._memory is None:
            raise RuntimeError("Device not open")
        return self._memory.alloc(
            size, location,
            executable=executable, public=public, uncached=uncached,
        )

    def free_memory(self, handle: MemoryHandle) -> None:
        """Free a previously allocated memory region."""
        if self._memory is None:
            raise RuntimeError("Device not open")
        self._memory.free(handle)

    def map_memory(self, handle: MemoryHandle) -> None:
        """Map memory into GPU page tables.

        On macOS, this writes PTEs via MMIO (same approach as Windows).
        The GPU page table management is done in Python using the GMC
        register definitions from the shared init modules.
        """
        # GPU page table writes will be implemented as part of GMC init.
        # For now, GTT allocations use identity-mapped IOMMU addresses
        # and VRAM uses BAR-relative offsets.
        pass

    def create_compute_queue(self) -> QueueHandle:
        """Create a compute queue."""
        if self._queues is None:
            raise RuntimeError("Device not open")
        return self._queues.create_compute_queue()

    def create_sdma_queue(self) -> QueueHandle:
        """Create an SDMA queue."""
        if self._queues is None:
            raise RuntimeError("Device not open")
        return self._queues.create_sdma_queue()

    def destroy_queue(self, handle: QueueHandle) -> None:
        """Destroy a hardware queue."""
        if self._queues is None:
            raise RuntimeError("Device not open")
        self._queues.destroy_queue(handle)

    def submit_packets(self, queue: QueueHandle, packets: bytes) -> None:
        """Submit command packets to a queue's ring buffer."""
        if self._queues is None:
            raise RuntimeError("Device not open")
        self._queues.submit_packets(queue, packets)

    def create_signal(self) -> SignalHandle:
        """Create a signal event for GPU-CPU synchronization."""
        if self._events is None:
            raise RuntimeError("Device not open")
        return self._events.create_signal()

    def destroy_signal(self, handle: SignalHandle) -> None:
        """Destroy a signal event."""
        if self._events is None:
            raise RuntimeError("Device not open")
        self._events.destroy_signal(handle)

    def wait_signal(self, handle: SignalHandle, timeout_ms: int = 5000) -> None:
        """Wait for a signal event to be triggered."""
        if self._events is None:
            raise RuntimeError("Device not open")
        self._events.wait_signal(handle, timeout_ms)

    # ---- Properties ----

    @property
    def gpu_id(self) -> int:
        """Synthetic GPU ID (no KFD on macOS)."""
        return self._device_index

    @property
    def gfx_target_version(self) -> int:
        """GFX target version integer (e.g., 120100 for gfx1201)."""
        if self._discovered is None:
            return 0
        return self._discovered.gfx_version

    @property
    def vram_size(self) -> int:
        """Total VRAM in bytes."""
        if self._discovered is None:
            return 0
        return self._discovered.info.vram_size

    @property
    def name(self) -> str:
        """Human-readable device name."""
        if self._discovered is not None:
            return self._discovered.device_name
        return "Unknown AMD GPU"

    def __repr__(self) -> str:
        if self._discovered:
            return (
                f"MacOSDevice(name={self.name!r}, "
                f"device_id=0x{self._discovered.info.device_id:04X}, "
                f"vram={self.vram_size // (1024**3)}GB)"
            )
        return "MacOSDevice(not opened)"
