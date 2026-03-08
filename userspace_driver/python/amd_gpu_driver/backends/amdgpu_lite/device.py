"""amdgpu_lite device backend implementation.

This backend talks to /dev/amdgpu_lite0 via our lightweight kernel module
instead of /dev/kfd. The kernel module handles only PCI BAR mapping,
DMA memory allocation, VRAM allocation, GART page tables, and MSI-X
interrupt forwarding. Everything else (IP discovery, firmware loading,
GMC init, MQD construction, queue management) happens in userspace.
"""

from __future__ import annotations

import ctypes
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

    # --- Register access (compatible with WindowsDevice interface) ---

    def read_reg32(self, byte_offset: int, bar_index: int = 0) -> int:
        """Read a 32-bit MMIO register at the given byte offset.

        This matches the WindowsDevice.read_reg32() interface so that
        the shared init modules (nbio_init, gmc_init, etc.) work with
        either backend via duck typing.
        """
        mmio = self.mmio_addr
        if mmio == 0:
            raise RuntimeError("MMIO BAR not mapped")
        return ctypes.c_uint32.from_address(mmio + byte_offset).value

    def write_reg32(self, byte_offset: int, value: int, bar_index: int = 0) -> None:
        """Write a 32-bit MMIO register at the given byte offset."""
        mmio = self.mmio_addr
        if mmio == 0:
            raise RuntimeError("MMIO BAR not mapped")
        ctypes.c_uint32.from_address(mmio + byte_offset).value = value & 0xFFFFFFFF

    def read_reg_indirect(self, address: int) -> int:
        """Read a register via SMN indirect access (NBIO index/data).

        Writes the target SMN address to the NBIO index register (byte
        offset 0x60), then reads the result from the data register (0x64).

        Note: On RDNA4 with amdgpu_lite, SMN indirect reads return zeros
        because the NBIO index/data path requires kernel-level SMN
        configuration. Use read_vram() for VRAM reads instead.
        """
        NBIO_INDEX = 0x60
        NBIO_DATA = 0x64
        self.write_reg32(NBIO_INDEX, address)
        return self.read_reg32(NBIO_DATA)

    def map_vram_bar(self) -> int:
        """Map the VRAM BAR if not already mapped. Returns CPU address."""
        if self._info is None:
            raise RuntimeError("Device not opened")
        vram_idx = self._info.vram_bar_index
        if vram_idx not in self._bar_mappings:
            self._map_bar(vram_idx)
        return self._bar_mappings[vram_idx].addr

    def read_vram(self, offset: int, size: int) -> bytes:
        """Read bytes from VRAM via the BAR aperture.

        This is used for reading the IP discovery table at the top of VRAM.
        Requires the VRAM BAR to be mapped (resizable BAR must cover the
        full VRAM range).
        """
        vram_addr = self.map_vram_bar()
        mapping = self._bar_mappings[self._info.vram_bar_index]
        if offset + size > mapping.size:
            raise RuntimeError(
                f"VRAM read at offset {offset:#x}+{size:#x} exceeds "
                f"BAR size {mapping.size:#x}"
            )
        return (ctypes.c_char * size).from_address(vram_addr + offset).raw

    def write_reg_indirect(self, address: int, value: int) -> None:
        """Write a register via SMN indirect access."""
        NBIO_INDEX = 0x60
        NBIO_DATA = 0x64
        self.write_reg32(NBIO_INDEX, address)
        self.write_reg32(NBIO_DATA, value)

    # --- Windows driver compatibility stubs ---

    def enable_msi(self, **kwargs) -> tuple[bool, int]:
        """No-op on Linux: the kernel module handles MSI-X setup.

        Returns (enabled=True, num_vectors=1) to satisfy ih_init.
        """
        return (True, 1)

    def map_bar(self, bar_index: int, offset: int, size: int) -> tuple[int, int]:
        """Return CPU address into an already-mapped BAR.

        On Windows, this does a separate ioctl per sub-region. On Linux with
        amdgpu_lite, the entire BAR is already mapped in open(), so we just
        return the base + offset.

        Note: ring_init hardcodes BAR2 for doorbells (Windows convention).
        On amdgpu_lite, doorbell may be a different BAR index. We remap
        BAR2 requests to the actual doorbell BAR if BAR2 isn't mapped.

        Returns (cpu_addr, handle). Handle is 0 (no separate mapping).
        """
        actual_index = bar_index
        if bar_index not in self._bar_mappings and self._info is not None:
            # Remap: if BAR2 is requested but doorbell is on a different BAR
            if bar_index == 2 and self._info.doorbell_bar_index != 2:
                actual_index = self._info.doorbell_bar_index

        if actual_index not in self._bar_mappings:
            self._map_bar(actual_index)
        mapping = self._bar_mappings[actual_index]
        if offset + size > mapping.size:
            raise RuntimeError(
                f"BAR{actual_index} map at offset {offset:#x}+{size:#x} "
                f"exceeds BAR size {mapping.size:#x}"
            )
        return (mapping.addr + offset, 0)

    # --- DMA allocation (compatible with WindowsDevice.driver interface) ---

    @property
    def driver(self) -> AmdgpuLiteDevice:
        """Self-reference for WindowsDevice.driver compatibility.

        The init code calls dev.driver.alloc_dma() — on Windows this
        goes through DriverInterface, here we implement it directly.
        """
        return self

    def alloc_dma(self, size: int) -> tuple[int, int, int]:
        """Allocate DMA-coherent memory.

        Returns (cpu_address, bus_address, allocation_handle).
        Compatible with WindowsDevice.driver.alloc_dma() interface.
        """
        assert self._memory is not None
        from amd_gpu_driver.backends.amdgpu_lite.memory import DMAAllocation
        alloc = self._memory.alloc_dma(size)
        return (alloc.cpu_addr, alloc.bus_addr, alloc.handle)

    def free_dma(self, allocation_handle: int) -> None:
        """Free a DMA allocation."""
        assert self._memory is not None
        self._memory.free_dma(allocation_handle)

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
