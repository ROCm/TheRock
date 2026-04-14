"""IOKit ctypes bindings for communicating with ROCmGPU.dext.

Wraps Apple's IOKit.framework C functions to provide a Python interface
for the DriverKit extension's IOUserClient escape commands. This is the
macOS equivalent of the Windows D3DKMTEscape interface.

Communication flow:
  Python (this module)
    -> IOConnectCallScalarMethod / IOConnectCallStructMethod
    -> IOKit.framework
    -> ROCmGPUUserClient::ExternalMethod()
    -> PCI hardware

Memory mapping flow:
  Python: IOConnectMapMemory64(conn, type, ...)
    -> ROCmGPUUserClient::CopyClientMemoryForType()
    -> Returns IOMemoryDescriptor for BAR or DMA buffer
    -> Kernel maps into client process address space
"""

from __future__ import annotations

import ctypes
import ctypes.util
import struct
import sys
from dataclasses import dataclass
from enum import IntEnum

if sys.platform != "darwin":
    raise ImportError("IOKit client is only available on macOS")


# ============================================================================
# Load IOKit framework
# ============================================================================

_iokit_path = ctypes.util.find_library("IOKit")
if not _iokit_path:
    raise ImportError("IOKit.framework not found")

_iokit = ctypes.cdll.LoadLibrary(_iokit_path)

_cf_path = ctypes.util.find_library("CoreFoundation")
_cf = ctypes.cdll.LoadLibrary(_cf_path) if _cf_path else None

# libSystem for mach_task_self
_libsystem = ctypes.cdll.LoadLibrary("/usr/lib/libSystem.B.dylib")


# ============================================================================
# Type definitions
# ============================================================================

io_object_t = ctypes.c_uint32
io_connect_t = ctypes.c_uint32
io_service_t = ctypes.c_uint32
io_iterator_t = ctypes.c_uint32
mach_port_t = ctypes.c_uint32
kern_return_t = ctypes.c_int32

kIOMasterPortDefault = mach_port_t(0)
kIOMapAnywhere = 0x01
kIOReturnSuccess = 0


# ============================================================================
# IOKit function declarations
# ============================================================================

def _setup_iokit_functions():
    """Declare IOKit function signatures for ctypes."""

    # mach_task_self()
    _libsystem.mach_task_self_.restype = mach_port_t
    _libsystem.mach_task_self_.argtypes = []

    # IOServiceMatching(name) -> CFDictionaryRef
    _iokit.IOServiceMatching.restype = ctypes.c_void_p
    _iokit.IOServiceMatching.argtypes = [ctypes.c_char_p]

    # IOServiceGetMatchingService(masterPort, matching) -> io_service_t
    _iokit.IOServiceGetMatchingService.restype = io_service_t
    _iokit.IOServiceGetMatchingService.argtypes = [mach_port_t, ctypes.c_void_p]

    # IOServiceGetMatchingServices(masterPort, matching, &iterator)
    _iokit.IOServiceGetMatchingServices.restype = kern_return_t
    _iokit.IOServiceGetMatchingServices.argtypes = [
        mach_port_t, ctypes.c_void_p, ctypes.POINTER(io_iterator_t)]

    # IOIteratorNext(iterator) -> io_object_t
    _iokit.IOIteratorNext.restype = io_object_t
    _iokit.IOIteratorNext.argtypes = [io_iterator_t]

    # IOObjectRelease(object)
    _iokit.IOObjectRelease.restype = kern_return_t
    _iokit.IOObjectRelease.argtypes = [io_object_t]

    # IOServiceOpen(service, owningTask, type, &connect)
    _iokit.IOServiceOpen.restype = kern_return_t
    _iokit.IOServiceOpen.argtypes = [
        io_service_t, mach_port_t, ctypes.c_uint32,
        ctypes.POINTER(io_connect_t)]

    # IOServiceClose(connect)
    _iokit.IOServiceClose.restype = kern_return_t
    _iokit.IOServiceClose.argtypes = [io_connect_t]

    # IOConnectCallScalarMethod(connect, selector, input, inputCnt,
    #                           output, outputCnt)
    _iokit.IOConnectCallScalarMethod.restype = kern_return_t
    _iokit.IOConnectCallScalarMethod.argtypes = [
        io_connect_t, ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint32)]

    # IOConnectCallStructMethod(connect, selector, inputStruct, inputSize,
    #                           outputStruct, outputSize)
    _iokit.IOConnectCallStructMethod.restype = kern_return_t
    _iokit.IOConnectCallStructMethod.argtypes = [
        io_connect_t, ctypes.c_uint32,
        ctypes.c_void_p, ctypes.c_size_t,
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]

    # IOConnectMapMemory64(connect, memoryType, intoTask,
    #                      &address, &size, options)
    _iokit.IOConnectMapMemory64.restype = kern_return_t
    _iokit.IOConnectMapMemory64.argtypes = [
        io_connect_t, ctypes.c_uint32, mach_port_t,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_uint32]

    # IOConnectUnmapMemory64(connect, memoryType, intoTask, address)
    _iokit.IOConnectUnmapMemory64.restype = kern_return_t
    _iokit.IOConnectUnmapMemory64.argtypes = [
        io_connect_t, ctypes.c_uint32, mach_port_t, ctypes.c_uint64]

    # IORegistryEntryGetName(entry, name_buf)
    _iokit.IORegistryEntryGetName.restype = kern_return_t
    _iokit.IORegistryEntryGetName.argtypes = [io_object_t, ctypes.c_char_p]

_setup_iokit_functions()


def _mach_task_self() -> mach_port_t:
    return _libsystem.mach_task_self_()


# ============================================================================
# Selector constants (must match ROCmGPUShared.h)
# ============================================================================

class Selector(IntEnum):
    GET_INFO       = 0
    RESET          = 1
    CFG_READ       = 2
    CFG_WRITE      = 3
    MMIO_READ32    = 4
    MMIO_WRITE32   = 5
    MAP_BAR        = 6
    UNMAP_BAR      = 7
    ALLOC_DMA      = 8
    FREE_DMA       = 9
    MAP_DMA        = 10
    ENABLE_MSI     = 11
    WAIT_INTERRUPT = 12


class MemoryType(IntEnum):
    BAR0     = 0
    BAR1     = 1
    BAR2     = 2
    BAR3     = 3
    BAR4     = 4
    BAR5     = 5
    DMA_BASE = 0x100


# ============================================================================
# Device info structure (mirrors ROCmGPUDeviceInfo in ROCmGPUShared.h)
# ============================================================================

class BARInfoStruct(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_uint64),
        ("memoryIndex", ctypes.c_uint8),
        ("type", ctypes.c_uint8),
        ("is64bit", ctypes.c_uint8),
        ("prefetchable", ctypes.c_uint8),
        ("_pad2", ctypes.c_uint8 * 4),
    ]


class DeviceInfoStruct(ctypes.Structure):
    _fields_ = [
        ("vendorID", ctypes.c_uint16),
        ("deviceID", ctypes.c_uint16),
        ("subsystemVendorID", ctypes.c_uint16),
        ("subsystemDeviceID", ctypes.c_uint16),
        ("revisionID", ctypes.c_uint8),
        ("_pad", ctypes.c_uint8 * 3),
        ("bars", BARInfoStruct * 6),
        ("vramSize", ctypes.c_uint64),
    ]


class DMASegment(ctypes.Structure):
    _fields_ = [
        ("address", ctypes.c_uint64),
        ("length", ctypes.c_uint64),
    ]


class DMAInfoStruct(ctypes.Structure):
    _fields_ = [
        ("bufferID", ctypes.c_uint64),
        ("size", ctypes.c_uint64),
        ("segmentCount", ctypes.c_uint32),
        ("_pad", ctypes.c_uint32),
        ("segments", DMASegment * 64),
    ]


# ============================================================================
# Device info dataclass (Python-friendly)
# ============================================================================

@dataclass
class BARInfo:
    """Information about a single PCI BAR."""
    index: int
    size: int
    memory_index: int
    bar_type: int  # 0=memory, 1=IO, 2=not present
    is_64bit: bool
    prefetchable: bool


@dataclass
class DeviceInfo:
    """Parsed device information from ROCmGPU.dext."""
    vendor_id: int
    device_id: int
    subsystem_vendor_id: int
    subsystem_device_id: int
    revision_id: int
    bars: list[BARInfo]
    vram_size: int


@dataclass
class DMAAllocation:
    """Represents an allocated DMA buffer."""
    buffer_id: int
    size: int
    cpu_addr: int  # Mapped virtual address in this process
    segments: list[tuple[int, int]]  # (phys_addr, length) pairs


# ============================================================================
# IOKit Client
# ============================================================================

class IOKitClient:
    """Communicates with ROCmGPU.dext via IOKit user client interface.

    This is the macOS equivalent of Windows' DriverInterface class.
    All GPU register access, memory allocation, and BAR mapping flows
    through this client.

    Usage:
        client = IOKitClient()
        client.open()
        info = client.get_info()
        print(f"GPU: vendor=0x{info.vendor_id:04x} device=0x{info.device_id:04x}")
        client.close()
    """

    DEXT_SERVICE_NAME = b"ROCmGPUDriver"

    def __init__(self) -> None:
        self._connection = io_connect_t(0)
        self._service = io_service_t(0)
        self._opened = False
        self._mapped_bars: dict[int, tuple[int, int]] = {}  # bar -> (addr, size)
        self._mapped_dma: dict[int, tuple[int, int]] = {}   # bufID -> (addr, size)

    def open(self, service_name: bytes | None = None) -> None:
        """Find and open a connection to the ROCmGPU DEXT."""
        if self._opened:
            raise RuntimeError("Already connected to DEXT")

        name = service_name or self.DEXT_SERVICE_NAME

        # Find the service in IOKit registry
        matching = _iokit.IOServiceMatching(name)
        if not matching:
            raise RuntimeError(
                f"IOServiceMatching({name!r}) returned NULL — "
                "is the DEXT installed?"
            )

        service = _iokit.IOServiceGetMatchingService(
            kIOMasterPortDefault, matching)
        if service == 0:
            raise RuntimeError(
                f"No IOKit service found for {name.decode()!r}. "
                "Ensure ROCmGPU.dext is installed and an AMD eGPU is connected."
            )

        # Open a user client connection
        connection = io_connect_t(0)
        ret = _iokit.IOServiceOpen(
            service, _mach_task_self(), 0, ctypes.byref(connection))
        if ret != kIOReturnSuccess:
            _iokit.IOObjectRelease(service)
            raise RuntimeError(
                f"IOServiceOpen failed: 0x{ret:08x}. "
                "Check DEXT entitlements and user client access."
            )

        self._service = service
        self._connection = connection
        self._opened = True

    def close(self) -> None:
        """Close the connection and release resources."""
        if not self._opened:
            return

        # Unmap all BAR and DMA mappings
        task = _mach_task_self()
        for bar_idx, (addr, _) in self._mapped_bars.items():
            _iokit.IOConnectUnmapMemory64(
                self._connection, bar_idx, task, addr)
        self._mapped_bars.clear()

        for buf_id, (addr, _) in self._mapped_dma.items():
            _iokit.IOConnectUnmapMemory64(
                self._connection, MemoryType.DMA_BASE + buf_id, task, addr)
        self._mapped_dma.clear()

        _iokit.IOServiceClose(self._connection)
        _iokit.IOObjectRelease(self._service)
        self._connection = io_connect_t(0)
        self._service = io_service_t(0)
        self._opened = False

    def _check_open(self) -> None:
        if not self._opened:
            raise RuntimeError("Not connected to DEXT — call open() first")

    # ---- Scalar method helpers ----

    def _call_scalar(
        self,
        selector: int,
        inputs: list[int],
        output_count: int = 0,
    ) -> list[int]:
        """Call IOConnectCallScalarMethod with uint64 scalars."""
        self._check_open()

        in_arr = (ctypes.c_uint64 * len(inputs))(*inputs) if inputs else None
        out_arr = (ctypes.c_uint64 * output_count)() if output_count > 0 else None
        out_cnt = ctypes.c_uint32(output_count)

        ret = _iokit.IOConnectCallScalarMethod(
            self._connection,
            selector,
            in_arr, len(inputs),
            out_arr, ctypes.byref(out_cnt) if output_count > 0 else None,
        )

        if ret != kIOReturnSuccess:
            raise RuntimeError(
                f"IOConnectCallScalarMethod(sel={selector}) "
                f"failed: 0x{ret:08x}"
            )

        return [out_arr[i] for i in range(out_cnt.value)] if out_arr else []

    def _call_struct(
        self,
        selector: int,
        input_data: bytes | None = None,
        output_type: type[ctypes.Structure] | None = None,
    ) -> ctypes.Structure | None:
        """Call IOConnectCallStructMethod with struct in/out."""
        self._check_open()

        in_ptr = None
        in_size = 0
        if input_data:
            in_buf = ctypes.create_string_buffer(input_data)
            in_ptr = ctypes.cast(in_buf, ctypes.c_void_p)
            in_size = len(input_data)

        out_struct = None
        out_ptr = None
        out_size = ctypes.c_size_t(0)
        if output_type:
            out_struct = output_type()
            out_ptr = ctypes.cast(ctypes.pointer(out_struct), ctypes.c_void_p)
            out_size = ctypes.c_size_t(ctypes.sizeof(output_type))

        ret = _iokit.IOConnectCallStructMethod(
            self._connection,
            selector,
            in_ptr, in_size,
            out_ptr, ctypes.byref(out_size) if out_ptr else None,
        )

        if ret != kIOReturnSuccess:
            raise RuntimeError(
                f"IOConnectCallStructMethod(sel={selector}) "
                f"failed: 0x{ret:08x}"
            )

        return out_struct

    # ---- Memory mapping ----

    def _map_memory(self, memory_type: int) -> tuple[int, int]:
        """Map a DEXT memory region into this process. Returns (addr, size)."""
        self._check_open()

        addr = ctypes.c_uint64(0)
        size = ctypes.c_uint64(0)

        ret = _iokit.IOConnectMapMemory64(
            self._connection,
            memory_type,
            _mach_task_self(),
            ctypes.byref(addr),
            ctypes.byref(size),
            kIOMapAnywhere,
        )

        if ret != kIOReturnSuccess:
            raise RuntimeError(
                f"IOConnectMapMemory64(type=0x{memory_type:x}) "
                f"failed: 0x{ret:08x}"
            )

        return addr.value, size.value

    # ================================================================
    # Public API — matches the escape command set in ROCmGPUShared.h
    # ================================================================

    def get_info(self) -> DeviceInfo:
        """Query device info (vendor/device ID, BARs, VRAM size)."""
        raw = self._call_struct(Selector.GET_INFO, output_type=DeviceInfoStruct)
        if raw is None:
            raise RuntimeError("GetInfo returned no data")

        bars = []
        for i in range(6):
            b = raw.bars[i]
            bars.append(BARInfo(
                index=i,
                size=b.size,
                memory_index=b.memoryIndex,
                bar_type=b.type,
                is_64bit=bool(b.is64bit),
                prefetchable=bool(b.prefetchable),
            ))

        return DeviceInfo(
            vendor_id=raw.vendorID,
            device_id=raw.deviceID,
            subsystem_vendor_id=raw.subsystemVendorID,
            subsystem_device_id=raw.subsystemDeviceID,
            revision_id=raw.revisionID,
            bars=bars,
            vram_size=raw.vramSize,
        )

    def reset(self) -> None:
        """Perform GPU Function-Level Reset."""
        self._call_scalar(Selector.RESET, [])

    def cfg_read(self, offset: int, width: int = 4) -> int:
        """Read PCI configuration space register."""
        result = self._call_scalar(Selector.CFG_READ, [offset, width], 1)
        return result[0]

    def cfg_write(self, offset: int, value: int, width: int = 4) -> None:
        """Write PCI configuration space register."""
        self._call_scalar(Selector.CFG_WRITE, [offset, width, value])

    def mmio_read32(self, bar: int, offset: int) -> int:
        """Read 32-bit MMIO register from a PCI BAR."""
        result = self._call_scalar(Selector.MMIO_READ32, [bar, offset], 1)
        return result[0] & 0xFFFFFFFF

    def mmio_write32(self, bar: int, offset: int, value: int) -> None:
        """Write 32-bit MMIO register to a PCI BAR."""
        self._call_scalar(Selector.MMIO_WRITE32, [bar, offset, value & 0xFFFFFFFF])

    def map_bar(self, bar_index: int) -> tuple[int, int]:
        """Map a PCI BAR into this process address space.

        Returns (virtual_address, size). The mapped region provides
        direct MMIO access -- reads/writes go straight to GPU registers
        or VRAM without kernel round-trips. Use this for bulk access
        (IP discovery, firmware loading, page table writes).
        """
        if bar_index in self._mapped_bars:
            return self._mapped_bars[bar_index]

        # Tell DEXT we want to map this BAR (validates it exists)
        size_result = self._call_scalar(Selector.MAP_BAR, [bar_index], 1)

        # Actually map it into our address space
        addr, size = self._map_memory(bar_index)
        self._mapped_bars[bar_index] = (addr, size)
        return addr, size

    def unmap_bar(self, bar_index: int) -> None:
        """Unmap a previously mapped PCI BAR."""
        if bar_index not in self._mapped_bars:
            return

        addr, _ = self._mapped_bars.pop(bar_index)
        _iokit.IOConnectUnmapMemory64(
            self._connection, bar_index, _mach_task_self(), addr)
        self._call_scalar(Selector.UNMAP_BAR, [bar_index])

    def alloc_dma(self, size: int, flags: int = 0) -> DMAAllocation:
        """Allocate a DMA-capable buffer.

        The buffer is allocated by the DEXT (via IOBufferMemoryDescriptor +
        IODMACommand), which returns IOMMU-translated physical addresses
        that the GPU can use for DMA.

        Returns a DMAAllocation with:
          - buffer_id: opaque ID for free/map calls
          - size: actual allocation size
          - cpu_addr: virtual address mapped into this process
          - segments: list of (phys_addr, length) for GPU page tables
        """
        raw = self._call_struct(
            Selector.ALLOC_DMA,
            input_data=struct.pack("<QI", size, flags),
            output_type=DMAInfoStruct,
        )
        if raw is None:
            raise RuntimeError("AllocDMA returned no data")

        buf_id = raw.bufferID

        # Map the DMA buffer into our process
        addr, mapped_size = self._map_memory(MemoryType.DMA_BASE + buf_id)
        self._mapped_dma[buf_id] = (addr, mapped_size)

        segments = []
        for i in range(raw.segmentCount):
            seg = raw.segments[i]
            if seg.length > 0:
                segments.append((seg.address, seg.length))

        return DMAAllocation(
            buffer_id=buf_id,
            size=raw.size,
            cpu_addr=addr,
            segments=segments,
        )

    def free_dma(self, buffer_id: int) -> None:
        """Free a previously allocated DMA buffer."""
        # Unmap from our process first
        if buffer_id in self._mapped_dma:
            addr, _ = self._mapped_dma.pop(buffer_id)
            _iokit.IOConnectUnmapMemory64(
                self._connection, MemoryType.DMA_BASE + buffer_id,
                _mach_task_self(), addr)

        self._call_scalar(Selector.FREE_DMA, [buffer_id])

    def enable_msi(self, vector_index: int = 0) -> None:
        """Enable MSI-X interrupt for the given vector."""
        self._call_scalar(Selector.ENABLE_MSI, [vector_index])

    def wait_interrupt(self, timeout_ms: int = 5000) -> int:
        """Wait for an interrupt. Returns status (0=OK, 1=timeout, 2=error)."""
        result = self._call_scalar(Selector.WAIT_INTERRUPT, [timeout_ms], 1)
        return result[0]

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        if self._opened:
            self.close()
