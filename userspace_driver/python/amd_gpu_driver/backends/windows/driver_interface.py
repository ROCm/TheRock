"""Low-level ctypes bindings for D3DKMT APIs and escape commands.

Provides the Python ↔ kernel communication channel:
  Python → gdi32.D3DKMTEscape → dxgkrnl.sys → DxgkDdiEscape → amdgpu_wddm.sys

The escape buffer carries an AMDGPU_ESCAPE_HEADER followed by command-specific
data, matching the structures defined in wddm_driver/amdgpu_wddm.h.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wintypes
import sys
from dataclasses import dataclass
from enum import IntEnum

if sys.platform != "win32":
    raise ImportError("Windows driver interface is only available on Windows")

# ============================================================================
# D3DKMT API bindings (gdi32.dll)
#
# CRITICAL: D3DKMT_HANDLE is UINT (4 bytes), NOT HANDLE (8 bytes on x64).
# Using wintypes.HANDLE causes struct misalignment and silent failures.
# ============================================================================

gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)

# D3DKMT_HANDLE is typedef UINT, always 4 bytes
D3DKMT_HANDLE = ctypes.c_uint32


class LUID(ctypes.Structure):
    """Locally Unique Identifier — used to identify display adapters."""
    _fields_ = [
        ("LowPart", wintypes.DWORD),
        ("HighPart", wintypes.LONG),
    ]


# --- D3DKMTEnumAdapters2 ---

class D3DKMT_ADAPTERINFO(ctypes.Structure):
    _fields_ = [
        ("hAdapter", D3DKMT_HANDLE),
        ("AdapterLuid", LUID),
        ("NumOfSources", ctypes.c_uint32),
        ("bPrecisePresentRegionsPreferred", wintypes.BOOL),
    ]


class D3DKMT_ENUMADAPTERS2(ctypes.Structure):
    _fields_ = [
        ("NumAdapters", ctypes.c_uint32),
        ("pAdapters", ctypes.POINTER(D3DKMT_ADAPTERINFO)),
    ]


# --- D3DKMTOpenAdapterFromLuid ---

class D3DKMT_OPENADAPTERFROMLUID(ctypes.Structure):
    _fields_ = [
        ("AdapterLuid", LUID),
        ("hAdapter", D3DKMT_HANDLE),
    ]


# --- D3DKMTCloseAdapter ---

class D3DKMT_CLOSEADAPTER(ctypes.Structure):
    _fields_ = [
        ("hAdapter", D3DKMT_HANDLE),
    ]


# --- D3DKMTEscape ---

D3DKMT_ESCAPE_DRIVERPRIVATE = 0

class D3DKMT_ESCAPE(ctypes.Structure):
    _fields_ = [
        ("hAdapter", D3DKMT_HANDLE),
        ("hDevice", D3DKMT_HANDLE),
        ("Type", ctypes.c_uint32),
        ("Flags", ctypes.c_uint32),
        ("pPrivateDriverData", ctypes.c_void_p),
        ("PrivateDriverDataSize", ctypes.c_uint32),
        ("hContext", D3DKMT_HANDLE),
    ]


# --- D3DKMTCreateDevice ---

class D3DKMT_CREATEDEVICE(ctypes.Structure):
    _fields_ = [
        ("hAdapter", D3DKMT_HANDLE),
        ("pCommandBuffer", ctypes.c_void_p),
        ("CommandBufferSize", ctypes.c_uint32),
        ("pAllocationList", ctypes.c_void_p),
        ("AllocationListSize", ctypes.c_uint32),
        ("pPatchLocationList", ctypes.c_void_p),
        ("PatchLocationListSize", ctypes.c_uint32),
        ("hDevice", D3DKMT_HANDLE),
    ]


class D3DKMT_DESTROYDEVICE(ctypes.Structure):
    _fields_ = [
        ("hDevice", D3DKMT_HANDLE),
    ]


# --- D3DKMTQueryAdapterInfo ---

KMTQAITYPE_DRIVER_DESCRIPTION = 76  # Gets driver description string

class D3DKMT_QUERYADAPTERINFO(ctypes.Structure):
    _fields_ = [
        ("hAdapter", D3DKMT_HANDLE),
        ("Type", ctypes.c_uint32),
        ("pPrivateDriverData", ctypes.c_void_p),
        ("PrivateDriverDataSize", ctypes.c_uint32),
    ]


# ============================================================================
# Set up D3DKMT function prototypes
# ============================================================================

_D3DKMTEnumAdapters2 = gdi32.D3DKMTEnumAdapters2
_D3DKMTOpenAdapterFromLuid = gdi32.D3DKMTOpenAdapterFromLuid
_D3DKMTCloseAdapter = gdi32.D3DKMTCloseAdapter
_D3DKMTEscape = gdi32.D3DKMTEscape
_D3DKMTCreateDevice = gdi32.D3DKMTCreateDevice
_D3DKMTDestroyDevice = gdi32.D3DKMTDestroyDevice
_D3DKMTQueryAdapterInfo = gdi32.D3DKMTQueryAdapterInfo


def _check_ntstatus(status: int, api_name: str) -> None:
    """Raise on NTSTATUS failure (negative = error)."""
    # NTSTATUS: bit 31 set = error
    if status < 0 or (status & 0x80000000):
        raise RuntimeError(f"{api_name} failed: NTSTATUS 0x{status & 0xFFFFFFFF:08X}")


# ============================================================================
# Escape command codes (must match kernel_driver/amdgpu_mcdm.h)
# ============================================================================

class EscapeCode(IntEnum):
    GET_INFO = 0x0001
    READ_REG32 = 0x0010
    WRITE_REG32 = 0x0011
    MAP_BAR = 0x0020
    UNMAP_BAR = 0x0021
    ALLOC_DMA = 0x0030
    FREE_DMA = 0x0031
    MAP_VRAM = 0x0040
    REGISTER_EVENT = 0x0050
    ENABLE_MSI = 0x0051
    GET_IOMMU_INFO = 0x0060


# ============================================================================
# Escape command structures (must match kernel_driver/amdgpu_mcdm.h)
# ============================================================================

class EscapeHeader(ctypes.Structure):
    _fields_ = [
        ("Command", ctypes.c_uint32),   # EscapeCode
        ("Status", ctypes.c_int32),     # NTSTATUS, filled by driver
        ("Size", ctypes.c_uint32),      # Total size including header
    ]


class BarInfo(ctypes.Structure):
    _fields_ = [
        ("PhysicalAddress", ctypes.c_int64),
        ("Length", ctypes.c_uint64),
        ("IsMemory", ctypes.c_uint8),
        ("Is64Bit", ctypes.c_uint8),
        ("IsPrefetchable", ctypes.c_uint8),
        ("Reserved", ctypes.c_uint8),
    ]


class EscapeGetInfoData(ctypes.Structure):
    _fields_ = [
        ("Header", EscapeHeader),
        ("VendorId", ctypes.c_uint16),
        ("DeviceId", ctypes.c_uint16),
        ("SubsystemVendorId", ctypes.c_uint16),
        ("SubsystemId", ctypes.c_uint16),
        ("RevisionId", ctypes.c_uint8),
        ("Reserved", ctypes.c_uint8 * 3),
        ("NumBars", ctypes.c_uint32),
        ("Bars", BarInfo * 6),
        ("VramSizeBytes", ctypes.c_uint64),
        ("VisibleVramSizeBytes", ctypes.c_uint64),
        ("MmioBarIndex", ctypes.c_uint32),
        ("VramBarIndex", ctypes.c_uint32),
        ("Headless", ctypes.c_uint8),
        ("Reserved2", ctypes.c_uint8 * 3),
    ]


class EscapeReg32Data(ctypes.Structure):
    _fields_ = [
        ("Header", EscapeHeader),
        ("BarIndex", ctypes.c_uint32),
        ("Offset", ctypes.c_uint32),
        ("Value", ctypes.c_uint32),
    ]


class EscapeMapBarData(ctypes.Structure):
    _fields_ = [
        ("Header", EscapeHeader),
        ("BarIndex", ctypes.c_uint32),
        ("Offset", ctypes.c_uint64),
        ("Length", ctypes.c_uint64),
        ("MappedAddress", ctypes.c_void_p),
        ("MappingHandle", ctypes.c_void_p),
    ]


class EscapeAllocDmaData(ctypes.Structure):
    _fields_ = [
        ("Header", EscapeHeader),
        ("Size", ctypes.c_uint64),
        ("CpuAddress", ctypes.c_void_p),
        ("BusAddress", ctypes.c_uint64),
        ("AllocationHandle", ctypes.c_void_p),
    ]


class EscapeMapVramData(ctypes.Structure):
    _fields_ = [
        ("Header", EscapeHeader),
        ("Offset", ctypes.c_uint64),
        ("Length", ctypes.c_uint64),
        ("MappedAddress", ctypes.c_void_p),
        ("MappingHandle", ctypes.c_void_p),
    ]


class EscapeRegisterEventData(ctypes.Structure):
    _fields_ = [
        ("Header", EscapeHeader),
        ("EventHandle", wintypes.HANDLE),
        ("InterruptSource", ctypes.c_uint32),
        ("RegistrationId", ctypes.c_uint32),
    ]


class EscapeEnableMsiData(ctypes.Structure):
    _fields_ = [
        ("Header", EscapeHeader),
        # Input: IH ring configuration
        ("IhRingDmaHandle", ctypes.c_void_p),
        ("IhRingSize", ctypes.c_uint32),
        ("IhRptrRegOffset", ctypes.c_uint32),
        ("IhWptrRegOffset", ctypes.c_uint32),
        # Output
        ("Enabled", ctypes.c_uint8),
        ("Reserved", ctypes.c_uint8 * 3),
        ("NumVectors", ctypes.c_uint32),
    ]


class EscapeGetIommuInfoData(ctypes.Structure):
    _fields_ = [
        ("Header", EscapeHeader),
        ("IommuPresent", ctypes.c_uint8),
        ("IommuEnabled", ctypes.c_uint8),
        ("DmaRemappingActive", ctypes.c_uint8),
        ("Reserved", ctypes.c_uint8),
    ]


# ============================================================================
# Device info result
# ============================================================================

@dataclass
class DeviceInfo:
    """Parsed result from ESCAPE_GET_INFO."""
    vendor_id: int
    device_id: int
    subsystem_vendor_id: int
    subsystem_id: int
    revision_id: int
    num_bars: int
    bars: list[dict[str, int | bool]]
    vram_size: int
    visible_vram_size: int
    mmio_bar_index: int
    vram_bar_index: int
    headless: bool


# ============================================================================
# DriverInterface — high-level wrapper for D3DKMT + escape commands
# ============================================================================

class DriverInterface:
    """Python interface to the amdgpu_mcdm.sys kernel driver.

    Handles adapter enumeration, D3DKMT device creation, and escape
    command dispatch.
    """

    def __init__(self) -> None:
        self._adapter_handle: int | None = None
        self._device_handle: int | None = None
        self._adapter_luid: LUID | None = None

    def enumerate_adapters(self) -> list[D3DKMT_ADAPTERINFO]:
        """Enumerate all WDDM display adapters.

        Returns list of adapter info structs. Our MCDM device will
        appear as a ComputeAccelerator with NumOfSources=0.
        """
        # First call: get count
        args = D3DKMT_ENUMADAPTERS2()
        args.NumAdapters = 0
        args.pAdapters = None

        status = _D3DKMTEnumAdapters2(ctypes.byref(args))
        _check_ntstatus(status, "D3DKMTEnumAdapters2 (count)")

        if args.NumAdapters == 0:
            return []

        # Second call: get adapter info
        adapter_array = (D3DKMT_ADAPTERINFO * args.NumAdapters)()
        args.pAdapters = adapter_array

        status = _D3DKMTEnumAdapters2(ctypes.byref(args))
        _check_ntstatus(status, "D3DKMTEnumAdapters2 (list)")

        return list(adapter_array[:args.NumAdapters])

    def open_adapter(self, luid: LUID) -> None:
        """Open an adapter by its LUID."""
        args = D3DKMT_OPENADAPTERFROMLUID()
        args.AdapterLuid = luid

        status = _D3DKMTOpenAdapterFromLuid(ctypes.byref(args))
        _check_ntstatus(status, "D3DKMTOpenAdapterFromLuid")

        self._adapter_handle = args.hAdapter
        self._adapter_luid = luid

    def create_device(self) -> None:
        """Create a D3DKMT device on the opened adapter.

        Required before D3DKMTEscape can be called (some versions of
        dxgkrnl require a valid hDevice).
        """
        if self._adapter_handle is None:
            raise RuntimeError("No adapter open — call open_adapter first")

        args = D3DKMT_CREATEDEVICE()
        args.hAdapter = self._adapter_handle

        status = _D3DKMTCreateDevice(ctypes.byref(args))
        _check_ntstatus(status, "D3DKMTCreateDevice")

        self._device_handle = args.hDevice

    def close(self) -> None:
        """Close the device and adapter handles."""
        if self._device_handle is not None:
            args = D3DKMT_DESTROYDEVICE()
            args.hDevice = self._device_handle
            _D3DKMTDestroyDevice(ctypes.byref(args))
            self._device_handle = None

        if self._adapter_handle is not None:
            args_close = D3DKMT_CLOSEADAPTER()
            args_close.hAdapter = self._adapter_handle
            _D3DKMTCloseAdapter(ctypes.byref(args_close))
            self._adapter_handle = None

    def escape(self, command_buffer: ctypes.Structure) -> None:
        """Send an escape command to the MCDM driver.

        The command_buffer must start with an EscapeHeader.
        After return, check command_buffer.Header.Status for the
        NTSTATUS result from the kernel driver.

        Raises RuntimeError if D3DKMTEscape itself fails (transport error).
        Driver-level errors are reported in Header.Status.
        """
        if self._adapter_handle is None:
            raise RuntimeError("No adapter open")

        args = D3DKMT_ESCAPE()
        args.hAdapter = self._adapter_handle
        args.hDevice = self._device_handle if self._device_handle is not None else 0
        args.Type = D3DKMT_ESCAPE_DRIVERPRIVATE
        args.Flags = 0
        args.pPrivateDriverData = ctypes.addressof(command_buffer)
        args.PrivateDriverDataSize = ctypes.sizeof(command_buffer)
        args.hContext = 0

        status = _D3DKMTEscape(ctypes.byref(args))
        _check_ntstatus(status, "D3DKMTEscape")

    # ---- Convenience methods for specific escape commands ----

    def get_info(self) -> DeviceInfo:
        """Query device information via ESCAPE_GET_INFO."""
        cmd = EscapeGetInfoData()
        cmd.Header.Command = EscapeCode.GET_INFO
        cmd.Header.Size = ctypes.sizeof(cmd)

        self.escape(cmd)

        bars = []
        for i in range(min(cmd.NumBars, 6)):
            b = cmd.Bars[i]
            bars.append({
                "physical_address": b.PhysicalAddress,
                "length": b.Length,
                "is_memory": bool(b.IsMemory),
                "is_64bit": bool(b.Is64Bit),
                "is_prefetchable": bool(b.IsPrefetchable),
            })

        return DeviceInfo(
            vendor_id=cmd.VendorId,
            device_id=cmd.DeviceId,
            subsystem_vendor_id=cmd.SubsystemVendorId,
            subsystem_id=cmd.SubsystemId,
            revision_id=cmd.RevisionId,
            num_bars=cmd.NumBars,
            bars=bars,
            vram_size=cmd.VramSizeBytes,
            visible_vram_size=cmd.VisibleVramSizeBytes,
            mmio_bar_index=cmd.MmioBarIndex,
            vram_bar_index=cmd.VramBarIndex,
            headless=bool(cmd.Headless),
        )

    def read_reg32(self, offset: int, bar_index: int = 0) -> int:
        """Read a 32-bit MMIO register."""
        cmd = EscapeReg32Data()
        cmd.Header.Command = EscapeCode.READ_REG32
        cmd.Header.Size = ctypes.sizeof(cmd)
        cmd.BarIndex = bar_index
        cmd.Offset = offset

        self.escape(cmd)

        if cmd.Header.Status != 0:
            raise RuntimeError(
                f"READ_REG32(bar={bar_index}, offset=0x{offset:X}) failed: "
                f"NTSTATUS 0x{cmd.Header.Status & 0xFFFFFFFF:08X}"
            )
        return cmd.Value

    def write_reg32(self, offset: int, value: int, bar_index: int = 0) -> None:
        """Write a 32-bit MMIO register."""
        cmd = EscapeReg32Data()
        cmd.Header.Command = EscapeCode.WRITE_REG32
        cmd.Header.Size = ctypes.sizeof(cmd)
        cmd.BarIndex = bar_index
        cmd.Offset = offset
        cmd.Value = value

        self.escape(cmd)

        if cmd.Header.Status != 0:
            raise RuntimeError(
                f"WRITE_REG32(bar={bar_index}, offset=0x{offset:X}, "
                f"value=0x{value:X}) failed: "
                f"NTSTATUS 0x{cmd.Header.Status & 0xFFFFFFFF:08X}"
            )

    def map_bar(self, bar_index: int, offset: int = 0, length: int = 0) -> tuple[int, int]:
        """Map a PCI BAR region into this process's address space.

        Returns (mapped_address, mapping_handle).
        """
        cmd = EscapeMapBarData()
        cmd.Header.Command = EscapeCode.MAP_BAR
        cmd.Header.Size = ctypes.sizeof(cmd)
        cmd.BarIndex = bar_index
        cmd.Offset = offset
        cmd.Length = length

        self.escape(cmd)

        if cmd.Header.Status != 0:
            raise RuntimeError(
                f"MAP_BAR(bar={bar_index}) failed: "
                f"NTSTATUS 0x{cmd.Header.Status & 0xFFFFFFFF:08X}"
            )
        return (cmd.MappedAddress, cmd.MappingHandle)

    def unmap_bar(self, mapping_handle: int) -> None:
        """Unmap a previously mapped BAR region."""
        cmd = EscapeMapBarData()
        cmd.Header.Command = EscapeCode.UNMAP_BAR
        cmd.Header.Size = ctypes.sizeof(cmd)
        cmd.MappingHandle = mapping_handle

        self.escape(cmd)

    def alloc_dma(self, size: int) -> tuple[int, int, int]:
        """Allocate contiguous DMA memory.

        Returns (cpu_address, bus_address, allocation_handle).
        """
        cmd = EscapeAllocDmaData()
        cmd.Header.Command = EscapeCode.ALLOC_DMA
        cmd.Header.Size = ctypes.sizeof(cmd)
        cmd.Size = size

        self.escape(cmd)

        if cmd.Header.Status != 0:
            raise RuntimeError(
                f"ALLOC_DMA(size={size}) failed: "
                f"NTSTATUS 0x{cmd.Header.Status & 0xFFFFFFFF:08X}"
            )
        return (cmd.CpuAddress, cmd.BusAddress, cmd.AllocationHandle)

    def free_dma(self, allocation_handle: int) -> None:
        """Free a previously allocated DMA buffer."""
        cmd = EscapeAllocDmaData()
        cmd.Header.Command = EscapeCode.FREE_DMA
        cmd.Header.Size = ctypes.sizeof(cmd)
        cmd.AllocationHandle = allocation_handle

        self.escape(cmd)

    def map_vram(self, offset: int, length: int) -> tuple[int, int]:
        """Map a VRAM region via BAR2 into this process's address space.

        Returns (mapped_address, mapping_handle).
        """
        cmd = EscapeMapVramData()
        cmd.Header.Command = EscapeCode.MAP_VRAM
        cmd.Header.Size = ctypes.sizeof(cmd)
        cmd.Offset = offset
        cmd.Length = length

        self.escape(cmd)

        if cmd.Header.Status != 0:
            raise RuntimeError(
                f"MAP_VRAM(offset=0x{offset:X}, length={length}) failed: "
                f"NTSTATUS 0x{cmd.Header.Status & 0xFFFFFFFF:08X}"
            )
        return (cmd.MappedAddress, cmd.MappingHandle)

    def register_event(
        self, event_handle: int, source_id: int
    ) -> int:
        """Register a Windows Event to be signaled on GPU interrupt.

        Args:
            event_handle: Win32 event HANDLE (from CreateEvent).
            source_id: IH source ID to match (e.g., CP fence, SDMA, etc.).

        Returns:
            Registration ID for later deregistration.
        """
        cmd = EscapeRegisterEventData()
        cmd.Header.Command = EscapeCode.REGISTER_EVENT
        cmd.Header.Size = ctypes.sizeof(cmd)
        cmd.EventHandle = event_handle
        cmd.InterruptSource = source_id

        self.escape(cmd)

        if cmd.Header.Status != 0:
            raise RuntimeError(
                f"REGISTER_EVENT(source={source_id}) failed: "
                f"NTSTATUS 0x{cmd.Header.Status & 0xFFFFFFFF:08X}"
            )
        return cmd.RegistrationId

    def enable_msi(
        self,
        ih_ring_dma_handle: int,
        ih_ring_size: int,
        rptr_reg_offset: int,
        wptr_reg_offset: int,
    ) -> tuple[bool, int]:
        """Configure IH ring for interrupt processing.

        Python allocates the IH ring buffer (ALLOC_DMA), programs IH
        hardware registers (WRITE_REG32), then calls this to tell the
        kernel where the ring is and which register offsets to use.

        Args:
            ih_ring_dma_handle: DMA allocation handle from alloc_dma().
            ih_ring_size: Ring buffer size in bytes (power of 2).
            rptr_reg_offset: BAR0 byte offset of IH_RB_RPTR register.
            wptr_reg_offset: BAR0 byte offset of IH_RB_WPTR register.

        Returns:
            (enabled, num_vectors) tuple.
        """
        cmd = EscapeEnableMsiData()
        cmd.Header.Command = EscapeCode.ENABLE_MSI
        cmd.Header.Size = ctypes.sizeof(cmd)
        cmd.IhRingDmaHandle = ih_ring_dma_handle
        cmd.IhRingSize = ih_ring_size
        cmd.IhRptrRegOffset = rptr_reg_offset
        cmd.IhWptrRegOffset = wptr_reg_offset

        self.escape(cmd)

        if cmd.Header.Status != 0:
            raise RuntimeError(
                f"ENABLE_MSI failed: "
                f"NTSTATUS 0x{cmd.Header.Status & 0xFFFFFFFF:08X}"
            )
        return (bool(cmd.Enabled), cmd.NumVectors)

    def get_iommu_info(self) -> dict[str, bool]:
        """Query IOMMU status."""
        cmd = EscapeGetIommuInfoData()
        cmd.Header.Command = EscapeCode.GET_IOMMU_INFO
        cmd.Header.Size = ctypes.sizeof(cmd)

        self.escape(cmd)

        return {
            "iommu_present": bool(cmd.IommuPresent),
            "iommu_enabled": bool(cmd.IommuEnabled),
            "dma_remapping_active": bool(cmd.DmaRemappingActive),
        }
