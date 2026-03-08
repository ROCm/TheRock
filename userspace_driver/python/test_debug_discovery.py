"""Debug script for adapter discovery on Windows.

Key insight: D3DKMT uses D3DKMT_HANDLE which is UINT (4 bytes), not HANDLE (8 bytes).
"""
from __future__ import annotations
import sys
import ctypes
import ctypes.wintypes as wintypes

if sys.platform != "win32":
    print("This script must run on Windows")
    sys.exit(1)

gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)

# D3DKMT_HANDLE is UINT (4 bytes), NOT HANDLE (8 bytes on x64)
D3DKMT_HANDLE = ctypes.c_uint32


class LUID(ctypes.Structure):
    _fields_ = [
        ("LowPart", wintypes.DWORD),
        ("HighPart", wintypes.LONG),
    ]


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


class D3DKMT_OPENADAPTERFROMLUID(ctypes.Structure):
    _fields_ = [
        ("AdapterLuid", LUID),
        ("hAdapter", D3DKMT_HANDLE),
    ]


class D3DKMT_CLOSEADAPTER(ctypes.Structure):
    _fields_ = [("hAdapter", D3DKMT_HANDLE)]


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
    _fields_ = [("hDevice", D3DKMT_HANDLE)]


class EscapeHeader(ctypes.Structure):
    _fields_ = [
        ("Command", ctypes.c_uint32),
        ("Status", ctypes.c_int32),
        ("Size", ctypes.c_uint32),
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
    ]


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


_EnumAdapters2 = gdi32.D3DKMTEnumAdapters2
_OpenFromLuid = gdi32.D3DKMTOpenAdapterFromLuid
_CloseAdapter = gdi32.D3DKMTCloseAdapter
_CreateDevice = gdi32.D3DKMTCreateDevice
_DestroyDevice = gdi32.D3DKMTDestroyDevice
_Escape = gdi32.D3DKMTEscape


def main() -> int:
    # Enumerate
    args = D3DKMT_ENUMADAPTERS2()
    args.NumAdapters = 0
    args.pAdapters = None
    _EnumAdapters2(ctypes.byref(args))
    count = args.NumAdapters

    adapter_array = (D3DKMT_ADAPTERINFO * count)()
    args.pAdapters = adapter_array
    _EnumAdapters2(ctypes.byref(args))

    print(f"Found {count} adapters")
    print(f"sizeof(D3DKMT_CREATEDEVICE) = {ctypes.sizeof(D3DKMT_CREATEDEVICE)}")
    print(f"sizeof(D3DKMT_ESCAPE) = {ctypes.sizeof(D3DKMT_ESCAPE)}")
    print()

    for i in range(min(count, 15)):
        ai = adapter_array[i]
        luid = ai.AdapterLuid
        if luid.LowPart == 0 and luid.HighPart == 0:
            continue

        # Open
        open_args = D3DKMT_OPENADAPTERFROMLUID()
        open_args.AdapterLuid = luid
        s = _OpenFromLuid(ctypes.byref(open_args))
        if s < 0 or (s & 0x80000000):
            continue
        h_adapter = open_args.hAdapter

        # Create device
        cd = D3DKMT_CREATEDEVICE()
        cd.hAdapter = h_adapter
        s = _CreateDevice(ctypes.byref(cd))
        if s < 0 or (s & 0x80000000):
            print(f"[{i}] LUID={luid.HighPart}:{luid.LowPart} CreateDevice FAIL 0x{s & 0xFFFFFFFF:08X}")
            ca = D3DKMT_CLOSEADAPTER()
            ca.hAdapter = h_adapter
            _CloseAdapter(ctypes.byref(ca))
            continue

        h_device = cd.hDevice

        # Escape GET_INFO
        cmd = EscapeGetInfoData()
        cmd.Header.Command = 0x0001
        cmd.Header.Size = ctypes.sizeof(cmd)

        esc = D3DKMT_ESCAPE()
        esc.hAdapter = h_adapter
        esc.hDevice = h_device
        esc.Type = D3DKMT_ESCAPE_DRIVERPRIVATE
        esc.Flags = 0
        esc.pPrivateDriverData = ctypes.addressof(cmd)
        esc.PrivateDriverDataSize = ctypes.sizeof(cmd)

        s = _Escape(ctypes.byref(esc))
        ok = not (s < 0 or (s & 0x80000000))

        info = ""
        if ok:
            info = (f" vendor=0x{cmd.VendorId:04X} device=0x{cmd.DeviceId:04X}"
                    f" vram={cmd.VramSizeBytes//(1024*1024)}MB bars={cmd.NumBars}"
                    f" mmio_bar={cmd.MmioBarIndex} vram_bar={cmd.VramBarIndex}")
        print(f"[{i}] LUID={luid.HighPart}:{luid.LowPart} src={ai.NumOfSources} "
              f"hDev=0x{h_device:X} escape={'OK' if ok else f'0x{s & 0xFFFFFFFF:08X}'}{info}")

        if ok and cmd.VendorId == 0x1002:
            print(f"    >>> AMD GPU (0x{cmd.DeviceId:04X}) FOUND!")
            for bi in range(min(cmd.NumBars, 6)):
                b = cmd.Bars[bi]
                if b.Length > 0:
                    print(f"    BAR{bi}: phys=0x{b.PhysicalAddress:012X} size={b.Length//(1024*1024)}MB")

        # Cleanup
        dd = D3DKMT_DESTROYDEVICE()
        dd.hDevice = h_device
        _DestroyDevice(ctypes.byref(dd))
        ca = D3DKMT_CLOSEADAPTER()
        ca.hAdapter = h_adapter
        _CloseAdapter(ctypes.byref(ca))

    return 0


if __name__ == "__main__":
    sys.exit(main())
