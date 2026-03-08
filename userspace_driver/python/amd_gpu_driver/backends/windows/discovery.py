"""Adapter discovery — find our MCDM device among WDDM adapters.

Uses D3DKMTEnumAdapters2 to enumerate adapters, then ESCAPE_GET_INFO
to identify which one is our AMD GPU MCDM compute device.
"""

from __future__ import annotations

from dataclasses import dataclass

from amd_gpu_driver.backends.windows.driver_interface import (
    DeviceInfo,
    DriverInterface,
)
from amd_gpu_driver.errors import DeviceNotFoundError


# AMD PCI vendor ID
AMD_VENDOR_ID = 0x1002

# Known RDNA4 device IDs (expand as needed)
KNOWN_DEVICE_IDS = {
    0x7551: "AMD Radeon RX 9070 XT",
    0x7550: "AMD Radeon RX 9070",
}


@dataclass
class DiscoveredDevice:
    """An AMD GPU found via D3DKMT enumeration."""
    adapter_index: int
    info: DeviceInfo
    device_name: str


def discover_devices() -> list[DiscoveredDevice]:
    """Enumerate WDDM adapters and find AMD GPU MCDM devices.

    Probes each adapter with ESCAPE_GET_INFO. Adapters that respond
    with AMD vendor ID are returned.
    """
    iface = DriverInterface()
    adapters = iface.enumerate_adapters()
    devices: list[DiscoveredDevice] = []

    for i, adapter in enumerate(adapters):
        probe = DriverInterface()
        try:
            probe.open_adapter(adapter.AdapterLuid)
            probe.create_device()
            info = probe.get_info()
        except RuntimeError:
            # Not our driver, or escape failed — skip
            probe.close()
            continue

        if info.vendor_id == AMD_VENDOR_ID:
            name = KNOWN_DEVICE_IDS.get(
                info.device_id,
                f"AMD GPU (0x{info.device_id:04X})"
            )
            devices.append(DiscoveredDevice(
                adapter_index=i,
                info=info,
                device_name=name,
            ))

        probe.close()

    return devices


def open_device(device_index: int = 0) -> tuple[DriverInterface, DiscoveredDevice]:
    """Open a specific AMD GPU MCDM device by index.

    Returns the DriverInterface (already opened) and device info.
    Raises DeviceNotFoundError if the index is out of range.
    """
    devices = discover_devices()

    if device_index >= len(devices):
        raise DeviceNotFoundError(device_index)

    dev = devices[device_index]

    # Re-open the adapter for long-term use
    iface = DriverInterface()
    adapters = iface.enumerate_adapters()
    iface.open_adapter(adapters[dev.adapter_index].AdapterLuid)
    iface.create_device()

    return iface, dev
