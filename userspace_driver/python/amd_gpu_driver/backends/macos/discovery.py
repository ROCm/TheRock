"""macOS device discovery via IOKit registry.

Enumerates AMD GPUs attached via Thunderbolt by querying the ROCmGPU
DEXT for connected devices. Falls back to IOKit registry scanning if
the DEXT supports multiple devices.
"""

from __future__ import annotations

from dataclasses import dataclass

from amd_gpu_driver.backends.macos.iokit_client import (
    DeviceInfo,
    IOKitClient,
)


# Known AMD device IDs and their properties
_DEVICE_TABLE: dict[int, tuple[str, int]] = {
    # RDNA4 (gfx1201)
    0x7551: ("AMD Radeon RX 9070 XT", 120001),
    0x7550: ("AMD Radeon RX 9070", 120001),
    # RDNA3 (gfx1100 / gfx1101 / gfx1102)
    0x744C: ("AMD Radeon RX 7900 XTX", 110000),
    0x7448: ("AMD Radeon RX 7900 XT", 110000),
    0x7480: ("AMD Radeon RX 7800 XT", 110001),
    0x7483: ("AMD Radeon RX 7700 XT", 110001),
    0x7460: ("AMD Radeon RX 7600", 110002),
}


@dataclass
class DiscoveredDevice:
    """A discovered AMD GPU device."""
    index: int
    info: DeviceInfo
    device_name: str
    gfx_version: int  # e.g., 120001 for gfx1201


def discover_devices() -> list[DiscoveredDevice]:
    """Discover all AMD GPUs accessible via the ROCmGPU DEXT.

    Currently supports a single device (the DEXT matches one PCI device).
    Multi-GPU support would require either:
      - Multiple DEXT instances (one per GPU)
      - A single DEXT that enumerates multiple functions
    """
    devices = []

    try:
        client = IOKitClient()
        client.open()
        info = client.get_info()
        client.close()
    except RuntimeError:
        # DEXT not installed or no AMD GPU connected
        return []

    if info.vendor_id != 0x1002:
        return []

    name, gfx = _DEVICE_TABLE.get(
        info.device_id,
        (f"AMD GPU 0x{info.device_id:04X}", 0),
    )

    devices.append(DiscoveredDevice(
        index=0,
        info=info,
        device_name=name,
        gfx_version=gfx,
    ))

    return devices


def open_device(device_index: int = 0) -> tuple[IOKitClient, DiscoveredDevice]:
    """Open a specific AMD GPU device.

    Returns (IOKitClient, DiscoveredDevice) tuple.
    The client is connected and ready for use.
    """
    devices = discover_devices()
    if not devices:
        raise RuntimeError(
            "No AMD GPU found. Ensure:\n"
            "  1. ROCmGPU.dext is installed (run ROCmGPUApp install)\n"
            "  2. An AMD GPU is connected via Thunderbolt\n"
            "  3. The DEXT is enabled in System Settings > Driver Extensions"
        )

    if device_index >= len(devices):
        raise IndexError(
            f"Device index {device_index} out of range "
            f"(found {len(devices)} device(s))"
        )

    device = devices[device_index]
    client = IOKitClient()
    client.open()
    return client, device
