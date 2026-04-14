# ROCmGPU — macOS eGPU Driver for AMD ROCm

DriverKit extension + Python userspace driver for running ROCm compute
workloads on external AMD GPUs connected via Thunderbolt to Apple Silicon Macs.

## Architecture

```
Python userspace driver (amd_gpu_driver)
    ↕ IOKit (IOConnectCallScalarMethod / IOConnectMapMemory64)
ROCmGPU.dext (DriverKit extension)
    ↕ PCIDriverKit framework
Thunderbolt 3/4 / USB4
    ↕
AMD RDNA4 eGPU
```

**ROCmGPU.dext** provides a minimal PCIe HAL:
- PCI BAR mapping into userspace
- MMIO register read/write
- DMA buffer allocation (with IOMMU translation)
- PCI config space access
- Function-Level Reset
- MSI-X interrupt forwarding

**Python backend** handles all GPU-specific logic:
- IP block discovery
- Memory controller initialization (GMC)
- Firmware loading (PSP)
- Compute/SDMA queue creation
- PM4 command packet building
- GPU page table management

## Requirements

### Hardware
- Apple Silicon Mac (M1/M2/M3/M4) with Thunderbolt 3/4
- Thunderbolt eGPU enclosure (Razer Core X, Sonnet Breakaway, etc.)
- AMD RDNA3 or RDNA4 GPU (RX 7900 XT, RX 9070 XT, etc.)

### Software
- macOS 12.1 (Monterey) or later
- Xcode 15+ with DriverKit SDK
- Python 3.10+

### Development (without Apple entitlements)
- SIP disabled (required for self-signed DEXT loading)
- System Extensions developer mode enabled

## Quick Start (Development)

```bash
# 1. Disable SIP (from Recovery Mode terminal)
csrutil disable

# 2. Enable developer mode (after normal reboot)
sudo systemextensionsctl developer on

# 3. Build and install
cd macos_driver
./scripts/build.sh
./scripts/install.sh

# 4. Approve in System Settings if prompted
#    System Settings > General > Login Items & Extensions > Driver Extensions

# 5. Test
python3 -c "
from amd_gpu_driver.backends.macos import MacOSDevice
dev = MacOSDevice()
dev.open()
print(dev)
print(f'VRAM: {dev.vram_size // (1024**3)} GB')
dev.close()
"
```

## Project Structure

```
macos_driver/
├── ROCmGPU/
│   ├── ROCmGPUDriver/           # DriverKit extension (C++)
│   │   ├── ROCmGPUShared.h      # Shared protocol definitions
│   │   ├── ROCmGPUDriver.h/cpp  # IOService: PCI device lifecycle
│   │   ├── ROCmGPUUserClient.h/cpp  # IOUserClient: escape dispatch
│   │   ├── Info.plist            # IOKit matching personality
│   │   └── *.entitlements        # DriverKit entitlements
│   └── ROCmGPUApp/              # Host app (DEXT installer)
│       ├── main.swift
│       └── *.entitlements
├── scripts/
│   ├── build.sh                  # Build script
│   └── install.sh                # Install/manage DEXT
└── README.md

python/amd_gpu_driver/backends/macos/
├── __init__.py                   # Exports MacOSDevice
├── iokit_client.py               # IOKit ctypes bindings
├── device.py                     # MacOSDevice(DeviceBackend)
├── discovery.py                  # Device enumeration
├── memory.py                     # DMA + VRAM allocation
├── queue.py                      # Compute/SDMA queues
├── events.py                     # Signal/event synchronization
└── bringup.py                    # GPU cold-boot initialization
```

## Escape Command Protocol

The DEXT exposes 13 commands via IOUserClient:

| Selector | Name | Inputs | Outputs |
|----------|------|--------|---------|
| 0 | GET_INFO | — | DeviceInfo struct |
| 1 | RESET | — | — |
| 2 | CFG_READ | offset, width | value |
| 3 | CFG_WRITE | offset, width, value | — |
| 4 | MMIO_READ32 | barIndex, offset | value |
| 5 | MMIO_WRITE32 | barIndex, offset, value | — |
| 6 | MAP_BAR | barIndex | size |
| 7 | UNMAP_BAR | barIndex | — |
| 8 | ALLOC_DMA | size, flags | DMAInfo struct |
| 9 | FREE_DMA | bufferID | — |
| 10 | MAP_DMA | bufferID | size |
| 11 | ENABLE_MSI | vectorIndex | — |
| 12 | WAIT_INTERRUPT | timeoutMS | status |

Memory mapping uses `IOConnectMapMemory64` with type:
- 0-5: PCI BARs
- 0x100+N: DMA buffer N

## Status

### Implemented
- [x] DEXT skeleton (IOService + IOUserClient)
- [x] Escape command dispatch (all 13 commands)
- [x] PCI BAR mapping
- [x] MMIO register access
- [x] DMA buffer allocation with IOMMU translation
- [x] PCI config space access
- [x] Function-Level Reset
- [x] MSI-X interrupt handling
- [x] Python IOKit client bindings
- [x] Device discovery
- [x] Memory manager (GTT + VRAM)
- [x] Queue manager (compute + SDMA)
- [x] Event/signal manager (polling)
- [x] Bringup orchestrator framework

### TODO (requires hardware testing)
- [ ] IP discovery with real hardware
- [ ] NBIO initialization
- [ ] GMC initialization (page tables)
- [ ] PSP firmware loading
- [ ] IH ring setup
- [ ] Compute ring bring-up
- [ ] End-to-end compute dispatch
- [ ] SDMA copy operations
- [ ] Interrupt-driven signal waits

## Apple Entitlement Process

For distribution (beyond development):
1. Request DriverKit entitlements at https://developer.apple.com/system-extensions/
2. Specifically request `com.apple.developer.driverkit.transport.pci`
3. Provide the PCI match string: `0x00001002&0x0000FFFF` (AMD vendor)
4. Apple reviews and approves (TinyGPU established precedent in March 2026)
5. Sign with distribution profile and notarize

## Related Work

- [TinyGPU](https://docs.tinygrad.org/tinygpu/) — First Apple-approved eGPU DEXT (tinygrad)
- [TheRock userspace_driver](../python/) — Linux/Windows AMD GPU userspace driver
- [ROCm](https://rocm.docs.amd.com/) — AMD's open-source GPU compute platform
