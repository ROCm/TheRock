# libpciaccess System Dependency

This directory contains the build configuration for `libpciaccess` as a bundled system dependency for TheRock.

## Overview

libpciaccess provides portable abstraction for PCI device access across different operating systems and architectures. It enables enumeration of PCI devices, reading configuration space, and device identification.

## Version

- **libpciaccess 0.18.1** (latest stable release as of January 2026)
- Source: https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/libpciaccess-0.18.1.tar.xz
  - Originally from: https://www.x.org/releases/individual/lib/libpciaccess-0.18.1.tar.xz
- Build System: Meson/Ninja (following TheRock's meson infrastructure pattern)

## Dependencies

- **System:** glibc (libc.so.6) only
- **Build:** patchelf, meson, ninja

This library has no external dependencies beyond the base system, making it highly portable.

## Build Configuration

The build uses Meson with a minimal feature set optimized for hwloc (the primary consumer):

### Meson Options:

- `--prefix "/"` - Relocatable installation (uses DESTDIR)
- `-Dpkgconfig.relocatable=true` - Relocatable pkg-config files
- `-Dlibdir=lib` - Install libraries to lib/ subdirectory
- `-Dzlib=disabled` - Disable gzip-compressed pci.ids support

### Enabled Features (default):

- PCI device enumeration (core, always enabled)
- Config space reading (core, always enabled)
- Vendor/device name lookup (core, always enabled)

### Minimal Configuration Rationale

This configuration provides the minimal feature set required by hwloc while avoiding unnecessary dependencies:

**What hwloc needs:**

- PCI device enumeration (`pci_system_init`, `pci_device_next`, etc.)
- Config space reading (`pci_device_cfg_read`)
- Vendor/device name lookup (`pci_device_get_vendor_name`, `pci_device_get_device_name`)

**What we disable:**

- `-Dzlib=disabled`: Disables gzip-compressed pci.ids database support. Name lookup still works with uncompressed system pci.ids files (e.g., `/usr/share/misc/pci.ids`), but we avoid a zlib dependency.

**What cannot be disabled:**

- PCI enumeration (core functionality, no configure option)
- Config space reading (required for device discovery, no configure option)
- ROM reading (built-in, no configure option)

This configuration was validated by examining hwloc's actual symbol usage, ensuring we only include features that are actively used.

## Installation

libpciaccess is installed to `lib/rocm_sysdeps` alongside other bundled system dependencies.

The library is built with:

- Symbol versioning: `AMDROCM_SYSDEPS_1.0`
- SONAME prefixing: `librocm_sysdeps_pciaccess.so.0`
- Relocatable pkg-config files
- Origin-relative RPATH

## Usage

Enable bundled system dependencies during CMake configuration:

```bash
cmake -DTHEROCK_BUNDLE_SYSDEPS=ON ...
```

This automatically enables `libpciaccess` as a transitive dependency when `hwloc` is enabled.

## hwloc Integration

libpciaccess is primarily used as a **transitive dependency** for hwloc to enable PCI/GPU device discovery. It is not intended for direct consumption by TheRock components.

With this bundled version, hwloc can be built with `--enable-pci` support without requiring system-installed libpciaccess packages.
