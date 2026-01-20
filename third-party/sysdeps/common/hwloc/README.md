# hwloc System Dependency

This directory contains the build configuration for `libhwloc` as a bundled system dependency for TheRock.

## Overview

hwloc (Hardware Locality) provides portable abstraction of the hierarchical topology of modern architectures, including NUMA memory nodes, sockets, shared caches, cores, and simultaneous multithreading.

## Version

- **hwloc 1.11.13** (last stable 1.x release, compatible with rocrtst)
- Source: https://download.open-mpi.org/release/hwloc/v1.11/hwloc-1.11.13.tar.gz
- Note: Using hwloc 1.x for compatibility with rocrtst which uses hwloc 1.11.6 API

## Dependencies

- **libnuma** (from therock-numactl) - CRITICAL for NUMA integration
- **libpciaccess** (from therock-libpciaccess) - CRITICAL for PCI/GPU device discovery

## Build Configuration

The build uses a minimal feature set optimized for rocrtst:

### Enabled Features:

- `--enable-libnuma` - NUMA integration (critical)
- `--enable-pci` - PCI device discovery (critical for GPU discovery)
- `--enable-cpuid` - CPU topology detection

### Disabled Features:

- `--disable-static` - Only build shared libraries
- `--disable-opencl` - OpenCL support not needed
- `--disable-cuda` - CUDA support not needed
- `--disable-nvml` - NVML support not needed
- `--disable-gl` - OpenGL support not needed
- `--disable-cairo` - Cairo graphics not needed
- `--disable-xml` - XML backend not needed
- `--disable-plugin-dlopen` - Plugin system not needed
- `--disable-plugin-ltdl` - Plugin system not needed

### Minimal Configuration Rationale

This configuration provides the minimal feature set required by rocrtst while avoiding unnecessary dependencies:

**What rocrtst needs:**

- Hardware topology discovery (CPU cores, caches, NUMA nodes)
- PCI device enumeration (GPU discovery via `--enable-pci`)
- NUMA memory node information (via `--enable-libnuma`)

**What we disable:**

- **GPU vendor APIs** (`--disable-opencl`, `--disable-cuda`, `--disable-nvml`): rocrtst uses HSA/ROCm APIs directly, not vendor-specific hwloc backends
- **Graphics/Rendering** (`--disable-gl`, `--disable-cairo`): hwloc's lstopo visualization tool not needed
- **XML backend** (`--disable-xml`): Configuration export/import not needed
- **Plugin system** (`--disable-plugin-dlopen`, `--disable-plugin-ltdl`): No dynamic plugin loading required

**Critical dependencies:**

- **libpciaccess** (bundled): Required for `--enable-pci` to discover PCI devices including GPUs
- **libnuma** (bundled): Required for `--enable-libnuma` to access NUMA topology

This configuration was validated by testing rocrtst's topology discovery functionality with the bundled libraries.

## Installation

hwloc is installed to `lib/rocm_sysdeps` alongside other bundled system dependencies.

The library is built with:

- Symbol versioning: `AMDROCM_SYSDEPS_1.0`
- SONAME prefixing: `librocm_sysdeps_hwloc.so.5`
- Relocatable pkg-config files
- Origin-relative RPATH

## Usage

To enable hwloc in your build:

```bash
cmake -DTHEROCK_ENABLE_SYSDEPS_HWLOC=ON ...
```

Components can link against hwloc using the CMake target:

```cmake
find_package(hwloc REQUIRED)
target_link_libraries(my_target PRIVATE hwloc::hwloc)
```

Or via pkg-config:

```bash
PKG_CONFIG_PATH=/path/to/rocm_sysdeps/lib/pkgconfig pkg-config --cflags --libs hwloc
```

## rocrtst Integration

rocrtst requires hwloc for hardware topology discovery. With this bundled version, rocrtst can be built without requiring system-installed hwloc packages.

The rocrtst CMakeLists.txt includes logic to find hwloc via CMake's `find_package(hwloc)` (which finds our `hwloc-config.cmake`) or pkg-config fallback. With the bundled version available, it will be automatically discovered.
