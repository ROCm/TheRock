# RFC0008: Multi-Architecture Packaging with Kpack

- **Author:** Stella Laurenzo
- **Created:** 2025-11-20
- **Modified:** 2025-11-20
- **Status:** Draft
- **Discussion:** https://github.com/ROCm/TheRock/discussions/2245

## Overview

ROCm currently ships "fat binaries" that embed device kernels for all supported GPU architectures within each binary. This approach has several limitations:

1. **Build scalability**: Each distribution family requires a complete build pipeline, limiting parallelism
1. **Binary bloat**: Users must download and store kernels for architectures they don't use
1. **Architecture support pressure**: Lesser-used architectures face deprecation due to build time and size costs
1. **Compression inefficiency**: No cross-kernel deduplication or advanced compression

This RFC proposes integrating `rocm-kpack` into TheRock to enable architecture-separated packaging. Kpack provides a structured format for separating device code from host code at packaging time, enabling parallel builds, selective deployment, and future compression improvements.

## Goals

1. **Enable build sharding by architecture groups** for high degree of parallelism

   - Shift from monolithic pipeline-per-family to multi-stage sharded builds
   - Build architecture-agnostic code once, architecture-specific code in parallel

1. **Group lesser-used Radeon architectures together** without build/binary size pressure

   - Support older architectures (e.g., Radeon VII) without impacting current builds
   - Reduce pressure to deprecate older architectures

1. **Let users pay for only what they need**

   - Radeon VII users don't download MI300X assets
   - Cloud deployments include only targeted architectures
   - Development environments can select specific architectures

1. **Add abstraction point for improved compression**

   - Enable future horizontal cross-kernel-arch compression
   - Support dictionary-trained compression across kernels
   - Provide foundation for kernel deduplication

## Non-Goals

- **Not replacing existing database formats immediately**: rocBLAS, hipBLASLt, and AOTriton will continue using their current formats initially
- **Not modifying host API/ABI**: Applications continue working without changes
- **Not requiring immediate adoption**: Libraries can migrate incrementally to using kpack to manage their pre-compiled kernel databases
- **Not optimizing for single-architecture deployments**: Focus is on multi-architecture scenarios

## Dependencies

- **rocm-kpack**: Core library and tools (integrated as submodule)
- **msgpack-cxx**: For TOC serialization (already in TheRock)
- **zstd**: For kernel compression (already in TheRock)
- **Python 3.9+**: For packaging tools

## Build System Transformation

### Current State: Monolithic Pipeline

```
gfx90X-build/  → Full ROCm build for gfx900,gfx906,gfx908,gfx90a
gfx110X-build/ → Full ROCm build for gfx1100,gfx1101,gfx1102,gfx1103
gfx942-build/ → MI300X, MI308, etc builds
```

Each pipeline builds all components (host + device) for its architecture family.

### Future State: Multi-Stage Sharded Build

```
Stage 1: Generic Build (Once)
  → Compiler
  → Host-only components
  → Architecture-independent libraries
  → Headers, documentation, tools

Stage 2: Architecture Builds (Parallel, with integrated kpack split)
  gfx900-build/  → Device code for gfx900 + kpack artifact split
  gfx906-build/  → Device code for gfx906 + kpack artifact split
  gfx908-build/  → Device code for gfx908 + kpack artifact split
  gfx90a-build/  → Device code for gfx90a + kpack artifact split
  gfx1100-build/ → Device code for gfx1100 + kpack artifact split
  ...
  Each build extracts device kernels and creates architecture-specific kpack archives

Stage 3: Package Assembly (with kpack recombine)
  → Recombine split artifacts from all architecture builds
  → Create base packages (host code + headers)
  → Create architecture packages (device code + kpack archives)
  → Generate metadata for dependency resolution
```

### Build Sharding Strategy

Sharding occurs at the top-level directory:

- `base/`: Built once (generic)
- `compiler/`: AMD LLVM Compiler (generic)
- `rocm-systems/`: CLR runtime (generic) + device libraries for all supported arches
- `rocm-libraries/` (sharded per-arch group): Host API (generic) + kernels (per-arch)
- ...

## Technical Design

### Integration Structure

```
TheRock/
├── base/
│   ├── rocm-kpack/        # New submodule
│   ├── rocm-cmake/
│   └── ...
└── core/
    └── clr/
        └── hipamd/src/
            └── kpack_loader.cpp  # Runtime integration
```

### Kpack Archive Format

```
┌────────────────────┐
│ Header (16 bytes)  │  Magic: "KPAK", Version, TOC offset
├────────────────────┤
│ Padding            │  Align to 64-byte boundary
├────────────────────┤
│ Blob Data          │  Compressed kernels (Zstd)
│ ...                │  Per-kernel compression
├────────────────────┤
│ TOC (MessagePack)  │  Metadata + kernel index
└────────────────────┘
```

### CLR Runtime Integration

```cpp
// core/clr/hipamd/src/kpack_loader.cpp
class KpackLoader {
  std::unordered_map<std::string, kpack_archive_t> archives_;

public:
  Status LoadKernel(const std::string& binary,
                    const std::string& arch,
                    void** kernel_data,
                    size_t* kernel_size) {
    // 1. Check if binary has .rocm_kpack_ref marker
    if (!HasKpackMarker(binary)) {
      return LoadLegacyFatBinary(binary, arch, kernel_data, kernel_size);
    }

    // 2. Determine kpack archive location
    std::string kpack_path = GetKpackPath(binary);

    // 3. Open archive (cached)
    kpack_archive_t archive = GetOrOpenArchive(kpack_path);

    // 4. Load kernel
    return kpack_get_kernel(archive, binary, arch,
                           kernel_data, kernel_size);
  }
};
```

### Artifact Split and Recombine Process

The kpack build integration uses two primary tools for managing architecture-separated artifacts. This approach has been proven viable for ELF files. COFF files (Windows) have been paper-prototyped and will be implemented fully as part of the project.

#### Split Phase (Per-Architecture Build)

The `split_artifacts` tool processes fat binaries from each architecture build:

1. **Scans artifacts** for shared libraries with embedded GPU code
1. **Unbundles GPU kernels** using `clang-offload-bundler`
1. **Creates kpack archives** per architecture with compressed kernels
1. **Strips GPU code** from host binaries (PROGBITS → NOBITS)
1. **Adds `.rocm_kpack_ref` marker** pointing to kpack files
1. **Separates databases** (rocBLAS, hipBLASLt) by architecture

Output structure from split:

```
artifact_generic/     # Host code only, no GPU kernels
artifact_gfx1100/     # gfx1100 kernels + databases only
artifact_gfx1101/     # gfx1101 kernels + databases only
...
```

#### Recombine Phase (Package Assembly)

The `recombine_artifacts` tool merges split artifacts according to packaging configuration:

1. **Creates generic packages** from primary shard's host code
1. **Groups architectures** (e.g., gfx1100,1101,1102 → gfx110X)
1. **Merges kpack archives** and databases for each group
1. **Generates `.kpm` manifests** listing available kernels

Example workflow:

```bash
# Split artifacts in each architecture build
python -m rocm_kpack.tools.split_artifacts \
    --batch-artifact-parent-dir /artifacts/gfx110X-build \
    --output-dir /split/gfx110X-build \
    --split-databases rocblas hipblaslt \
    --clang-offload-bundler /path/to/bundler

# Recombine for packaging
python -m rocm_kpack.tools.recombine_artifacts \
    --input-shards-dir /split \
    --config packaging-config.json \
    --output-dir /recombined
```

See https://github.com/stellaraccident/rocm-kpack/blob/main/docs/tutorial_split_artifacts.md for detailed workflow.

### Packaging Structure

#### Native Packages (DEB/RPM)

*Note: This structure is a strawman and will be elaborated once integration is underway.*

Based on the current packaging structure in `build_tools/packaging/linux/package.json`, packages would be split into generic and architecture-specific variants:

```
# Generic packages (host code, headers, documentation)
rocm-core8.0_8.0.0_amd64.deb           # Core utilities and version info
rocm-cmake8.0_8.0.0_amd64.deb          # CMake modules
hip-runtime8.0_8.0.0_amd64.deb         # HIP runtime (host)
rocblas8.0_8.0.0_amd64.deb             # rocBLAS host API

# Architecture-specific packages (device code)
rocm-device-libs-gfx900_8.0.0_amd64.deb   # Device libraries for gfx900
rocm-device-libs-gfx1100_8.0.0_amd64.deb  # Device libraries for gfx1100
rocblas-kernels-gfx900_8.0.0_amd64.deb    # rocBLAS kernels for gfx900
rocblas-kernels-gfx1100_8.0.0_amd64.deb   # rocBLAS kernels for gfx1100

# Meta-packages for architecture families
rocm-gfx90X_8.0.0_amd64.deb  # Depends on all gfx90X packages
rocm-gfx110X_8.0.0_amd64.deb # Depends on all gfx110X packages
```

#### Python Packaging

##### ROCm Python Packages

The ROCm SDK will be distributed as Python packages from a single multi-architecture index, with host and device packages resulting from the recombination process:

```
# Single index serving all architectures
https://rocm.nightlies.amd.com/v2/multi-arch/

# Package structure after recombination
rocm                           # Meta-package with dynamic dependency resolution
rocm-sdk-core                  # Compiler and utility tools (generic)
rocm-sdk-libraries             # Math libraries host API (generic)
rocm-sdk-libraries-gfx900     # Device code for gfx900
rocm-sdk-libraries-gfx1100    # Device code for gfx1100
rocm-sdk-libraries-gfx1101    # Device code for gfx1101
rocm-sdk-libraries-gfx942     # Device code for gfx942
...
rocm-sdk-devel                # Development tools (generic)

# Installation automatically resolves architecture dependencies
pip install --index-url https://rocm.nightlies.amd.com/v2/multi-arch/ "rocm[libraries,devel]"
```

##### PyTorch Packages

PyTorch wheels will be optimized for WheelNext using split packages:

```
# Base wheel (host code)
torch_rocm-2.12.0-cp311-linux_x86_64.whl

# Architecture-specific wheels (device code)
torch_rocm_gfx110X-2.12.0-cp311-linux_x86_64.whl
torch_rocm_gfx94X-2.12.0-cp311-linux_x86_64.whl

# WheelNext will dynamically resolve dependencies based on detected GPU or explicit hints
```

## Implementation Plan

*Target: Minimally functional in a sandbox before end of year.*

### Phase 1: Infrastructure

- Move `rocm-kpack` to the ROCm organization
- Add `rocm-kpack` as submodule at `base/rocm-kpack`
- Integrate into TheRock CMake build system
- Add CI jobs for testing (pure CPU-only unit tests)

### Phase 2: Build System

- Implement multi-stage build configuration
- Add per-architecture build jobs with integrated split
- Create package assembly with recombine

### Phase 3: CLR Integration

- Implement `KpackLoader` in CLR runtime
- Add fallback to legacy fat binaries
- Update HIP runtime to check for kpack markers

### Phase 4: Library Migration

- Some libraries will need rework to support split architecture pre-compiled databases
- Initial focus on libraries with clear separation needs
- Gradual migration as libraries adapt their database formats
- rocBLAS, hipBLASLt: Continue with current format initially

### Phase 5: Packaging

- Extend DEB/RPM generation for split packages
- Implement Python wheel splitter
- Add package dependency management
- WheelNext integration for PyTorch

## Performance Considerations

The initial release should have no measurable impact on startup time or memory footprint compared to current fat binaries. The kpack runtime is designed to be transparent to applications.

Future versions will explore more advanced compression techniques which may trade time/memory for compression ratio on less frequently used kernels. Options include:

- Dictionary-trained compression across kernel families
- Cross-kernel deduplication at the block level
- Lazy decompression with kernel-specific caching strategies
- Progressive loading based on usage patterns

## Migration Path

### Library Categories

1. **Libraries with database-based kernels**

   - Libraries currently using SQLite or similar databases
   - Clear benefit from unified kpack structure
   - Can migrate database contents to kpack TOC or refactor into arch-separated directory trees

1. **Libraries with existing separation** (rocBLAS, hipBLASLt, AOTriton)

   - Already have architecture-separated layouts
   - Complex existing infrastructure
   - Continue with current formats initially
   - Migrate after kpack matures

1. **New libraries**

   - Start with kpack from inception
   - No legacy format support needed
   - Benefit from established tooling

### Validation Strategy

- Round-trip testing: fat binary → kpack → runtime
- Performance regression testing
- Multi-architecture deployment testing
- Package installation verification

## Follow-on Work

### Near Term

1. **Cross-kernel deduplication**: Identify common code sequences
1. **Dictionary-trained compression**: Train on kernel corpus
1. **Python wheel splitter**: Automated wheel processing

### Medium Term

1. **Compiler integration**: `--emit-kpack` flag
1. **Streaming write mode**: For very large libraries
1. **Kernel preloading**: Predictive loading based on usage

### Long Term

1. **Cross-architecture sharing**: Common IR or bytecode

## Summary

Integrating rocm-kpack into TheRock enables a fundamental shift in how ROCm handles multi-architecture support. By separating device code from host code and enabling parallel architecture builds, we can:

- Achieve high build parallelism through architecture sharding
- Support lesser-used architectures without impacting mainstream users
- Let users download only the architectures they need
- Create foundation for future compression improvements

The phased implementation approach ensures backward compatibility while providing immediate benefits to early adopters. The architecture provides a clean abstraction point for future optimizations while maintaining compatibility with existing applications.

This transformation positions ROCm for sustainable growth across an expanding range of GPU architectures while improving both developer and user experience.
