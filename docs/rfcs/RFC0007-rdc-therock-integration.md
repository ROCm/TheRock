---
author: Stella Laurenzo
created: 2025-01-11
modified: 2025-01-11
status: Draft
discussion: [TBD - Add GitHub issue/PR link]
---

# RFC0007: RDC TheRock Integration with Static gRPC

## Overview

RDC (ROCm Data Center Tool) is a datacenter GPU monitoring and administration tool that provides telemetry, diagnostics, and management capabilities for AMD GPUs. Currently distributed only through system packages (DEB/RPM), RDC needs to be integrated into TheRock to create portable, distribution-neutral builds suitable for containerized deployments.

RDC operates in two distinct modes:

1. **Embedded Mode**: Direct library access for in-process GPU monitoring
1. **Standalone Mode**: Client-server architecture with rdcd daemon and rdci CLI

This RFC defines how to integrate RDC into TheRock's build system with gRPC statically linked for the standalone components, ensuring portability across Linux distributions without runtime dependency conflicts.

### Software Artifacts Breakdown

RDC produces the following build artifacts:

#### Core Libraries (Always Built)

| Artifact                | Size   | Mode | Direct Dependencies                     | Purpose                               |
| ----------------------- | ------ | ---- | --------------------------------------- | ------------------------------------- |
| **librdc_bootstrap.so** | ~200KB | All  | pthread, dl                             | Plugin loader, core utilities         |
| **librdc.so**           | ~2MB   | All  | rdc_bootstrap, pthread, amd_smi, libcap | Main RDC functionality, embedded mode |

#### Standalone Mode Only

| Artifact             | Size   | Mode       | Direct Dependencies                                               | Purpose             |
| -------------------- | ------ | ---------- | ----------------------------------------------------------------- | ------------------- |
| **librdc_client.so** | ~500KB | Standalone | rdc_bootstrap, pthread, rt, **gRPC::grpc++**, dl                  | gRPC client library |
| **rdcd**             | ~2MB   | Standalone | pthread, rt, **gRPC::grpc++**, libcap, dl, amd_smi, rdc_bootstrap | Daemon server       |
| **rdci**             | ~1MB   | Standalone | pthread, dl, **gRPC::grpc++**, rdc_bootstrap                      | CLI client          |

#### Optional Plugin Modules

| Artifact           | Size   | Build Flag        | Direct Dependencies                                  | Purpose              |
| ------------------ | ------ | ----------------- | ---------------------------------------------------- | -------------------- |
| **librdc_rocr.so** | ~1MB   | BUILD_RUNTIME=ON  | rdc, rdc_bootstrap, hsa-runtime64, pthread, dl       | ROCr diagnostics     |
| **librdc_rocp.so** | ~500KB | BUILD_PROFILER=ON | hsa-runtime64, rocprofiler-sdk, pthread, dl, amd_smi | Profiler integration |
| **librdc_rvs.so**  | ~200KB | BUILD_RVS=ON      | rdc, rdc_bootstrap, rvs, pthread, dl                 | Validation suite     |

The critical observation is that **embedded mode requires only ~2.2MB** of libraries (no gRPC), while **standalone mode adds ~45MB** of gRPC dependencies.

### Portable Daemon Distribution Strategy

RDC will use static linking for gRPC dependencies to ensure maximum portability:

#### Static-Linked Architecture

```
portable-rdc/
├── bin/
│   ├── rdcd                    # ~50MB (gRPC statically linked)
│   └── rdci                    # ~50MB (gRPC statically linked)
├── lib/
│   ├── librdc_bootstrap.so     # 200KB
│   ├── librdc.so               # 2MB (embedded mode)
│   ├── librdc_client.so        # ~50MB (gRPC statically linked)
│   └── rdc/
│       ├── librdc_rocr.so      # 1MB (optional module)
│       ├── librdc_rocp.so      # 500KB (optional module)
│       └── librdc_rvs.so       # 200KB (optional module)
└── share/
    └── rdc/
        └── conf/
            └── rdc_options.conf  # User-level configuration
```

**Static Linking Rationale:**

- Avoids SONAME conflicts with system gRPC libraries
- Eliminates need for symbol versioning management
- More portable across glibc versions
- Simplifies distribution (no need to bundle 15+ shared libraries)
- Trade-off: Larger binaries but simpler deployment

**Future Optimization (Post-Initial Implementation):**
Create a "busy-box" style `librdc_grpc.so` containing:

- `rdcd_main()` and `rdci_main()` entry points
- All gRPC dependencies statically linked with hidden visibility
- Light-weight executable shims that call into the shared library
- This would reduce total size from ~150MB to ~55MB for all standalone components

### Goals

1. Integrate RDC into TheRock build system under `dctools/` directory
1. Add gRPC to TheRock's third-party dependencies for static linking
1. Create portable, distribution-neutral builds of all RDC components
1. Build both embedded and standalone modes from the outset
1. Maintain compatibility with existing system package installations

### Non-Goals

1. Replacing the system package distribution
1. Modifying the core RDC architecture or APIs
1. Supporting Windows or macOS (Linux x86_64 only initially)
1. Python wheel packaging (deferred to future work)

## Dependencies

### Embedded Mode Dependencies

- **amd-smi-lib** (>=26.0.0): Required for GPU telemetry/monitoring
- **libcap**: Linux capabilities for privileged operations
- **pthread, rt, dl**: Standard system libraries

### Standalone Mode Dependencies

The standalone mode (rdcd daemon and rdci CLI) requires gRPC v1.67.1 and its transitive dependencies:

#### gRPC Stack (~40-50MB total)

- **libgrpc++.so.1.67** (~5MB): C++ gRPC library
- **libgrpc.so.41** (~10MB): Core gRPC C library
- **libprotobuf.so.3.25** (~3MB): Protocol buffer runtime
- **libabsl\_\*.so** (~20MB): 15+ Abseil libraries (strings, time, synchronization, etc.)
- **libupb*.so*\* (~2MB): Micro-protobuf implementation
- **libre2.so** (~500KB): Regular expression engine
- **System libraries**: OpenSSL (libssl, libcrypto), zlib

#### Version Requirement Rationale

RDC specifically requires gRPC v1.67.1 due to:

- Clang 18+ ABI compatibility fixes in Abseil
- Protobuf 27.x+ requirement
- Symbol versioning improvements for manylinux environments

### Optional Module Dependencies

- **hsa-runtime64**: Required for librdc_rocr.so (ROCr diagnostics)
- **rocprofiler-sdk** (>=1.1.0): Required for librdc_rocp.so (profiler integration)
- **rvs**: Required for librdc_rvs.so (validation suite)

## Technical Design

### gRPC Integration Strategy: Static Linking

RDC is unique in the ROCm ecosystem as the only component using gRPC. To support standalone mode while maintaining portability, gRPC will be integrated as a third-party dependency and statically linked into RDC's standalone components.

### Static Linking Architecture

#### gRPC Third-Party Integration

Add gRPC to TheRock's third-party dependencies following TheRock's idiom:

```cmake
# In third-party/grpc/CMakeLists.txt

# Option 1: Use BoringSSL (bundled with gRPC)
therock_subproject_fetch(therock-grpc-sources
  CMAKE_PROJECT
  GIT_REPOSITORY https://github.com/grpc/grpc
  GIT_TAG v1.67.1
  GIT_SHALLOW ON
  GIT_SUBMODULES_RECURSE ON
)

therock_cmake_subproject_declare(therock-grpc
  BACKGROUND_BUILD
  EXCLUDE_FROM_ALL
  NO_MERGE_COMPILE_COMMANDS
  OUTPUT_ON_FAILURE
  EXTERNAL_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/source"
  CMAKE_ARGS
    -DBUILD_SHARED_LIBS=OFF           # Critical: Build as static
    -DgRPC_INSTALL=ON
    -DgRPC_BUILD_TESTS=OFF
    -DgRPC_PROTOBUF_PROVIDER=module   # Build protobuf from submodule
    -DgRPC_ZLIB_PROVIDER=package      # Use TheRock's zlib
    -DgRPC_CARES_PROVIDER=module      # Build c-ares from submodule
    -DgRPC_RE2_PROVIDER=module        # Build re2 from submodule
    -DgRPC_SSL_PROVIDER=module        # Build BoringSSL from submodule
    -DgRPC_ABSL_PROVIDER=module       # Build abseil from submodule
    -Dprotobuf_BUILD_SHARED_LIBS=OFF  # Ensure protobuf is also static
    # Symbol visibility to prevent pollution when statically linked
    -DCMAKE_CXX_VISIBILITY_PRESET=hidden
    -DCMAKE_C_VISIBILITY_PRESET=hidden
    -DCMAKE_VISIBILITY_INLINES_HIDDEN=ON
)
therock_cmake_subproject_provide_package(therock-grpc gRPC lib/cmake/grpc)
therock_cmake_subproject_activate(therock-grpc)

add_dependencies(therock-third-party therock-grpc)
```

#### SSL/TLS Dependency Handling

**Challenge:** gRPC requires an SSL library at build time, even though RDC supports insecure mode at runtime.

**Options for SSL handling:**

1. **Option A: Use BoringSSL (Recommended)**

   - Built statically from gRPC's git submodule
   - Google's fork of OpenSSL, designed for static linking
   - No licensing concerns (ISC license)
   - Symbols hidden via visibility controls
   - Self-contained, no system dependency
   - Trade-off: Increases binary size by ~2-3MB

1. **Option B: Add OpenSSL to TheRock sysdeps**

   - Build OpenSSL as shared library with custom SONAME
   - Apply symbol versioning patches for isolation
   - Distribute with ROCm as `librocm_ssl.so`
   - Complex but follows ROCm precedent (see libdrm)
   - Trade-off: Significant maintenance burden

1. **Option C: Require system OpenSSL (Not portable)**

   - Use `-DgRPC_SSL_PROVIDER=package`
   - Fails portability goal
   - Not recommended for TheRock

**Decision:** Use Option A (BoringSSL) for initial implementation. RDC commonly runs in insecure mode for development/testing (using the `-u` flag), making the SSL dependency overhead acceptable for the portability gained.

**Key Points:**

- `BUILD_SHARED_LIBS=OFF` ensures gRPC and all dependencies built as static
- Using "module" provider for most dependencies ensures consistency
- BoringSSL statically linked and symbols hidden
- TheRock's existing zlib used where possible

#### Symbol Visibility Management

To prevent symbol pollution, gRPC and its dependencies must be built with hidden visibility:

```cmake
# In third-party/grpc/CMakeLists.txt
therock_cmake_subproject_declare(therock-grpc
  ...
  CMAKE_ARGS
    -DCMAKE_CXX_VISIBILITY_PRESET=hidden
    -DCMAKE_C_VISIBILITY_PRESET=hidden
    -DCMAKE_VISIBILITY_INLINES_HIDDEN=ON
    # Additional gRPC-specific visibility controls if available
    # May need to patch gRPC if it doesn't respect CMAKE visibility settings
)
```

**Important Notes:**

- gRPC and all its dependencies (protobuf, abseil, BoringSSL) must be built with hidden visibility
- This ensures symbols are hidden when statically linked into any binary
- If gRPC doesn't have explicit visibility control knobs, it may need to be patched
- This follows the TheRock third-party pattern where libraries are built with hidden symbols
- RDC's own symbols should remain properly exported as needed

**Verification:**
After building, verify symbol visibility:

```bash
nm -C librdc_client.so | grep -c " T grpc::"  # Should be 0 or very few
```

### Build Structure in TheRock

#### Option 1: Monolithic Build (Recommended Initially)

Keep RDC as a single subproject with conditional features:

```cmake
# In dctools/CMakeLists.txt
therock_cmake_subproject_declare(rdc
  SOURCE_DIR ${THEROCK_SOURCE_DIR}/rocm-systems/projects/rdc
  BUILD_DEPS
    amd-smi
    therock-grpc  # Static gRPC dependency
  BUILD_OPTIONS
    -DBUILD_STANDALONE=ON    # Build all modes
    -DBUILD_RUNTIME=ON
    -DBUILD_PROFILER=OFF
    -DBUILD_RVS=OFF
    # Note: RDC currently uses -DGRPC_ROOT which is legacy
    # TODO: Update RDC to use find_package(gRPC) for modern CMake discovery
    -DGRPC_ROOT=${therock-grpc_INSTALL_DIR}
)
```

**RDC Modernization Note:**
RDC's CMakeLists.txt should be updated to use modern CMake package discovery:

```cmake
# Replace this pattern in RDC:
find_package(gRPC ${GRPC_DESIRED_VERSION} HINTS ${GRPC_ROOT} CONFIG REQUIRED)

# With standard CMake:
find_package(gRPC REQUIRED CONFIG)
# TheRock's package provision will ensure gRPC is found correctly
```

#### Option 2: Split Subprojects (Future Consideration)

If build complexity warrants, split into:

- `rdc-embedded`: Core libraries without gRPC
- `rdc-standalone`: Daemon/CLI with static gRPC

Currently unnecessary given RDC's existing conditional build support.

```
therock/
├── third-party/
│   └── grpc/
│       └── CMakeLists.txt       # Static gRPC build configuration
├── dctools/                     # New directory for datacenter tools
│   ├── CMakeLists.txt
│   └── rdc/
│       └── CMakeLists.txt       # RDC integration configuration
└── rocm-systems/
    └── projects/
        └── rdc/                 # Existing RDC source (submodule)
```

### Important Implementation Notes

1. **gRPC Static Build Confirmation**:

   - Building gRPC with `-DBUILD_SHARED_LIBS=OFF` will automatically build all dependencies (protobuf, abseil, re2, etc.) as static libraries when using the "module" provider
   - This has been confirmed to work correctly in gRPC v1.67.1

1. **Symbol Duplication Awareness**:

   - Recent gRPC versions (1.64.0+) have known issues with symbol duplication in static builds
   - Must use `-Wl,--exclude-libs=ALL` to prevent symbol pollution
   - Test thoroughly for "multiple definition" linker errors

1. **Future ODR Considerations**:

   - If protobuf, abseil, or other gRPC dependencies ever need standalone use in TheRock, they MUST be added as separate third-party dependencies
   - Current approach assumes gRPC is the sole consumer of these libraries
   - Document this constraint prominently in the gRPC third-party CMakeLists.txt

1. **RDC Insecure Mode Support**:

   - RDC extensively supports running without SSL/TLS via the `-u` flag
   - Common for development, testing, and trusted network deployments
   - SSL library still required at build time but not used at runtime in this mode
   - This reduces the practical impact of the SSL dependency

## Alternatives Considered

### Alternative 1: Vendor Shared gRPC Libraries

**Approach**: Bundle gRPC as shared libraries with custom SONAME and symbol versioning.

**Pros**:

- Smaller individual binary sizes
- Shared code between rdcd, rdci, and librdc_client.so
- Easier to update gRPC independently

**Cons**:

- Complex SONAME management to avoid conflicts
- Symbol versioning prone to errors
- RPATH complexity for finding bundled libraries
- Potential runtime conflicts with system gRPC

**Decision**: Rejected in favor of static linking for simplicity and portability

### Alternative 2: Use System gRPC

**Approach**: Require users to install gRPC from their distribution.

**Pros**:

- No vendoring needed
- Reduces distribution size
- Leverages system package management

**Cons**:

- Most distributions lack gRPC 1.67.1
- Version incompatibility issues
- Not portable across distributions
- Defeats purpose of distribution-neutral build

**Decision**: Rejected for portable distribution

### Alternative 3: Embedded-Only Mode

**Approach**: Build only embedded mode, exclude standalone entirely.

**Pros**:

- No gRPC dependency at all
- Tiny footprint (~2.2MB)
- Simple build and distribution

**Cons**:

- No daemon capability (rdcd)
- No CLI tool (rdci)
- No remote monitoring
- Limited to single-machine use cases

**Decision**: Rejected as it eliminates important datacenter use cases

## Related RFCs

- RFC0003: Build Tree Normalization - Establishes patterns for third-party dependencies
- RFC0005: Build hipDNN - Example of library integration into TheRock

## Migration Path

### For Existing System Package Users

No changes required. System packages continue to work as before. The TheRock build will produce compatible binaries that can be packaged using existing DEB/RPM infrastructure.

### For Containerized Deployments

The statically-linked binaries will be fully portable across Linux distributions with glibc ≥2.17, enabling simple tarball distribution for container images.

## Open Questions

1. **Module inclusion policy**: Which optional modules to include by default?

   - Recommendation: Include librdc_rocr.so by default, make librdc_rocp.so and librdc_rvs.so optional

1. **Binary size optimization**: Should we prioritize the "busy-box" optimization immediately?

   - Recommendation: Defer to post-initial implementation, ship working solution first

1. **BoringSSL vs OpenSSL sysdep**: Should we reconsider if SSL usage becomes more critical?

   - Current decision: BoringSSL for simplicity, given RDC's common use of insecure mode
   - Re-evaluate if other ROCm components need SSL/TLS functionality

## Summary

This RFC establishes the integration of RDC into TheRock with the following key decisions:

1. **Add gRPC to TheRock third-party** built as static libraries with BoringSSL
1. **Static link gRPC** into RDC standalone components with hidden visibility
1. **Use BoringSSL** bundled with gRPC to avoid OpenSSL dependency complexity
1. **Build all modes** (embedded and standalone) from the outset
1. **Use dctools/ directory** for datacenter tool organization
1. **Defer Python packaging** to future work
1. **Single monolithic build** initially, with option to split later if needed

The approach prioritizes portability and distribution simplicity over binary size, accepting larger executables (50MB each for rdcd/rdci/librdc_client.so) in exchange for avoiding runtime dependency management complexity. The use of BoringSSL provides a self-contained solution without system SSL dependencies, which is acceptable given RDC's extensive support for insecure mode in development environments.

## References

- [RDC Source Code](https://github.com/ROCm/rdc)
- [TheRock Repository](https://github.com/ROCm/TheRock)
- [gRPC Build Documentation](https://github.com/grpc/grpc/blob/master/BUILDING.md)
- [Manylinux Specification](https://github.com/pypa/manylinux)

## Revision History

- 2025-01-11: Initial draft with comprehensive gRPC analysis
- 2025-01-11: Revised to make static gRPC integration the plan of record, removed Python packaging
- 2025-01-11: Updated to use TheRock idiom for third-party deps, added BoringSSL decision for SSL
