# Dependencies

TheRock manages two categories of third-party dependencies, distinguished by
their packaging behavior (see [`BUILD_TOPOLOGY.toml`](/BUILD_TOPOLOGY.toml) for
the build topology):

1. **Sysdeps** ([`third-party/sysdeps/`](/third-party/sysdeps/)) — libraries (zlib,
   elfutils, libdrm, numactl, etc.) built from source with SONAME rewriting and
   symbol versioning so that ROCm can ship private copies without conflicting with
   system-installed versions. This is the mechanism that enables portable
   distribution.
1. **Other third-party libraries** ([`third-party/`](/third-party/)) — libraries
   used as build dependencies that are not exposed externally and/or are not
   typically available as system packages (fmt, spdlog, flatbuffers,
   googletest, etc.). These are not given special packaging treatment. Most are
   `CORE` dependencies required by some subproject unconditionally while the
   `HOST_MATH` libraries (host-blas, SuiteSparse, fftw3) are optional.

The rest of this document covers the sysdeps in detail.

## Sysdeps

The ROCm projects have a number of dependencies. Typically those that escape
any specific library and are generally available as part of an OS distribution
are the concern of TheRock. In these cases, TheRock prefers to build them
all from source in such that:

- They are installed into the `lib/rocm_sysdeps` prefix.
- All ROCm libraries can find them by adding an appropriate relative RPATH.
- For symbol-versioned libraries, all symbols will be prefixed with
  `AMDROCM_SYSDEPS_1.0_`; whereas for non-versioned libraries, they will be
  built to version all symbols with `AMDROCM_SYSDEPS_1.0`.
- SONAMEs and semantic version symlink redirects are altered so that a single
  SONAME shared library with a prefix of `rocm_sysdeps_` is available to
  link against, using a `lib{originalname}.so` as a dev symlink.
- Any PackageConfig descriptors are altered to be location independent.
- PackageConfig and CMake `find_package` config files are advertised (being
  created as necessary) so that package resolution happens the same as if
  they were OS installed.

In order for this setup to work, a number of conventions need to be followed
project wide:

- Sub-projects should declare their use of a sysdep by including one or more of
  the global variables in their `RUNTIME_DEPS` (these will be empty if
  bundling is not enabled or supported for the target OS):
  - `THEROCK_BUNDLED_BZIP2`
  - `THEROCK_BUNDLED_ELFUTILS`
  - `THEROCK_BUNDLED_HWLOC`
  - `THEROCK_BUNDLED_LIBATOMIC`
  - `THEROCK_BUNDLED_LIBCAP`
  - `THEROCK_BUNDLED_LIBDRM`
  - `THEROCK_BUNDLED_LIBLZMA`
  - `THEROCK_BUNDLED_LIBMNL`
  - `THEROCK_BUNDLED_LIBNL`
  - `THEROCK_BUNDLED_NUMACTL`
  - `THEROCK_BUNDLED_SQLITE3`
  - `THEROCK_BUNDLED_UTIL_LINUX`
  - `THEROCK_BUNDLED_ZLIB`
  - `THEROCK_BUNDLED_ZSTD`
- Sub-projects must arrange for any libraries that depend on these to add the
  RPATH `lib/rocm_sysdeps/lib`.
- All projects should use the same package resolution technique (see below).

## Canonical Way to Depend

Per usual with CMake and the proliferation of operating systems, there is no
one true way to depend on a library. In general, if common distributions make
a library available via `find_package(foo CONFIG)`, we prefer that mechanism
be used consistently.

Implementation notes for each library is below:

## BZip2

- Canonical method: `find_package(BZip2)`
- Import library: `BZip2::BZip2`
- Alternatives: None (some OS vendors will provide alternatives but the source
  distribution of bzip2 has no opinion)

## Expat

- Canonical method: `find_package(expat)`
- Import library: `expat::expat`

## GMP

- Canonical method: `find_package(gmp)`
- Import library: `gmp::gmp`

## hwloc

- Canonical method: `find_package(hwloc CONFIG)`
- Import library: `hwloc::hwloc`

## ELFUTILS

Supported sub-libraries: `libelf`, `libdw`.

### libelf

- Canonical method: `find_package(LibElf)`
- Import library: `elf::elf`
- Alternatives: `pkg_check_modules(ELF libelf)`

### libdw

- Canonical method: `find_package(libdw)`
- Import library: `libdw::libdw`
- Alternatives: `pkg_check_modules(DW libdw)`

## libatomic

Provides the out-of-line atomic helper routines (`__atomic_*`, including the
16-byte `__atomic_*_16` variants) that the compiler cannot always lower inline
on x86-64. A few ROCm libraries (e.g. `librocprofiler-sdk-rocattach`,
`libroctracer_tool`) genuinely import these symbols and therefore carry a
`NEEDED libatomic.so.1`. Clean distros that do not ship libatomic (RHEL,
SLES, minimal images) cannot load those libraries unless it is bundled.

Unlike other sysdeps, libatomic is part of the compiler runtime rather than an
upstream project, so it is not fetched and built from source. Instead the
toolchain copy is located via `${CMAKE_C_COMPILER} -print-file-name=libatomic.so.1`
and staged into `lib/rocm_sysdeps/lib` with its original `libatomic.so.1`
SONAME (no `rocm_sysdeps_` prefix), because consumers reference it by that
name.

Because it is a prebuilt binary from the compiler runtime, libatomic is the
only sysdep that does not get the `rocm_sysdeps_` SONAME rewrite or the
`AMDROCM_SYSDEPS_1.0` symbol versioning that the other bundled libraries use.
This is intentional and safe: libatomic is ABI-stable and backward compatible,
and keeping the canonical SONAME is required for the consumers' `NEEDED` entry
to resolve.

### Licensing

libatomic is licensed **GPLv3 with the GCC Runtime Library Exception 3.1**.
Redistributing the unmodified runtime is permitted under that Exception when it
was produced by an Eligible Compilation Process — i.e. built by GCC. The sysdep
therefore refuses to configure unless `CMAKE_C_COMPILER` is GCC, so it always
ships the `gcc-toolset-13` build from the manylinux build container.

To satisfy GPLv3 the following are shipped to `lib/rocm_sysdeps/share/doc/libatomic/`:

- `COPYING.RUNTIME` — GCC Runtime Library Exception 3.1
- `COPYING3` — GNU GPL version 3
- `README.md` — points at the corresponding-source mirror

The corresponding source (GPLv3 §6) is the `gcc-toolset-13` source mirrored in
the `rocm-third-party-deps` S3 bucket. The pinned version must track the
`gcc-toolset-13` package installed in
`dockerfiles/build_manylinux_x86_64.Dockerfile`.

- Canonical method: none; consumers link `-latomic` implicitly via the
  compiler. Add `THEROCK_BUNDLED_LIBATOMIC` to `RUNTIME_DEPS` and ensure the
  `lib/rocm_sysdeps/lib` RPATH is present.

## libcap

Provides Linux capabilities for privileged operations (used by RDC).

- Canonical method: `find_package(Libcap)`
- Import library: `Libcap::Libcap`
- Alternatives: `pkg_check_modules(LIBCAP libcap)` or direct linking (used by RDC)

## libdrm

Supported sub-libraries: `libdrm`, `libdrm_amdgpu`

### libdrm

- Canonical method: `pkg_check_modules(DRM REQUIRED IMPORTED_TARGET libdrm)`
- Import library: `PkgConfig::DRM`
- Vars: `DRM_INCLUDE_DIRS`

### libdrm_amdgpu

- Canonical method: `pkg_check_modules(DRM_AMDGPU REQUIRED IMPORTED_TARGET libdrm_amdgpu)`
- Import library: `PkgConfig::DRM_AMDGPU`
- Vars: `DRM_AMDGPU_INCLUDE_DIRS`

## liblzma

- Canonical method: `find_package(LibLZMA)`
- Import library: `LibLZMA::LibLZMA`
- Alternatives: `pkg_check_modules(LZMA liblzma)`

## libmnl

Minimal netlink library for low-level netlink socket operations (used by amdsmi).

- Canonical method: `find_package(libmnl CONFIG)`
- Import library: `libmnl::libmnl`
- Alternatives: `pkg_check_modules(LIBMNL REQUIRED IMPORTED_TARGET libmnl)`
- Note: Provides minimalist netlink socket interface, typically used as a dependency for libnl

## libnl

Netlink protocol library suite providing high-level interfaces for Linux kernel netlink sockets (used by amdsmi).

Supported sub-libraries: `libnl` (core), `libnl-genl` (generic netlink)

### Core libnl

- Canonical method: `find_package(libnl CONFIG)`
- Import library: `libnl::libnl`
- Alternatives: `pkg_check_modules(LIBNL3 REQUIRED IMPORTED_TARGET libnl-3.0)`

### Generic netlink (libnl-genl)

- Canonical method: `find_package(libnl CONFIG)` (provides `libnl::genl`)
- Import library: `libnl::genl`
- Alternatives: `pkg_check_modules(LIBNL3_GENL REQUIRED IMPORTED_TARGET libnl-genl-3.0)`
- Dependencies: Automatically links `libnl::libnl`

## MPFR

- Canonical method: `find_package(mpfr)`
- Import library: `mpfr::mpfr`

## NCurses

- Canonical method: `find_package(ncurses)`
- Import library: `ncurses::ncurses`

### numactl

Provides the `libnuma` library. Tools are not included in bundled sysdeps.

- Canonical method: `find_package(NUMA)`
- Import library: `numa::numa`
- Vars: `NUMA_INCLUDE_DIRS`, `NUMA_INCLUDE_LIBRARIES` (can be used to avoid
  a hard-coded dep on `numa::numa`, which seems to vary across systems)
- Alternatives: `pkg_check_modules(NUMA numa)`

## simde

SIMDe (SIMD Everywhere) is a header-only portability library for SIMD intrinsics.

- Canonical method: `pkg_check_modules(simde REQUIRED IMPORTED_TARGET simde)`
- Import library: `PkgConfig::simde`
- Vars: `simde_INCLUDE_DIRS`
- Alternatives: none
- Note: Header-only library, provides portable SIMD intrinsics (SSE, AVX, NEON, etc.)

## sqlite3

- Canonical method: `find_package(SQLite3)`
- Import library: `SQLite::SQLite3`
- Alternatives: none

## util-linux

Provides the `libmount` (mount table parsing/manipulation) and `libblkid`
(block device identification) shared libraries. The full util-linux project
ships many more utilities and libraries, but TheRock only builds these two.

Note: `libmount` links `libblkid` transitively, so consumers that only use
`libmount` typically do not need to depend on `libblkid` directly.

### libmount

- Canonical method: `pkg_check_modules(LIBMOUNT REQUIRED IMPORTED_TARGET mount)`
- Import library: `PkgConfig::LIBMOUNT`
- Alternatives: none

### libblkid

- Canonical method: `pkg_check_modules(LIBBLKID REQUIRED IMPORTED_TARGET blkid)`
- Import library: `PkgConfig::LIBBLKID`
- Alternatives: none

## zlib

- Canonical method: `find_package(ZLIB)`
- Import library: `ZLIB::ZLIB`
- Alternatives: `pkg_check_modules(ZLIB zlib)`

## zstd

- Canonical method: `find_package(zstd)`
- Import library: `zstd::libzstd` (preferred - INTERFACE target that wraps the concrete library)
- Alternatives:
  - `zstd::libzstd_shared` - explicit shared library target
  - `pkg_check_modules(ZSTD libzstd)`
- Note: Upstream zstd's CMake install generates both `zstd::libzstd` (INTERFACE) and
  `zstd::libzstd_shared` (SHARED IMPORTED). The INTERFACE target forwards to the
  appropriate concrete target, abstracting static vs shared selection. Prefer
  `zstd::libzstd` for new code.
