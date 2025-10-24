# Dependencies

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
  - `THEROCK_BUNDLED_LIBDRM`
  - `THEROCK_BUNDLED_LIBLZMA`
  - `THEROCK_BUNDLED_NUMACTL`
  - `THEROCK_BUNDLED_SQLITE3`
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

### numactl

Provides the `libnuma` library. Tools are not included in bundled sysdeps.

- Canonical method: `find_package(NUMA)`
- Import library: `numa::numa`
- Vars: `NUMA_INCLUDE_DIRS`, `NUMA_INCLUDE_LIBRARIES` (can be used to avoid
  a hard-coded dep on `numa::numa`, which seems to vary across systems)
- Alternatives: `pkg_check_modules(NUMA numa)`

## sqlite3

- Canonical method: `find_package(SQLite3)`
- Import library: `SQLite::SQLite3`
- Alternatives: none

## zlib

- Canonical method: `find_package(ZLIB)`
- Import library: `ZLIB::ZLIB`
- Alternatives: `pkg_check_modules(ZLIB zlib)`

## zstd

- Canonical method: `find_package(zstd)`
- Import library: `zstd::libzstd_shared`
- Alternatives: `pkg_check_modules(ZSTD libzstd)`
