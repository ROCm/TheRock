# Bundled libatomic

This directory bundles `libatomic.so.1` from the build toolchain into the
`sysdeps` artifact (installed to `lib/rocm_sysdeps/lib/libatomic.so.1`).

## Why it is bundled

A few ROCm libraries emit out-of-line `__atomic_*` helper calls that the
compiler cannot lower inline on x86-64 (the generic size-argument
`__atomic_load` / `__atomic_store` / `__atomic_compare_exchange` forms, plus the
16-byte `__atomic_load_16` / `__atomic_store_16` variants). These leave a
genuine `NEEDED libatomic.so.1` in, at minimum,
`librocprofiler-sdk-rocattach.so.1` and `libroctracer_tool.so`. Distros that do
not ship libatomic (RHEL, SLES, minimal images) then fail to load them, so
`rocm-sdk test` fails on a clean system while `rocminfo` works.

libatomic is part of the compiler runtime rather than an upstream project we
build, so it is not fetched and compiled from source. The toolchain copy is
located via `${CMAKE_C_COMPILER} -print-file-name=libatomic.so.1` and staged
with its original `libatomic.so.1` SONAME (no `rocm_sysdeps_` prefix) so the
existing `$ORIGIN/rocm_sysdeps/lib` RPATH on consumers resolves it.

## Licensing and corresponding source

`libatomic` is part of GCC and is licensed under **GPLv3 with the GCC Runtime
Library Exception 3.1**. The Exception permits redistributing the runtime
library (including alongside otherwise-incompatible code) when it is built by an
Eligible Compilation Process. The bundled binary is the unmodified
`gcc-toolset-13` build from the manylinux_2_28 build container (see
`dockerfiles/build_manylinux_x86_64.Dockerfile`), so this condition is met.

- License texts are shipped next to the library:
  - `COPYING.RUNTIME` — GCC Runtime Library Exception 3.1
  - `COPYING3` — GNU GPL version 3
- **Corresponding source** for the redistributed binary (GPLv3 §6) is the
  `gcc-toolset-13` source, mirrored at:

  `https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/gcc-toolset-13-gcc-<VERSION>.src.rpm`

  This is the exact source of the binary shipped in the SDK. The `SOURCE_URL`
  in `CMakeLists.txt` must be kept in sync with the `gcc-toolset-13` version
  installed in the build container.

> Maintainer note: the mirrored source RPM must be uploaded to the
> `rocm-third-party-deps` bucket and the version pinned in `CMakeLists.txt`
> before this can ship. See the SOURCE_URL / SOURCE_VERSION variables.
