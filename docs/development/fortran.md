# Fortran Support

TheRock can build a Flang-based Fortran toolchain on Linux and optionally wire
it into ROCm math-lib clients (wrappers, samples, tests). Windows builds leave
Flang disabled.

## How Flang Is Delivered

There are two mutually exclusive ways to get `lib/llvm/bin/flang`:

| Path | Flag | What builds |
|------|------|-------------|
| amd-llvm distribution (default) | `THEROCK_ENABLE_COMPILER_FORTRAN_DISTRIBUTION=ON` | Stages the LLVM `fortran` distribution from `amd-llvm` |
| Standalone (opt-in) | `THEROCK_ENABLE_FLANG_STANDALONE=ON` | `amd-llvm-flang` (Flang + MLIR; flang-rt and OpenMP Fortran modules via a runtimes ExternalProject) |

Host OpenMP (`libomp`) continues to come from the main `amd-llvm` / `runtimes`
build. The standalone path does **not** use a separate `amd-llvm-offload`
artifact for Flang runtime/modules.

Related artifacts / packaging:

- `amd-llvm` — base compiler (Clang, LLD, device libs, host OpenMP, …).
- `amd-llvm-flang` — standalone Flang driver, Fortran static libs, `libflang_rt*`,
  and `lib/llvm/lib/clang/*/finclude/**` (including `omp_lib.mod` / intrinsics).

## Master Switch

`THEROCK_ENABLE_FLANG` is the single on/off for “ship Flang”:

- **OFF** (also the Windows / no-COMPILER default): forces
  `THEROCK_ENABLE_FLANG_STANDALONE` and
  `THEROCK_ENABLE_COMPILER_FORTRAN_DISTRIBUTION` off, and forces
  `THEROCK_FLAG_BUILD_FORTRAN_LIBS` / `ROCM_BUILD_FORTRAN_LIBS` off.
- **ON**: exactly one delivery path must be enabled:
  - fortran distribution **or**
  - standalone
  Configure fails if both are off. If standalone is on, the fortran
  distribution option is forced off (do not build Flang twice).

When Flang is enabled, `base/aux-overlay` installs the public `bin/amdflang`
symlink and `flang.cfg` (points at `lib/llvm/bin/flang`).

## Build Flags

Defaults below are for a typical **Linux** configure with
`THEROCK_ENABLE_ALL=ON` (so `COMPILER` is on). On Windows, Flang-related
options are off.

Effective default Flang path on Linux with ALL: **amd-llvm fortran distribution**
(`FLANG=ON`, `FLANG_STANDALONE=OFF`, `COMPILER_FORTRAN_DISTRIBUTION=ON`).

| Flag | Default (Linux / ALL) | Default (Windows) | Role |
|------|----------------------|-------------------|------|
| `THEROCK_ENABLE_FLANG` | **ON** | **OFF** | Master enable for Flang (Linux + `COMPILER` only). |
| `THEROCK_ENABLE_FLANG_STANDALONE` | **OFF** | **OFF** | Build `amd-llvm-flang`. Opt-in via topology `feature_default = false`. Forced OFF if Flang is off. |
| `THEROCK_ENABLE_COMPILER_FORTRAN_DISTRIBUTION` | **ON** | **OFF** | Stage amd-llvm’s `fortran` distribution. ON when Flang is on and standalone is off; forced off when standalone is on. |
| `THEROCK_FLAG_BUILD_FORTRAN_LIBS` | **ON** | **OFF** | Build Fortran clients/wrappers/tests in subprojects. Forced OFF if Flang is off. |
| `ROCM_BUILD_FORTRAN_LIBS` | same as flag | same as flag | Same policy, mirrored into subprojects. |
| `FLANG_PARALLEL_COMPILE_JOBS` / `LLVM_PARALLEL_LINK_JOBS` | (unset) | (unset) | Cap Flang/LLVM memory use. |

Subprojects opt into the toolchain with `therock_cmake_subproject_declare`:

- `FORTRAN_OPTIONAL` — if `THEROCK_FLAG_BUILD_FORTRAN_LIBS=ON`, depend on Flang
  (`amd-llvm-flang` or `amd-llvm`) and set `CMAKE_Fortran_COMPILER` to staged
  `flang`; otherwise configure without a Fortran toolchain.
- `FORTRAN_REQUIRED` — fail configure if Fortran libs / Flang are unavailable.

Wired examples today: `rocBLAS`, `hipBLAS`, `rocSOLVER` (`BUILD_FORTRAN_CLIENTS`),
and `rocRAND` / `hipRAND` (`BUILD_FORTRAN_WRAPPER`). Project CMake then skips
Fortran samples/clients when those options are off; C/C++ clients are unchanged.

## Developer Builds

Flang and LLVM links are memory-heavy; keep job counts inside the machine’s
envelope.

Disable Flang (and Fortran math clients):

```bash
cmake -B build -S . -GNinja \
  -DTHEROCK_ENABLE_FLANG=OFF \
  -DTHEROCK_AMDGPU_FAMILIES=gfx1100 \
  -DLLVM_PARALLEL_LINK_JOBS=1
```

Default path — Flang via amd-llvm `fortran` distribution:

```bash
cmake -B build -S . -GNinja \
  -DTHEROCK_ENABLE_COMPILER=ON \
  -DTHEROCK_ENABLE_FLANG=ON \
  -DTHEROCK_ENABLE_FLANG_STANDALONE=OFF \
  -DTHEROCK_ENABLE_COMPILER_FORTRAN_DISTRIBUTION=ON \
  -DTHEROCK_FLAG_BUILD_FORTRAN_LIBS=ON \
  -DTHEROCK_AMDGPU_FAMILIES=gfx1100 \
  -DLLVM_PARALLEL_LINK_JOBS=1
```

Opt-in standalone Flang (`amd-llvm-flang`):

```bash
cmake -B build -S . -GNinja \
  -DTHEROCK_ENABLE_COMPILER=ON \
  -DTHEROCK_ENABLE_FLANG=ON \
  -DTHEROCK_ENABLE_FLANG_STANDALONE=ON \
  -DTHEROCK_FLAG_BUILD_FORTRAN_LIBS=ON \
  -DTHEROCK_AMDGPU_FAMILIES=gfx1100 \
  -DLLVM_PARALLEL_LINK_JOBS=1 \
  -DFLANG_PARALLEL_COMPILE_JOBS=32
```

After changing Flang or Fortran-client state for a downstream project, expunge
cached compiler discovery:

```bash
ninja -C build rocBLAS+expunge
ninja -C build rocBLAS
```

## Verification

When Flang is enabled:

- `build/dist/rocm/lib/llvm/bin/flang` exists.
- `build/dist/rocm/bin/amdflang` exists (aux-overlay symlink to
  `../lib/llvm/bin/flang`).
- `build/dist/rocm/lib/llvm/lib` (or per-target lib dirs) contains Flang runtime
  libraries such as `libflang_rt*`, `libFortran*`, `libflang*`.
- `build/dist/rocm/lib/llvm/lib/clang/*/finclude/**` contains intrinsic /
  OpenMP modules (e.g. `iso_c_binding.mod`, `omp_lib.mod`) when the standalone
  runtimes ExternalProject (or fortran distribution) installed them.

When Flang is disabled, those Flang binaries/modules and `bin/amdflang` should
be absent, and math-lib Fortran clients/samples should not build.

## Known Follow-Ups

- `lib/llvm/bin/amdflang` from `amdllvm` (`CLANG_ENABLE_AMDCLANG`) is still
  created with the other `amd*` symlinks unless gated separately (e.g.
  `CLANG_ENABLE_AMDFLANG` tied to `THEROCK_ENABLE_FLANG`).
- Other Fortran-aware projects (hipSOLVER, hipfort, …) can adopt the same
  `FORTRAN_OPTIONAL` + `BUILD_FORTRAN_*` mapping pattern.
- rocRAND / hipRAND may still install Fortran source trees under the package
  prefix for config compatibility; prefer fixing that upstream when possible.
